# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 Engram projection and residual gating from host rows or local lookup tables."""

import math
from pathlib import Path

import pypto.language as pl
from pypto.runtime import pto_isa_include_dir
import torch

from config import FLASH as M, T_DYN
from gemm import gemm_bf16_via_mxfp8_to_bf16_staged


# model config
D = M.hidden_size
HC_MULT = M.hc_mult


N_HASH_COLS = (M.engram_max_ngram_size - 1) * M.engram_n_heads  # 24
HEAD_DIM = M.engram_head_dim  # 256
ENGRAM_K = N_HASH_COLS * HEAD_DIM  # 6144 flattened lookup width
KV_OUT = (HC_MULT + 1) * D  # 5 * 5120 = 25600
CLAMP = 1e-6
NORM_EPS = M.rms_norm_eps
D_INV = 1.0 / D
DOT_SCALE = D**-0.5

# PyPTO comparison tiling.
GATE_T_TILE = 8
GATE_D_TILE = 1280
PROJECTION_N_TILE = 256
PROJECTION_K_TILE = 256


@pl.jit.extern(core_type="aiv", source=Path(__file__).parent / "kernels/engram.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _engram_quantize(lookup: pl.Tensor, payload: pl.Out[pl.Tensor], scales: pl.Out[pl.Tensor],
                      tokens: pl.Scalar[pl.INT32], padded: pl.Scalar[pl.INDEX],
                      mode: pl.Scalar[pl.INDEX]) -> tuple[pl.Tensor, pl.Tensor]:
    """Write FP8 activation payload and packed E8M0 scales."""
    return payload, scales


@pl.jit.extern(core_type="aic", source=Path(__file__).parent / "kernels/engram.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _engram_project(payload: pl.Tensor, weight: pl.Tensor, scales: pl.Tensor, weight_scale: pl.Tensor,
                    kv: pl.Out[pl.Tensor], tokens: pl.Scalar[pl.INDEX], row_tile: pl.Scalar[pl.INDEX]) -> pl.Tensor:
    """Project quantized rows into BF16 keys and values."""
    return kv


@pl.jit.extern(core_type="aiv", source=Path(__file__).parent / "kernels/engram.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _engram_gate_native(kv: pl.Tensor, weight: pl.Tensor, x: pl.Tensor, out: pl.Out[pl.Tensor],
                        tokens: pl.Scalar[pl.INDEX], mode: pl.Scalar[pl.INDEX]) -> pl.Tensor:
    """Compute the normalized-dot gate and update the residual."""
    return out


@pl.jit.inline(auto_scope=False)
def engram_gate(
    kv: pl.Tensor, weight: pl.Tensor, x: pl.Tensor, out: pl.Tensor,
    lookup: pl.Tensor, mask_zero_rows: pl.constexpr,
):
    """Run native gating and preserve zero-lookup FP32 residual rows."""
    tokens = pl.tensor.dim(x, 0)
    if mask_zero_rows:
        original = pl.create_tensor([tokens, HC_MULT * D], dtype=pl.FP32)
        hidden = pl.reshape(x, [tokens, HC_MULT * D])
        with pl.spmd(tokens, name_hint="engram_save_residual") as saved_ready:
            row = pl.tile.get_block_idx()
            for column in pl.range(0, HC_MULT * D, 512):
                value = pl.load(hidden, [row, column], [1, 512])
                pl.store(value, [row, column], original)
        with pl.spmd(pl.system.available_aiv_count(), name_hint="engram_gate", deps=[saved_ready]):
            result = _engram_gate_native(kv, weight, x, out, tokens, 3)
        output = pl.reshape(result, [tokens, HC_MULT * D])
        for block in pl.spmd((tokens + 7) // 8, name_hint="engram_restore_masked"):
            row = block * 8
            valid_rows = pl.min(8, tokens - row)
            total = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
            reduction_tmp = pl.create_tile([8, 256], dtype=pl.FP32)
            for column in pl.range(0, ENGRAM_K, 256):
                source = pl.load(lookup, [row, column], [8, 256], valid_shape=[valid_rows, 256])
                padded = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 8, 256)
                source_fp32 = pl.cast(padded, pl.FP32)
                part = pl.reshape(pl.row_sum(pl.abs(source_fp32), reduction_tmp), [1, 8])
                total = pl.add(total, part)
            ones = pl.tile.full([8, 512], dtype=pl.FP32, value=1.0)
            active = pl.cmps(pl.row_expand_mul(ones, pl.reshape(total, [8, 1])), 0.0, cmp_type=4)
            for column in pl.range(0, HC_MULT * D, 512):
                saved = pl.load(original, [row, column], [8, 512], valid_shape=[valid_rows, 512])
                updated = pl.load(output, [row, column], [8, 512], valid_shape=[valid_rows, 512])
                restored = pl.tile.select(active, pl.set_validshape(updated, 8, 512), pl.set_validshape(saved, 8, 512))
                pl.store(pl.set_validshape(restored, valid_rows, 512), [row, column], output)
    else:
        with pl.spmd(pl.system.available_aiv_count(), name_hint="engram_gate"):
            result = _engram_gate_native(kv, weight, x, out, tokens, 3)
    return out


@pl.jit.inline(auto_scope=False)
def _engram_rows_tile(
    lookup: pl.Tensor, wkv: pl.Tensor, wkv_scale: pl.Tensor, weight: pl.Tensor,
    x: pl.Tensor, out: pl.Tensor, tokens: pl.Scalar[pl.INT32],
    row_tile: pl.constexpr, mask_zero_rows: pl.constexpr,
):
    """Quantize and project host rows before gating the residual."""
    padded = ((tokens + row_tile - 1) // row_tile) * row_tile
    payload = pl.create_tensor([padded, ENGRAM_K], dtype=pl.FP8E4M3FN)
    scales = pl.create_tensor([1, padded * (ENGRAM_K // 32)], dtype=pl.FP8E8M0)
    kv = pl.create_tensor([tokens, KV_OUT], dtype=pl.BF16)
    with pl.spmd(pl.system.available_aiv_count(), name_hint="engram_quant"):
        quantized, quantized_scales = _engram_quantize(lookup, payload, scales, tokens, padded, 1)
    with pl.spmd(pl.system.available_cluster_count(), name_hint="engram_projection"):
        projected = _engram_project(quantized, wkv, quantized_scales, wkv_scale, kv, tokens, row_tile)
    return engram_gate(projected, weight, x, out, lookup, mask_zero_rows)


@pl.jit.inline(auto_scope=False)
def engram_from_rows(
    lookup: pl.Tensor,
    wkv_weight: pl.Tensor[[ENGRAM_K, KV_OUT], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[ENGRAM_K // 32, KV_OUT], pl.FP8E8M0],
    weight: pl.Tensor[[HC_MULT, D], pl.FP32],
    x: pl.Tensor,
    out: pl.Tensor,
    mask_zero_rows: pl.constexpr = False,
):
    """Project host-gathered ND rows and gate the FP32 residual in place or out of place."""
    tokens = pl.tensor.dim(x, 0)
    if tokens > 0:
        if tokens <= 80:
            result = _engram_rows_tile(
                lookup, wkv_weight, wkv_scale, weight, x, out, tokens, 80, mask_zero_rows,
            )
        elif tokens <= 128:
            result = _engram_rows_tile(
                lookup, wkv_weight, wkv_scale, weight, x, out, tokens, 128, mask_zero_rows,
            )
        else:
            result = _engram_rows_tile(
                lookup, wkv_weight, wkv_scale, weight, x, out, tokens, 256, mask_zero_rows,
            )
    else:
        result = out
    return result


@pl.jit.inline
def _engram_gate_pypto(
    kv: pl.Tensor,
    weight: pl.Tensor[[HC_MULT, D], pl.FP32],
    x: pl.Tensor,
    out: pl.Tensor,
):
    """Gate projected keys and values into each residual stream copy."""
    t_dim = pl.tensor.dim(kv, 0)
    x_flat = pl.reshape(x, [t_dim, HC_MULT * D])
    out_flat = pl.reshape(out, [t_dim, HC_MULT * D])
    t_blocks = (t_dim + GATE_T_TILE - 1) // GATE_T_TILE
    workers = pl.min(t_blocks * HC_MULT, pl.system.available_aiv_count())
    for worker in pl.spmd(workers, name_hint="_engram_gate_pypto"):
        for wave in pl.pipeline((t_blocks * HC_MULT + workers - 1) // workers, stage=2):
            block = wave * workers + worker
            if block < t_blocks * HC_MULT:
                t0 = (block // HC_MULT) * GATE_T_TILE
                c = block % HC_MULT
                valid_rows = pl.min(GATE_T_TILE, t_dim - t0)
                reduction_tmp = pl.create_tile([GATE_T_TILE, GATE_D_TILE], dtype=pl.FP32)
                # Reduction accumulators have 32-byte aligned rows.
                sq_x = pl.tile.full([1, GATE_T_TILE], dtype=pl.FP32, value=0.0)
                sq_k = pl.tile.full([1, GATE_T_TILE], dtype=pl.FP32, value=0.0)
                dot = pl.tile.full([1, GATE_T_TILE], dtype=pl.FP32, value=0.0)
                for d0 in pl.range(0, D, GATE_D_TILE):
                    x_raw = pl.load(x_flat, [t0, c * D + d0], [GATE_T_TILE, GATE_D_TILE], valid_shape=[valid_rows, GATE_D_TILE])
                    key_raw = pl.load(kv, [t0, c * D + d0], [GATE_T_TILE, GATE_D_TILE], valid_shape=[valid_rows, GATE_D_TILE])
                    key_d = pl.cast(pl.set_validshape(pl.fillpad(key_raw, pad_value=pl.PadValue.zero), GATE_T_TILE, GATE_D_TILE), pl.FP32)
                    x_d = pl.set_validshape(pl.fillpad(x_raw, pad_value=pl.PadValue.zero), GATE_T_TILE, GATE_D_TILE)
                    sq_x = pl.add(sq_x, pl.reshape(pl.row_sum(pl.mul(x_d, x_d), reduction_tmp), [1, GATE_T_TILE]))
                    sq_k = pl.add(sq_k, pl.reshape(pl.row_sum(pl.mul(key_d, key_d), reduction_tmp), [1, GATE_T_TILE]))
                    w_d = pl.load(weight, [c, d0], [1, GATE_D_TILE])
                    xw_d = pl.col_expand_mul(x_d, w_d)
                    dot = pl.add(dot, pl.reshape(pl.row_sum(pl.mul(xw_d, key_d), reduction_tmp), [1, GATE_T_TILE]))
                inv_x = pl.rsqrt(pl.add(pl.mul(sq_x, D_INV), NORM_EPS))
                inv_k = pl.rsqrt(pl.add(pl.mul(sq_k, D_INV), NORM_EPS))
                dot = pl.mul(pl.mul(pl.mul(dot, inv_x), inv_k), DOT_SCALE)
                root = pl.sqrt(pl.maximum(pl.abs(dot), CLAMP))
                nonnegative = pl.cmps(dot, 0.0, cmp_type=5)
                signed = pl.tile.select(nonnegative, root, pl.neg(root))
                gate = pl.reshape(pl.recip(pl.add(pl.exp(pl.neg(signed)), 1.0)), [GATE_T_TILE, 1])
                for d0 in pl.range(0, D, GATE_D_TILE):
                    output_source = pl.load(x_flat, [t0, c * D + d0], [GATE_T_TILE, GATE_D_TILE], valid_shape=[valid_rows, GATE_D_TILE])
                    output_value = pl.load(kv, [t0, HC_MULT * D + d0], [GATE_T_TILE, GATE_D_TILE], valid_shape=[valid_rows, GATE_D_TILE])
                    gated = pl.row_expand_mul(pl.cast(output_value, pl.FP32), gate)
                    y = pl.add(output_source, gated)
                    y_bf16 = pl.cast(y, target_type=pl.BF16, mode="rint")
                    y_rounded = pl.cast(y_bf16, pl.FP32)
                    pl.store(pl.set_validshape(y_rounded, valid_rows, GATE_D_TILE), [t0, c * D + d0], out_flat)
    return out


@pl.jit.inline(auto_scope=False)
def engram_from_rows_pypto(
    lookup: pl.Tensor,
    wkv_weight: pl.Tensor[[ENGRAM_K, KV_OUT], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[ENGRAM_K // 32, KV_OUT], pl.FP8E8M0],
    weight: pl.Tensor[[HC_MULT, D], pl.FP32],
    x: pl.Tensor,
    out: pl.Tensor,
):
    """Compute Engram in PyPTO DSL for explicit comparison with the production entry."""
    tokens = pl.tensor.dim(x, 0)
    if tokens > 0:
        kv = pl.create_tensor([tokens, KV_OUT], dtype=pl.BF16)
        projected = gemm_bf16_via_mxfp8_to_bf16_staged(
            lookup, wkv_weight, wkv_scale, kv, tokens,
            ENGRAM_K, KV_OUT, "engram_pypto_projection", "engram_pypto_quant",
            PROJECTION_N_TILE, PROJECTION_K_TILE, 1, 2147483647,
        )
        _engram_gate_pypto(projected, weight, x, out)
    return out


class HostEngram:
    """Hash text history and read selected FP8 checkpoint table rows on the CPU.

    Reuse one instance across prefill chunks and decode calls. Each request_id
    owns its compressed-token history; start_pos=0 resets that request, and
    continuation requires the preceding history. False token_mask entries exclude
    image positions and stop n-grams from crossing image spans.
    """

    def __init__(self, checkpoint, tokenizer):
        import numpy as np
        from tokenizers import Regex, normalizers

        self.checkpoint = checkpoint
        sentinel = "\ue000"
        normalizer = normalizers.Sequence([
            normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(), normalizers.Replace(sentinel, " "),
        ])
        keys = {}
        token_map = []
        for token in range(tokenizer.get_vocab_size(with_added_tokens=True)):
            text = tokenizer.decode([token], skip_special_tokens=False)
            key = tokenizer.id_to_token(token) if "\ufffd" in text else normalizer.normalize_str(text) or text
            token_map.append(keys.setdefault(key, len(keys)))
        if len(keys) != M.engram_compressed_vocab_size:
            raise ValueError(f"Engram compressed vocabulary has {len(keys)} entries")
        self.token_map = torch.tensor(token_map, dtype=torch.int64)
        self.pad_id = token_map[M.engram_pad_token_id]
        self.history = {}
        primes, seen = [], set()
        for layer in M.engram_layer_ids:
            buckets = []
            for _ in range(M.engram_max_ngram_size - 1):
                candidate = M.engram_vocab_size - 1
                for _ in range(M.engram_n_heads):
                    candidate += 1
                    while candidate in seen or any(candidate % d == 0 for d in range(2, math.isqrt(candidate) + 1)):
                        candidate += 1
                    seen.add(candidate)
                    buckets.append(candidate)
            primes.append(buckets)
        if tuple(map(sum, primes)) != M.engram_num_embeddings:
            raise ValueError("Engram prime buckets do not match checkpoint table sizes")
        self.primes = torch.tensor(primes, dtype=torch.int64)
        self.offsets = self.primes.cumsum(-1) - self.primes
        bound = max(1, (np.iinfo(np.int64).max // len(keys)) // 2)
        self.multipliers = torch.stack([
            torch.from_numpy(np.random.default_rng(10007 * layer).integers(
                0, bound, size=M.engram_max_ngram_size, dtype=np.int64,
            ) * 2 + 1)
            for layer in M.engram_layer_ids
        ])

    def hash_ids(self, token_ids, start_pos=0, *, request_id=0, token_mask=None):
        """Return [layer, token, hash column] IDs and commit the current text history."""
        ids = torch.as_tensor(token_ids, device="cpu")
        if ids.numel() and ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Engram token IDs must have an integer dtype")
        ids = ids.long()
        if ids.ndim != 1 or bool(((ids < 0) | (ids >= self.token_map.numel())).any()):
            raise ValueError("Engram requires a one-dimensional sequence of valid token IDs")
        history = self.history.get(request_id, torch.empty(0, dtype=torch.int64))
        if isinstance(start_pos, bool) or not isinstance(start_pos, int) or not 0 <= start_pos <= history.numel():
            raise ValueError("Engram continuation requires all preceding token history")
        current = self.token_map[ids]
        if token_mask is not None:
            mask = torch.as_tensor(token_mask, device="cpu")
            if mask.dtype != torch.bool or mask.shape != ids.shape:
                raise ValueError("Engram token_mask must be a boolean vector matching token IDs")
            current = torch.where(mask, current, -1)
        compressed = torch.cat((history[:start_pos], current))
        positions = torch.arange(start_pos, compressed.numel())
        lookback = []
        blocked = torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(M.engram_max_ngram_size):
            source = (positions - shift).clamp_min(0)
            values = compressed[source]
            blocked |= (positions < shift) | (values == -1)
            lookback.append(torch.where(blocked, self.pad_id, values))
        tokens = torch.stack(lookback, dim=-1)
        products = tokens.unsqueeze(0) * self.multipliers.unsqueeze(1)
        rolling, hashes = products[..., 0], []
        for shift in range(1, M.engram_max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., shift])
            begin = (shift - 1) * M.engram_n_heads
            primes = self.primes[:, begin:begin + M.engram_n_heads].unsqueeze(1)
            hashes.append(rolling.unsqueeze(-1) % primes)
        self.history[request_id] = compressed
        return torch.cat(hashes, dim=-1) + self.offsets.unsqueeze(1)

    def lookup(self, token_ids, capacity, start_pos=0, *, request_id=0, token_mask=None):
        """Read current rows and group-32 E8M0 scales into [2, capacity, 6144] BF16.

        Tables stay in host checkpoint files; only selected rows are read.
        Unused token rows are zero-padded.
        """
        from quant import decode_e8m0

        if len(token_ids) > capacity or capacity <= 0:
            raise ValueError("Engram rows must fit the current-token capacity")
        hashes = self.hash_ids(token_ids, start_pos, request_id=request_id, token_mask=token_mask)
        active = torch.ones(len(token_ids), dtype=torch.bool)
        if token_mask is not None:
            active = torch.as_tensor(token_mask, device="cpu")
        layers = len(M.engram_layer_ids)
        width = self.primes.shape[1] * M.engram_head_dim
        result = torch.zeros((layers, capacity, width), dtype=torch.bfloat16)
        for slot, layer in enumerate(M.engram_layer_ids):
            prefix = f"layers.{layer}.engram.embed."
            if not bool(active.any()):
                continue
            rows = self.checkpoint.tensor_rows(prefix + "weight", hashes[slot, active])
            scales = self.checkpoint.tensor_rows(prefix + "scale", hashes[slot, active])
            if rows.dtype != torch.float8_e4m3fn or rows.shape[-1] != M.engram_head_dim:
                raise ValueError(f"unsupported native Engram table layout: {prefix}weight")
            if scales.dtype != torch.float8_e8m0fnu or scales.shape[-1] != M.engram_head_dim // 32:
                raise ValueError(f"unsupported native Engram table scales: {prefix}scale")
            values = rows.float() * decode_e8m0(scales).repeat_interleave(32, dim=-1)
            result[slot, :len(token_ids)][active] = values.flatten(1).to(torch.bfloat16)
        return result


def build_masked_gate_specs(tokens):
    """Build mixed text/image rows with non-BF16 FP32 residual values."""
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(9)
    lookup = torch.randn(tokens, ENGRAM_K, generator=generator).bfloat16()
    kv = torch.randn(tokens, KV_OUT, generator=generator).bfloat16()
    lookup[1::3] = 0
    kv[1::3] = 0
    hidden = torch.randn(tokens, HC_MULT, D, generator=generator)
    return [
        TensorSpec("kv", [tokens, KV_OUT], torch.bfloat16, init_value=kv),
        TensorSpec("weight", [HC_MULT, D], torch.float32, init_value=1.0),
        TensorSpec("x", [tokens, HC_MULT, D], torch.float32, init_value=hidden),
        TensorSpec("lookup", [tokens, ENGRAM_K], torch.bfloat16, init_value=lookup),
        TensorSpec("out", [tokens, HC_MULT, D], torch.float32),
    ]


def golden_masked_gate(tensors):
    """Evaluate gated text rows and exact image-row residual passthrough."""
    hidden = tensors["x"]
    keys = tensors["kv"][:, :HC_MULT * D].float().reshape(-1, HC_MULT, D)
    value = tensors["kv"][:, HC_MULT * D:].float()
    dot = (hidden * tensors["weight"] * keys).sum(-1)
    dot *= torch.rsqrt(hidden.square().mean(-1) + NORM_EPS)
    dot *= torch.rsqrt(keys.square().mean(-1) + NORM_EPS) * DOT_SCALE
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(CLAMP).sqrt(), dot))
    expected = (hidden + gate.unsqueeze(-1) * value[:, None]).bfloat16().float()
    masked = tensors["lookup"].count_nonzero(dim=-1) == 0
    expected[masked] = hidden[masked]
    tensors["out"].copy_(expected)


def compare_masked_gate(actual, expected, *, inputs, **kwargs):
    """Require exact masked residuals and the existing Engram text comparator."""
    from golden import ratio_allclose

    masked = inputs["lookup"].count_nonzero(dim=-1) == 0
    if not torch.equal(actual[masked], inputs["x"][masked]):
        return False, "masked FP32 residual changed"
    if bool(masked.all()):
        return True, ""
    return ratio_allclose(atol=1e-3, rtol=1e-2)(actual[~masked], expected[~masked], inputs=inputs, **kwargs)


@pl.jit
def l2_engram_masked_gate(
    kv: pl.Tensor[[T_DYN, KV_OUT], pl.BF16],
    weight: pl.Tensor[[HC_MULT, D], pl.FP32],
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    lookup: pl.Tensor[[T_DYN, ENGRAM_K], pl.BF16],
    out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    """Validate masked host-lookup gating over FP32 residual streams."""
    kv.bind_dynamic(0, T_DYN)
    x.bind_dynamic(0, T_DYN)
    lookup.bind_dynamic(0, T_DYN)
    out.bind_dynamic(0, T_DYN)
    engram_gate(kv, weight, x, out, lookup, mask_zero_rows=True)
    return out


@pl.jit
def l2_engram_rows(
    lookup: pl.Tensor[[T_DYN, ENGRAM_K], pl.BF16],
    wkv_weight: pl.Tensor[[ENGRAM_K, KV_OUT], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[ENGRAM_K // 32, KV_OUT], pl.FP8E8M0],
    weight: pl.Tensor[[HC_MULT, D], pl.FP32],
    x: pl.InOut[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    """Validate the same in-place entry used by decode and prefill forwards."""
    lookup.bind_dynamic(0, T_DYN)
    x.bind_dynamic(0, T_DYN)
    engram_from_rows(lookup, wkv_weight, wkv_scale, weight, x, x, False)
    return x


@pl.jit
def l2_engram_rows_pypto(
    lookup: pl.Tensor[[T_DYN, ENGRAM_K], pl.BF16],
    wkv_weight: pl.Tensor[[ENGRAM_K, KV_OUT], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[ENGRAM_K // 32, KV_OUT], pl.FP8E8M0],
    weight: pl.Tensor[[HC_MULT, D], pl.FP32],
    x: pl.InOut[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    """Validate the explicit PyPTO comparison implementation in place."""
    lookup.bind_dynamic(0, T_DYN)
    x.bind_dynamic(0, T_DYN)
    engram_from_rows_pypto(lookup, wkv_weight, wkv_scale, weight, x, x)
    return x


def build_engram_rows_specs(tokens: int):
    """Build deterministic host-row inputs with nonuniform packed MX scales."""
    from golden import TensorSpec
    from quant import pack_mx_b_scale

    generator = torch.Generator().manual_seed(3)

    def init_lookup():
        return (torch.randn(tokens, ENGRAM_K, generator=generator) * 0.2).bfloat16()

    def init_wkv():
        value = (torch.randn(ENGRAM_K, KV_OUT, generator=generator) * 0.02).to(torch.float8_e4m3fn)
        return value

    def init_scale():
        codes = torch.randint(120, 128, (ENGRAM_K // 32, KV_OUT), generator=generator, dtype=torch.uint8)
        return pack_mx_b_scale(codes).view(torch.float8_e8m0fnu)

    def init_weight():
        return torch.rand(HC_MULT, D, generator=generator) + 0.5

    def init_x():
        return torch.randn(tokens, HC_MULT, D, generator=generator)

    return [
        TensorSpec("lookup", [tokens, ENGRAM_K], torch.bfloat16, init_value=init_lookup),
        TensorSpec("wkv_weight", [ENGRAM_K, KV_OUT], torch.float8_e4m3fn, init_value=init_wkv),
        TensorSpec("wkv_scale", [ENGRAM_K // 32, KV_OUT], torch.float8_e8m0fnu, init_value=init_scale),
        TensorSpec("weight", [HC_MULT, D], torch.float32, init_value=init_weight),
        TensorSpec("x", [tokens, HC_MULT, D], torch.float32, init_value=init_x),
    ]


def golden_engram_rows_case(tensors):
    """Apply independent sequential MX projection and the released gate formula."""
    from gemm import golden_gemm_bf16_via_mxfp8_to_bf16_sequential

    wkv = tensors["wkv_weight"]
    kv = golden_gemm_bf16_via_mxfp8_to_bf16_sequential(tensors["lookup"], wkv, tensors["wkv_scale"])
    key = kv[:, :HC_MULT * D].float().reshape(-1, HC_MULT, D)
    value = kv[:, HC_MULT * D:].float()
    h = tensors["x"].float()
    inverse = torch.rsqrt(h.square().mean(-1) + NORM_EPS) * torch.rsqrt(key.square().mean(-1) + NORM_EPS)
    dot = (h * tensors["weight"] * key).sum(-1) * inverse * DOT_SCALE
    root = dot.abs().clamp_min(CLAMP).sqrt()
    gate = torch.sigmoid(torch.where(dot >= 0, root, -root))
    tensors["x"][:] = (h + gate.unsqueeze(-1) * value.unsqueeze(-2)).bfloat16().float()


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 Engram validation")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="0",
        help="comma-separated device ids; provide one device id",
    )
    parser.add_argument("--masked-gate", action="store_true")
    parser.add_argument("--implementation", choices=["cpp", "pypto"], default="cpp",
                        help="cpp is the production default; pypto is for explicit comparison")
    parser.add_argument("--tokens", type=int, help="explicit row count; default checks tile boundaries and tails")
    parser.add_argument("--runtime-dir", help="reuse a CPU-prebuilt runtime; requires --tokens")
    parser.add_argument("--golden-data", help="reuse frozen input/reference tensors; requires --tokens")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    if len(device_ids) != 1:
        raise ValueError(f"need exactly one device id, got {device_ids}")

    if args.tokens is not None and args.tokens <= 0:
        parser.error("--tokens must be positive")
    if (args.runtime_dir or args.golden_data or args.save_data) and args.tokens is None:
        parser.error("replay/capture options require an explicit --tokens case")
    if args.runtime_dir:
        from pypto.runtime._artifact_sources import read_kernel_config

        kernels = read_kernel_config(Path(args.runtime_dir) / "kernel_config.py").KERNELS
        native_runtime = any(kernel.get("external", False) for kernel in kernels)
        if native_runtime != (args.implementation == "cpp"):
            parser.error("--runtime-dir implementation does not match --implementation")
    entry = l2_engram_rows if args.implementation == "cpp" else l2_engram_rows_pypto
    token_cases = [17, 81, 129, 256] if args.execution_mode else [17, 80, 81, 128, 129, 256, 257]
    if args.tokens is not None:
        token_cases = [args.tokens]
    for tokens in ([] if args.masked_gate else token_cases):
        specs = build_engram_rows_specs(tokens)
        result = run(
            fn=entry,
            specs=specs,
            golden_fn=golden_engram_rows_case,
            config={
                "backend_type": args.backend_type, "execution_mode": args.execution_mode,
                "device_id": device_ids[0],
                "enable_chip_swimlane": args.enable_chip_swimlane,
            },
            rtol=1e-3,
            atol=1e-3,
            compare_fn={"x": ratio_allclose(atol=1e-3, rtol=1e-2)},
            compile_only=args.compile_only,
            runtime_dir=args.runtime_dir,
            golden_data=args.golden_data,
            save_data=args.save_data,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)

    result = run(
        fn=l2_engram_masked_gate,
        specs=build_masked_gate_specs(8),
        golden_fn=golden_masked_gate,
        config={"backend_type": args.backend_type, "execution_mode": args.execution_mode,
                "device_id": device_ids[0]},
        compare_fn={"out": compare_masked_gate},
        compile_only=args.compile_only,
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
