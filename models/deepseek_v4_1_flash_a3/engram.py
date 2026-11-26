# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 Engram host lookup, projection, and residual gating."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"

import math

import pypto.language as pl
import torch

from .config import FLASH as M
from .dsl_types import T_DYN
from .gemm import PREFILL_BF16_K_SUBTILE
from .weights import unpack_bf16_nz


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

# tiling
T_TILE = 16
K_TILE = 128
N_TILE = 256
DECODE_ENGRAM_K_TILE = 256
DECODE_ENGRAM_N_TILE = 128
PREFILL_M_TILE = 128
PREFILL_MIN_ROWS = 1024
D_TILE = 512
ROT_TILE = 32
ROT_M_TILE = 64
ROT_WORKER_TILE = 24


@pl.jit.inline
def engram_cast_to_fp32(source: pl.Tensor, output: pl.Tensor):
    """Widen a BF16 matrix into FP32 tiles."""
    rows = pl.tensor.dim(source, 0)
    width = pl.tensor.dim(source, 1)
    blocks = (rows + T_TILE - 1) // T_TILE * (width // N_TILE)
    workers = pl.max(1, pl.min(pl.min(blocks, 32), pl.system.available_aiv_count()))
    for worker in pl.spmd(workers, name_hint="engram_widen"):
        for block in pl.range(worker, blocks, workers):
            t0 = block // (width // N_TILE) * T_TILE
            n0 = block % (width // N_TILE) * N_TILE
            valid_rows = pl.min(T_TILE, rows - t0)
            value = pl.slice(source, [T_TILE, N_TILE], [t0, n0], valid_shape=[valid_rows, N_TILE])
            widened = pl.cast(value, pl.FP32)
            output[t0:t0 + T_TILE, n0:n0 + N_TILE] = pl.set_validshape(widened, valid_rows, N_TILE)
    return output


@pl.jit.inline(auto_scope=False)
def engram_unrotate(
    source: pl.Tensor,
    rotation: pl.Tensor[[D // 32, 32, 32], pl.FP32],
    output: pl.Tensor,
):
    """Map rotated residual streams into the original Engram key and gate basis."""
    tokens = pl.tensor.dim(source, 0)
    source_flat = pl.reshape(source, [tokens, HC_MULT * D])
    output_flat = pl.reshape(output, [tokens, HC_MULT * D])
    rotation_flat = pl.reshape(rotation, [D, 32])
    tasks = (tokens + 15) // 16 * HC_MULT * (D // 32)
    workers = pl.max(1, pl.min(pl.min(tasks, 32), pl.system.available_aiv_count()))
    for worker in pl.spmd(workers, name_hint="engram_unrotate"):
        for task in pl.range(worker, tasks, workers):
            block = task % (D // 32)
            stream = task // (D // 32) % HC_MULT
            row = task // ((D // 32) * HC_MULT) * 16
            valid_rows = pl.min(16, tokens - row)
            column = stream * D + block * 32
            value = pl.load(source_flat, [row, column], [16, 32], valid_shape=[valid_rows, 32])
            result = pl.tile.full([32, 16], dtype=pl.FP32, value=0.0)
            temporary = pl.create_tile([16, 32], dtype=pl.FP32)
            for component in pl.unroll(32):
                coefficients = pl.load(rotation_flat, [block * 32 + component, 0], [1, 32])
                product = pl.col_expand_mul(value, coefficients)
                summed = pl.reshape(pl.row_sum(product, tmp_tile=temporary), [1, 16])
                result = pl.tile.assemble(result, summed, [component, 0])
            row_major = pl.tile.transpose(result, 0, 1)
            output_flat = pl.store(pl.set_validshape(row_major, valid_rows, 32), [row, column], output_flat)
    return output


@pl.jit.inline(auto_scope=False)
def prefill_engram_unrotate(source: pl.Tensor, rotation: pl.Tensor, output: pl.Tensor):
    """Apply each FP32 rotation block to a panel of residual streams on Cube."""
    rows = pl.tensor.dim(source, 0) * HC_MULT
    source_flat = pl.reshape(source, [rows, D])
    output_flat = pl.reshape(output, [rows, D])
    rotation_flat = pl.reshape(rotation, [D, ROT_TILE])
    workers = pl.max(1, pl.min(pl.min(D // ROT_TILE, ROT_WORKER_TILE), pl.system.available_cluster_count()))
    for worker in pl.spmd(workers, name_hint="engram_unrotate_cube"):
        for block in pl.range(worker, D // ROT_TILE, workers):
            column = block * ROT_TILE
            weight = pl.load(rotation_flat, [column, 0], [ROT_TILE, ROT_TILE], target_memory=pl.MemorySpace.Mat)
            weight_t = pl.tile.transpose_view(weight)
            for row in pl.pipeline(0, rows, ROT_M_TILE, stage=2):
                valid_rows = pl.min(ROT_M_TILE, rows - row)
                value = pl.load(source_flat, [row, column], [ROT_M_TILE, ROT_TILE], valid_shape=[valid_rows, ROT_TILE])
                result = pl.tile.matmul(value, weight_t)
                result = pl.set_validshape(result, valid_rows, ROT_TILE)
                pl.store(result, [row, column], output_flat)
    return output


@pl.jit.inline
def engram_gate(
    kv: pl.Tensor,
    weight: pl.Tensor[[HC_MULT, D], pl.FP32],
    original_query: pl.Tensor,
    x: pl.Tensor,
    out: pl.Tensor,
    output_fp32: pl.constexpr,
    kv_bf16: pl.constexpr,
):
    """Gate projected keys and values into each residual stream copy."""
    t_dim = pl.tensor.dim(kv, 0)
    x_flat = pl.reshape(x, [t_dim, HC_MULT * D])
    query_flat = pl.reshape(original_query, [t_dim, HC_MULT * D])
    out_flat = pl.reshape(out, [t_dim, HC_MULT * D])
    t_blocks = (t_dim + T_TILE - 1) // T_TILE
    gate_tiles = t_blocks * HC_MULT
    gate_workers = pl.max(1, pl.min(gate_tiles, pl.system.available_aiv_count()))
    for gate_worker in pl.spmd(gate_workers, name_hint="engram_gate"):
        for block in pl.range(gate_worker, gate_tiles, gate_workers):
            t0 = (block // HC_MULT) * T_TILE
            c = block % HC_MULT
            valid_rows = pl.min(T_TILE, t_dim - t0)
            reduction_tmp = pl.create_tile([T_TILE, D_TILE], dtype=pl.FP32)
            # Reduction accumulators have 32-byte aligned rows.
            sq_x = pl.tile.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            sq_k = pl.tile.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            dot = pl.tile.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            for d0 in pl.range(0, D, D_TILE):
                x_raw = pl.load(query_flat, [t0, c * D + d0], [T_TILE, D_TILE], valid_shape=[valid_rows, D_TILE])
                key_raw = pl.load(kv, [t0, c * D + d0], [T_TILE, D_TILE], valid_shape=[valid_rows, D_TILE])
                if kv_bf16:
                    key_raw = pl.cast(key_raw, pl.FP32)
                key_d = pl.set_validshape(pl.fillpad(key_raw, pad_value=pl.PadValue.zero), T_TILE, D_TILE)
                x_d = pl.set_validshape(pl.fillpad(x_raw, pad_value=pl.PadValue.zero), T_TILE, D_TILE)
                sq_x = pl.add(sq_x, pl.reshape(pl.row_sum(pl.mul(x_d, x_d), reduction_tmp), [1, T_TILE]))
                sq_k = pl.add(sq_k, pl.reshape(pl.row_sum(pl.mul(key_d, key_d), reduction_tmp), [1, T_TILE]))
                w_d = pl.load(weight, [c, d0], [1, D_TILE])
                xw_d = pl.col_expand_mul(x_d, w_d)
                dot = pl.add(dot, pl.reshape(pl.row_sum(pl.mul(xw_d, key_d), reduction_tmp), [1, T_TILE]))
            query_variance = pl.add(pl.mul(sq_x, D_INV), NORM_EPS)
            key_variance = pl.add(pl.mul(sq_k, D_INV), NORM_EPS)
            rsqrt_tmp = pl.create_tile([1, T_TILE], dtype=pl.FP32)
            inv_x = pl.tile.rsqrt(query_variance, rsqrt_tmp, high_precision=True)
            inv_k = pl.tile.rsqrt(key_variance, rsqrt_tmp, high_precision=True)
            dot = pl.mul(pl.mul(pl.mul(dot, inv_x), inv_k), DOT_SCALE)
            root = pl.sqrt(pl.maximum(pl.abs(dot), CLAMP))
            nonnegative = pl.cmps(dot, 0.0, cmp_type=5)
            signed = pl.tile.select(nonnegative, root, pl.neg(root))
            gate = pl.reshape(pl.recip(pl.add(pl.exp(pl.neg(signed)), 1.0)), [T_TILE, 1])
            for d0 in pl.range(0, D, D_TILE):
                output_source = pl.load(x_flat, [t0, c * D + d0], [T_TILE, D_TILE], valid_shape=[valid_rows, D_TILE])
                output_value = pl.load(kv, [t0, HC_MULT * D + d0], [T_TILE, D_TILE], valid_shape=[valid_rows, D_TILE])
                if kv_bf16:
                    output_value = pl.cast(output_value, pl.FP32)
                gated = pl.row_expand_mul(output_value, gate)
                y = pl.add(output_source, gated)
                y_bf16 = pl.cast(y, target_type=pl.BF16, mode="rint")
                if output_fp32:
                    y_rounded = pl.cast(y_bf16, pl.FP32)
                    pl.store(pl.set_validshape(y_rounded, valid_rows, D_TILE), [t0, c * D + d0], out_flat)
                else:
                    pl.store(pl.set_validshape(y_bf16, valid_rows, D_TILE), [t0, c * D + d0], out_flat)
    return out


@pl.jit.inline(auto_scope=False)
def engram_project_nz(
    lookup: pl.Tensor,
    weight: pl.Tensor[[ENGRAM_K, KV_OUT], pl.BF16, pl.NZ],
    output: pl.Tensor,
    tokens: pl.Scalar[pl.INT32],
    row_tile: pl.constexpr,
    n_tile: pl.constexpr,
    k_tile: pl.constexpr,
    columns_first: pl.constexpr,
    k_subtile: pl.constexpr,
):
    """Project token tiles from a resident BF16 NZ matrix."""
    row_blocks = (tokens + row_tile - 1) // row_tile
    projection_tasks = KV_OUT // n_tile
    if columns_first:
        projection_tasks = projection_tasks * row_blocks
    workers = pl.max(1, pl.min(projection_tasks, pl.system.available_cluster_count()))
    for worker in pl.spmd(workers, name_hint="engram_projection"):
        if columns_first:
            panel_start = 0
            panel_step = 1
        else:
            panel_start = worker
            panel_step = workers
        for panel in pl.range(panel_start, KV_OUT // n_tile, panel_step):
            col = pl.max(0, panel) * n_tile
            if columns_first:
                # Distribute consecutive row tiles of the same weight panel across workers.
                first_row_tile = (worker + workers - (panel * row_blocks) % workers) % workers
                row_step = workers
            else:
                first_row_tile = 0
                row_step = 1
            for row_block in pl.range(first_row_tile, row_blocks, row_step):
                row = row_block * row_tile
                rows = pl.min(row_tile, tokens - row)
                seed = pl.tile.create([row_tile, n_tile], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc, compact=True)
                acc = pl.set_validshape(seed, rows, n_tile)
                for k in pl.pipeline(0, ENGRAM_K, k_tile, stage=2):
                    if k_subtile > 0:
                        a_panel = pl.load(lookup, [row, k], [row_tile, k_tile], valid_shape=[rows, k_tile],
                                          target_memory=pl.MemorySpace.Mat)
                        b_panel = pl.load(weight, [k, col], [k_tile, n_tile], target_memory=pl.MemorySpace.Mat)
                        for inner_k in pl.pipeline(0, k_tile, k_subtile, stage=2):
                            a = pl.tile.slice(a_panel, [row_tile, k_subtile], [0, inner_k])
                            b = pl.tile.slice(b_panel, [k_subtile, n_tile], [inner_k, 0])
                            acc = pl.matmul_acc(acc, a, b, init_cond=(k + inner_k == 0))
                    else:
                        source = pl.load(lookup, [row, k], [row_tile, k_tile], valid_shape=[rows, k_tile])
                        weights = pl.load(weight, [k, col], [k_tile, n_tile], target_memory=pl.MemorySpace.Mat)
                        acc = pl.matmul_acc(acc, source, weights, init_cond=(k == 0))
                pl.store(pl.set_validshape(acc, rows, n_tile), [row, col], output)
    return output


def make_engram_from_rows(*, prefill):
    @pl.jit.inline(auto_scope=False)
    def engram_from_rows(
        lookup: pl.Tensor,
        wkv_weight: pl.Tensor[[ENGRAM_K, KV_OUT], pl.BF16, pl.NZ],
        weight: pl.Tensor[[HC_MULT, D], pl.FP32],
        rotation: pl.Tensor[[D // 32, 32, 32], pl.FP32],
        x: pl.Tensor,
        out: pl.Tensor,
    ):
        """Project host-gathered BF16 rows and gate the FP32 residual stream."""
        tokens = pl.tensor.dim(x, 0)
        kv = pl.create_tensor([tokens, KV_OUT], dtype=pl.BF16)
        if prefill and tokens >= PREFILL_MIN_ROWS:
            engram_project_nz(lookup, wkv_weight, kv, tokens, PREFILL_M_TILE, N_TILE, K_TILE, True, PREFILL_BF16_K_SUBTILE)
        elif not prefill and tokens == 1:
            engram_project_nz(lookup, wkv_weight, kv, tokens, T_TILE, DECODE_ENGRAM_N_TILE, DECODE_ENGRAM_K_TILE, False, 0)
        else:
            engram_project_nz(lookup, wkv_weight, kv, tokens, 64, N_TILE, K_TILE, False, 0)
        pl.dump_tag(kv)
        if prefill:
            original_query = pl.create_tensor([tokens, HC_MULT, D], dtype=pl.FP32)
            if tokens >= PREFILL_MIN_ROWS:
                prefill_engram_unrotate(x, rotation, original_query)
            else:
                engram_unrotate(x, rotation, original_query)
            pl.dump_tag(original_query)
            engram_gate(kv, weight, original_query, x, out, output_fp32=True, kv_bf16=True)
        else:
            original_query = pl.create_tensor([tokens, HC_MULT, D], dtype=pl.FP32)
            if tokens == 1:
                engram_unrotate(x, rotation, original_query)
                pl.dump_tag(original_query)
                engram_gate(kv, weight, original_query, x, out, output_fp32=True, kv_bf16=True)
            else:
                kv_fp32 = pl.create_tensor([tokens, KV_OUT], dtype=pl.FP32)
                engram_cast_to_fp32(kv, kv_fp32)
                engram_unrotate(x, rotation, original_query)
                pl.dump_tag(original_query)
                engram_gate(kv_fp32, weight, original_query, x, out, output_fp32=True, kv_bf16=False)
        return out

    return engram_from_rows


engram_from_rows = make_engram_from_rows(prefill=False)
prefill_engram_from_rows = make_engram_from_rows(prefill=True)


@pl.jit(auto_scope=False)
def l2_engram(
    lookup: pl.Tensor[[T_DYN, ENGRAM_K], pl.BF16],
    wkv_weight: pl.Tensor[[ENGRAM_K, KV_OUT], pl.BF16, pl.NZ],
    weight: pl.Tensor[[HC_MULT, D], pl.FP32],
    rotation: pl.Tensor[[D // 32, 32, 32], pl.FP32],
    source: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    lookup.bind_dynamic(0, T_DYN)
    source.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    engram_from_rows(lookup, wkv_weight, weight, rotation, source, output)
    return output


@pl.jit(auto_scope=False)
def l2_prefill_engram(
    lookup: pl.Tensor[[T_DYN, ENGRAM_K], pl.BF16],
    wkv_weight: pl.Tensor[[ENGRAM_K, KV_OUT], pl.BF16, pl.NZ],
    weight: pl.Tensor[[HC_MULT, D], pl.FP32],
    rotation: pl.Tensor[[D // 32, 32, 32], pl.FP32],
    source: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    lookup.bind_dynamic(0, T_DYN)
    source.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    prefill_engram_from_rows(lookup, wkv_weight, weight, rotation, source, output)
    return output


def golden_engram(tensors, *, packed_weight=False):
    """Project real host rows, unrotate queries and gate the rotated value residual."""
    source = tensors["source"].float()
    tokens = source.shape[0]
    weight = tensors["wkv_weight"]
    if packed_weight:
        weight = unpack_bf16_nz(weight)
    projection = (tensors["lookup"].float() @ weight.float()).bfloat16().float()
    key = projection[:, :HC_MULT * D].reshape(tokens, HC_MULT, D)
    value = projection[:, HC_MULT * D:]
    original = torch.einsum("thbi,boi->thbo", source.reshape(tokens, HC_MULT, D // 32, 32),
                            tensors["rotation"]).reshape_as(source)
    inv_query = torch.rsqrt(original.square().mean(-1) + NORM_EPS)
    inv_key = torch.rsqrt(key.square().mean(-1) + NORM_EPS)
    dot = (original * tensors["weight"] * key).sum(-1) * inv_query * inv_key * DOT_SCALE
    signed = torch.copysign(dot.abs().clamp_min(CLAMP).sqrt(), dot)
    gate = signed.sigmoid().unsqueeze(-1)
    tensors["output"].copy_((source + gate * value.unsqueeze(1)).bfloat16().float())


class HostEngram:
    """Hash text history and read selected INT8 checkpoint table rows on the CPU.

    Reuse one instance across prefill chunks and decode calls. Each request_id
    owns its compressed-token history; start_pos=0 resets that request, and
    continuation requires the preceding history. Image-token masking is unsupported.
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

    def hash_ids(self, token_ids, start_pos=0, *, request_id=0):
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
        compressed = torch.cat((history[:start_pos], self.token_map[ids]))
        positions = torch.arange(start_pos, compressed.numel())
        lookback = []
        for shift in range(M.engram_max_ngram_size):
            source = (positions - shift).clamp_min(0)
            values = compressed[source]
            lookback.append(torch.where(positions >= shift, values, self.pad_id))
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

    def lookup(self, token_ids, capacity, start_pos=0, *, request_id=0):
        """Read current rows and group-32 FP32 scales into [2, capacity, 6144] BF16.

        Tables stay in host checkpoint files; only selected rows are read.
        Unused token rows are zero-padded.
        """

        if len(token_ids) > capacity or capacity <= 0:
            raise ValueError("Engram rows must fit the current-token capacity")
        hashes = self.hash_ids(token_ids, start_pos, request_id=request_id)
        layers = len(M.engram_layer_ids)
        width = self.primes.shape[1] * M.engram_head_dim
        result = torch.zeros((layers, capacity, width), dtype=torch.bfloat16)
        for slot, layer in enumerate(M.engram_layer_ids):
            prefix = f"layers.{layer}.engram.embed."
            row_ids = hashes[slot].flatten()
            repeated = len(token_ids) > 1 and hashes[slot, :, 0].unique().numel() * 2 <= len(token_ids)
            if repeated:
                row_ids, restore = torch.unique(row_ids, return_inverse=True)
            rows = self.checkpoint.tensor_rows(prefix + "weight", row_ids)
            scales = self.checkpoint.tensor_rows(prefix + "scale", row_ids)
            if rows.dtype != torch.int8 or rows.shape[-1] != M.engram_head_dim:
                raise ValueError(f"unsupported native Engram table layout: {prefix}weight")
            if scales.dtype != torch.float32 or scales.shape[-1] != M.engram_head_dim // 32:
                raise ValueError(f"unsupported native Engram table scales: {prefix}scale")
            groups = rows.unflatten(-1, (-1, 32))
            decoded = (groups * scales.unsqueeze(-1)).bfloat16().flatten(1)
            dest = result[slot, :len(token_ids)].view(-1, M.engram_head_dim)
            if repeated:
                torch.index_select(decoded, 0, restore, out=dest)
            else:
                dest.copy_(decoded)
        return result


@pl.jit
def validate_rotation(
    source: pl.Tensor[[16, HC_MULT, D], pl.FP32],
    rotation: pl.Tensor[[D // 32, 32, 32], pl.FP32],
    output: pl.Out[pl.Tensor[[16, HC_MULT, D], pl.FP32]],
):
    engram_unrotate(source, rotation, output)
    return output


if __name__ == "__main__":
    import argparse
    import json
    from functools import partial

    from golden import TensorSpec, ratio_allclose, run

    from .weights import Checkpoint, pack_bf16_nz

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("-p", "--platform", choices=["a2a3"], default="a2a3")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--mode", choices=["rotation", "full", "all"], default="all")
    parser.add_argument("--tokens", type=int, choices=[1, 2, 33, 8192], default=33)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(4141)
    checkpoint = Checkpoint(args.checkpoint) if args.checkpoint else None
    rotation_shape = [D // 32, 32, 32]
    if checkpoint is None:
        blocks = torch.randn(*rotation_shape, generator=generator) / math.sqrt(32)
    else:
        blocks = checkpoint.rotation_blocks()
    rotation_spec = TensorSpec("rotation", rotation_shape, torch.float32, init_value=blocks)
    config = {"backend_type": args.platform, "execution_mode": args.sim, "device_id": args.device}

    def golden_rotation(tensors):
        value = tensors["source"].double().reshape(16, HC_MULT, D // 32, 32)
        reference = torch.einsum("thbi,boi->thbo", value, tensors["rotation"].double())
        tensors["output"].copy_(reference.reshape(16, HC_MULT, D))

    records = []
    if args.mode in ("rotation", "all"):
        source = torch.randn(16, HC_MULT, D, generator=generator)
        specs = [
            TensorSpec("source", [16, HC_MULT, D], torch.float32, init_value=source),
            rotation_spec,
            TensorSpec("output", [16, HC_MULT, D], torch.float32),
        ]
        result = run(
            fn=validate_rotation,
            specs=specs,
            golden_fn=golden_rotation,
            compare_fn={"output": ratio_allclose(atol=2e-6, rtol=2e-6, max_error_ratio=0.0)},
            config=config,
            compile_only=args.compile_only,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
        records.append(dict(case="rotation", passed=result.passed, work_dir=str(result.work_dir)))
    if args.mode in ("full", "all"):
        tokens = args.tokens
        source = torch.randn(tokens, HC_MULT, D, generator=generator)
        if checkpoint is None:
            lookups = (torch.randn(tokens, ENGRAM_K, dtype=torch.bfloat16, generator=generator),)
            layers = (None,)
        else:
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(checkpoint.root / "tokenizer.json"))
            host = HostEngram(checkpoint, tokenizer)
            block = tokenizer.encode("A mathematical proof explains each step. ", add_special_tokens=False).ids
            token_ids = (block * ((tokens + len(block) - 1) // len(block)))[:tokens]
            lookups = host.lookup(token_ids, tokens)
            layers = M.engram_layer_ids
            config = dict(config, ring_heap=(2 * 1024**3, 1024**3, 1024**3, 1024**3))

        def synthetic_projection():
            projection = torch.empty(ENGRAM_K, KV_OUT, dtype=torch.bfloat16)
            projection.uniform_(-0.02, 0.02, generator=generator)
            return pack_bf16_nz(projection)

        for slot, layer in enumerate(layers):
            if layer is None:
                weight = torch.rand(HC_MULT, D, generator=generator) + 0.5
                projection_spec = TensorSpec("wkv_weight", [ENGRAM_K, KV_OUT], torch.bfloat16, synthetic_projection)
            else:
                prefix = f"layers.{layer}.engram."
                q_weight = checkpoint.tensor(prefix + "q_weight").float()
                weight = q_weight * checkpoint.tensor(prefix + "k_weight").float()
                packed_nz = pack_bf16_nz(checkpoint.tensor(prefix + "wkv.weight").t())
                projection_spec = TensorSpec("wkv_weight", list(packed_nz.shape), torch.bfloat16, packed_nz)
            specs = [
                TensorSpec("lookup", [tokens, ENGRAM_K], torch.bfloat16, init_value=lookups[slot]),
                projection_spec,
                TensorSpec("weight", [HC_MULT, D], torch.float32, init_value=weight),
                rotation_spec,
                TensorSpec("source", [tokens, HC_MULT, D], torch.float32, init_value=source),
                TensorSpec("output", [tokens, HC_MULT, D], torch.float32),
            ]
            for phase, entry in (("decode", l2_engram), ("prefill", l2_prefill_engram)):
                result = run(
                    fn=entry,
                    specs=specs,
                    golden_fn=partial(golden_engram, packed_weight=True),
                    compare_fn={"output": ratio_allclose(atol=0.003, rtol=0.03, max_error_ratio=0.005)},
                    config=config,
                    compile_only=args.compile_only,
                    save_data=checkpoint is not None,
                )
                if not result.passed:
                    raise RuntimeError(result.error)
                records.append(dict(phase=phase, layer=layer, tokens=tokens, passed=result.passed, work_dir=str(result.work_dir)))
    if args.report:
        identity = checkpoint.identity if checkpoint is not None else "synthetic"
        args.report.write_text(json.dumps(dict(checkpoint=identity, cases=records), indent=2) + "\n")
