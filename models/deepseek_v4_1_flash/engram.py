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

import pypto.language as pl
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
# Validation-sized table; the released layer-1 table (384006168 rows) does not fit
# in host memory for a standalone case, so this exercises the same kernel against a
# much smaller row count. The kernel only reads gathered rows, so any size works.
NUM_EMBEDDINGS = 1 << 20  # 1048576 rows (~512 MiB as BF16)
CLAMP = 1e-6
NORM_EPS = M.rms_norm_eps
D_INV = 1.0 / D
DOT_SCALE = D**-0.5

T_TILE = 16
K_TILE = 128
N_TILE = 256
D_TILE = 512


@pl.jit.inline
def engram_cast_to_fp32(source: pl.Tensor, output: pl.Tensor):
    """Widen a BF16 matrix into FP32 tiles."""
    rows = pl.tensor.dim(source, 0)
    width = pl.tensor.dim(source, 1)
    for block in pl.spmd(((rows + T_TILE - 1) // T_TILE) * (width // N_TILE), name_hint="engram_widen"):
        t0 = block // (width // N_TILE) * T_TILE
        n0 = block % (width // N_TILE) * N_TILE
        valid_rows = pl.min(T_TILE, rows - t0)
        value = pl.slice(source, [T_TILE, N_TILE], [t0, n0], valid_shape=[valid_rows, N_TILE])
        widened = pl.cast(value, pl.FP32)
        output[t0:t0 + T_TILE, n0:n0 + N_TILE] = pl.set_validshape(widened, valid_rows, N_TILE)
    return output


@pl.jit.inline
def engram_gate(
    kv: pl.Tensor,
    weight: pl.Tensor[[HC_MULT, D], pl.FP32],
    x: pl.Tensor,
    out: pl.Tensor,
    mask_source: pl.Tensor,
    output_fp32: pl.constexpr,
    mask_zero_rows: pl.constexpr,
):
    """Gate projected keys and values into each residual stream copy."""
    t_dim = pl.tensor.dim(kv, 0)
    x_flat = pl.reshape(x, [t_dim, HC_MULT * D])
    out_flat = pl.reshape(out, [t_dim, HC_MULT * D])
    t_blocks = (t_dim + T_TILE - 1) // T_TILE
    for block in pl.spmd(t_blocks * HC_MULT, name_hint="engram_gate"):
        t0 = (block // HC_MULT) * T_TILE
        c = block % HC_MULT
        valid_rows = pl.min(T_TILE, t_dim - t0)
        reduction_tmp = pl.create_tile([T_TILE, D_TILE], dtype=pl.FP32)
        # Reduction accumulators have 32-byte aligned rows.
        sq_x = pl.tile.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        sq_k = pl.tile.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        dot = pl.tile.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for d0 in pl.range(0, D, D_TILE):
            x_raw = pl.load(x_flat, [t0, c * D + d0], [T_TILE, D_TILE], valid_shape=[valid_rows, D_TILE])
            key_raw = pl.load(kv, [t0, c * D + d0], [T_TILE, D_TILE], valid_shape=[valid_rows, D_TILE])
            key_d = pl.set_validshape(pl.fillpad(key_raw, pad_value=pl.PadValue.zero), T_TILE, D_TILE)
            x_d = pl.set_validshape(pl.fillpad(x_raw, pad_value=pl.PadValue.zero), T_TILE, D_TILE)
            sq_x = pl.add(sq_x, pl.reshape(pl.row_sum(pl.mul(x_d, x_d), reduction_tmp), [1, T_TILE]))
            sq_k = pl.add(sq_k, pl.reshape(pl.row_sum(pl.mul(key_d, key_d), reduction_tmp), [1, T_TILE]))
            w_d = pl.load(weight, [c, d0], [1, D_TILE])
            xw_d = pl.col_expand_mul(x_d, w_d)
            dot = pl.add(dot, pl.reshape(pl.row_sum(pl.mul(xw_d, key_d), reduction_tmp), [1, T_TILE]))
        inv_x = pl.rsqrt(pl.add(pl.mul(sq_x, D_INV), NORM_EPS))
        inv_k = pl.rsqrt(pl.add(pl.mul(sq_k, D_INV), NORM_EPS))
        dot = pl.mul(pl.mul(pl.mul(dot, inv_x), inv_k), DOT_SCALE)
        root = pl.sqrt(pl.maximum(pl.abs(dot), CLAMP))
        nonnegative = pl.cmps(dot, 0.0, cmp_type=5)
        signed = pl.tile.select(nonnegative, root, pl.neg(root))
        gate = pl.reshape(pl.recip(pl.add(pl.exp(pl.neg(signed)), 1.0)), [T_TILE, 1])
        lookup_sum = pl.tile.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        if mask_zero_rows:
            for k0 in pl.range(0, ENGRAM_K, K_TILE):
                lookup_raw = pl.load(mask_source, [t0, k0], [T_TILE, K_TILE], valid_shape=[valid_rows, K_TILE])
                lookup_fp32 = pl.cast(lookup_raw, pl.FP32)
                lookup_fp32 = pl.set_validshape(pl.fillpad(lookup_fp32, pad_value=pl.PadValue.zero), T_TILE, K_TILE)
                lookup_abs = pl.abs(lookup_fp32)
                lookup_part = pl.reshape(pl.row_sum(lookup_abs, reduction_tmp), [1, T_TILE])
                lookup_sum = pl.add(lookup_sum, lookup_part)
        for d0 in pl.range(0, D, D_TILE):
            output_source = pl.load(x_flat, [t0, c * D + d0], [T_TILE, D_TILE], valid_shape=[valid_rows, D_TILE])
            output_value = pl.load(kv, [t0, HC_MULT * D + d0], [T_TILE, D_TILE], valid_shape=[valid_rows, D_TILE])
            gated = pl.row_expand_mul(output_value, gate)
            y = pl.add(output_source, gated)
            y_bf16 = pl.cast(y, target_type=pl.BF16, mode="rint")
            if output_fp32:
                y_rounded = pl.cast(y_bf16, pl.FP32)
                if mask_zero_rows:
                    lookup_column = pl.reshape(lookup_sum, [T_TILE, 1])
                    mask_ones = pl.tile.full([T_TILE, D_TILE], dtype=pl.FP32, value=1.0)
                    lookup_expanded = pl.row_expand_mul(mask_ones, lookup_column)
                    lookup_valid = pl.set_validshape(lookup_expanded, valid_rows, D_TILE)
                    lookup_active = pl.cmps(lookup_valid, 0.0, cmp_type=4)
                    y_rounded = pl.tile.select(lookup_active, y_rounded, output_source)
                pl.store(pl.set_validshape(y_rounded, valid_rows, D_TILE), [t0, c * D + d0], out_flat)
            else:
                pl.store(pl.set_validshape(y_bf16, valid_rows, D_TILE), [t0, c * D + d0], out_flat)
    return out


@pl.jit.inline(auto_scope=False)
def engram_from_rows(
    lookup: pl.Tensor,
    wkv_weight: pl.Tensor[[ENGRAM_K, KV_OUT], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[ENGRAM_K // 32, KV_OUT], pl.FP8E8M0],
    weight: pl.Tensor[[HC_MULT, D], pl.FP32],
    x: pl.Tensor,
    out: pl.Tensor,
    mask_zero_rows: pl.constexpr,
):
    """Project host BF16 rows; all-zero lookup rows preserve the FP32 residual exactly."""
    tokens = pl.tensor.dim(x, 0)
    kv = pl.create_tensor([tokens, KV_OUT], dtype=pl.BF16)
    gemm_bf16_via_mxfp8_to_bf16_staged(
        lookup, wkv_weight, wkv_scale, kv, tokens,
        ENGRAM_K, KV_OUT, "engram_projection", "engram_quant", N_TILE, 256, 1, 2147483647,
    )
    kv_fp32 = pl.create_tensor([tokens, KV_OUT], dtype=pl.FP32)
    engram_cast_to_fp32(kv, kv_fp32)
    engram_gate(kv_fp32, weight, x, out, lookup, output_fp32=True, mask_zero_rows=mask_zero_rows)
    return out


@pl.jit
def l2_engram_masked_gate(
    kv: pl.Tensor[[T_DYN, KV_OUT], pl.FP32],
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
    engram_gate(kv, weight, x, out, lookup, output_fp32=True, mask_zero_rows=True)
    return out


@pl.jit.inline
def engram(
    hash_ids: pl.Tensor[[T_DYN, N_HASH_COLS], pl.INT32],
    engram_table: pl.Tensor[[NUM_EMBEDDINGS, HEAD_DIM], pl.BF16],
    wkv_weight: pl.Tensor[[ENGRAM_K, KV_OUT], pl.BF16],
    weight: pl.Tensor[[HC_MULT, D], pl.FP32],
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    out: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
):
    t_dim = pl.tensor.dim(hash_ids, 0)
    t_blocks = (t_dim + T_TILE - 1) // T_TILE

    # GM intermediate: kv[t, n] = flatten(engram_table[hash_ids[t]]) @ wkv[:, n]
    kv = pl.create_tensor([t_dim, KV_OUT], dtype=pl.FP32)

    # ---- Stage 1: gather n-gram rows, project to key|value, write to GM
    n_blocks = KV_OUT // N_TILE
    for block in pl.spmd(t_blocks * n_blocks, name_hint="engram_matmul"):
        t0 = (block // n_blocks) * T_TILE
        n0 = (block % n_blocks) * N_TILE
        valid_rows = pl.min(T_TILE, t_dim - t0)

        # gather the 24 n-gram rows for this token tile into [T_TILE, 6144]
        lookup = pl.create_tensor([T_TILE, ENGRAM_K], dtype=pl.BF16)
        for c in pl.range(N_HASH_COLS):
            for tt in pl.range(T_TILE):
                if t0 + tt < t_dim:
                    row = pl.read(hash_ids, [t0 + tt, c])
                    row = pl.max(0, pl.min(NUM_EMBEDDINGS - 1, row))
                    lookup[tt : tt + 1, c * HEAD_DIM : (c + 1) * HEAD_DIM] = engram_table[
                        row : row + 1, 0:HEAD_DIM
                    ]
                else:
                    lookup[tt : tt + 1, c * HEAD_DIM : (c + 1) * HEAD_DIM] = pl.full(
                        [1, HEAD_DIM], dtype=pl.BF16, value=0.0
                    )

        # matmul: [T,6144] @ [6144,N_TILE], bf16 in, fp32 acc
        acc = pl.create_tensor([T_TILE, N_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(ENGRAM_K // K_TILE, stage=2):
            k0 = kb * K_TILE
            a_tile = pl.slice(lookup, [T_TILE, K_TILE], [0, k0])
            w_tile = pl.slice(wkv_weight, [K_TILE, N_TILE], [k0, n0])
            acc = pl.matmul_acc(acc, a_tile, w_tile, init_cond=(kb == 0))
        kv[t0 : t0 + T_TILE, n0 : n0 + N_TILE] = pl.set_validshape(acc, valid_rows, N_TILE)

    # ---- Stage 2: per (token block, hc copy) gate and residual add
    x_flat = pl.reshape(x, [t_dim, HC_MULT * D])
    x_fp32 = pl.create_tensor([t_dim, HC_MULT * D], dtype=pl.FP32)
    engram_cast_to_fp32(x_flat, x_fp32)
    x_hc_fp32 = pl.reshape(x_fp32, [t_dim, HC_MULT, D])
    engram_gate(kv, weight, x_hc_fp32, out, kv, output_fp32=False, mask_zero_rows=False)
    return out


def golden_engram(
    hash_ids: torch.Tensor,
    engram_table: torch.Tensor,
    wkv_weight: torch.Tensor,
    weight: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Torch reference mirroring the released Engram.forward (world_size == 1)."""
    rows = engram_table[hash_ids.long()]  # [T, N_HASH_COLS, HEAD_DIM], bf16
    flattened = rows.float().flatten(1)  # [T, ENGRAM_K]
    kv = flattened @ wkv_weight.float()  # [T, KV_OUT]
    key, value = kv.split([HC_MULT * D, D], dim=-1)
    key = key.unflatten(-1, (HC_MULT, D))  # [T, HC_MULT, D]

    h = x.float()
    w = weight.float().unsqueeze(0)  # [1, HC_MULT, D]
    rstd = torch.rsqrt(h.square().mean(-1, keepdim=True) + M.rms_norm_eps) * torch.rsqrt(
        key.square().mean(-1, keepdim=True) + M.rms_norm_eps
    )
    dot = (h * w * key).sum(-1, keepdim=True) * rstd * (D**-0.5)  # [T, HC_MULT, 1]
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(CLAMP).sqrt(), dot))
    return (h + gate * value.unsqueeze(1)).to(torch.bfloat16)


@pl.jit
def l2_engram(
    hash_ids: pl.Tensor[[T_DYN, N_HASH_COLS], pl.INT32],
    engram_table: pl.Tensor[[NUM_EMBEDDINGS, HEAD_DIM], pl.BF16],
    wkv_weight: pl.Tensor[[ENGRAM_K, KV_OUT], pl.BF16],
    weight: pl.Tensor[[HC_MULT, D], pl.FP32],
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]],
):
    """Run the Engram block for standalone validation."""
    hash_ids.bind_dynamic(0, T_DYN)
    x.bind_dynamic(0, T_DYN)
    out.bind_dynamic(0, T_DYN)
    engram(hash_ids, engram_table, wkv_weight, weight, x, out)
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
    kv = torch.randn(tokens, KV_OUT, generator=generator)
    lookup[1::3] = 0
    kv[1::3] = 0
    hidden = torch.randn(tokens, HC_MULT, D, generator=generator)
    return [
        TensorSpec("kv", [tokens, KV_OUT], torch.float32, init_value=kv),
        TensorSpec("weight", [HC_MULT, D], torch.float32, init_value=1.0),
        TensorSpec("x", [tokens, HC_MULT, D], torch.float32, init_value=hidden),
        TensorSpec("lookup", [tokens, ENGRAM_K], torch.bfloat16, init_value=lookup),
        TensorSpec("out", [tokens, HC_MULT, D], torch.float32),
    ]


def golden_masked_gate(tensors):
    """Evaluate gated text rows and exact image-row residual passthrough."""
    hidden = tensors["x"]
    keys = tensors["kv"][:, :HC_MULT * D].reshape(-1, HC_MULT, D)
    value = tensors["kv"][:, HC_MULT * D:]
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
    return ratio_allclose(atol=1e-3, rtol=1e-2)(actual, expected, inputs=inputs, **kwargs)


def build_engram_tensor_specs(batch: int = 2, sequence: int = 4):
    """Build deterministic inputs and the output for Engram validation."""
    from golden import TensorSpec

    tokens = batch * sequence
    generator = torch.Generator().manual_seed(3)

    def init_hash_ids():
        return torch.randint(0, NUM_EMBEDDINGS, (tokens, N_HASH_COLS), generator=generator)

    def init_table():
        # small-magnitude rows keep the projection in a numerically tame range
        return (torch.randn(NUM_EMBEDDINGS, HEAD_DIM, generator=generator) * 0.02).to(
            torch.bfloat16
        )

    def init_wkv():
        return (torch.randn(ENGRAM_K, KV_OUT, generator=generator) / math.sqrt(ENGRAM_K)).to(
            torch.bfloat16
        )

    def init_weight():
        # released module keeps q_weight and k_weight separate; fold them
        # host-side so the kernel takes a single per-channel scale
        q = torch.rand(HC_MULT, D, generator=generator) + 0.5
        k = torch.rand(HC_MULT, D, generator=generator) + 0.5
        return q * k

    def init_x():
        return torch.randn(tokens, HC_MULT, D, generator=generator).to(torch.bfloat16)

    return [
        TensorSpec("hash_ids", [tokens, N_HASH_COLS], torch.int32, init_value=init_hash_ids),
        TensorSpec("engram_table", [NUM_EMBEDDINGS, HEAD_DIM], torch.bfloat16, init_value=init_table),
        TensorSpec("wkv_weight", [ENGRAM_K, KV_OUT], torch.bfloat16, init_value=init_wkv),
        TensorSpec("weight", [HC_MULT, D], torch.float32, init_value=init_weight),
        TensorSpec("x", [tokens, HC_MULT, D], torch.bfloat16, init_value=init_x),
        TensorSpec("out", [tokens, HC_MULT, D], torch.bfloat16),
    ]


def golden_engram_case(tensors):
    """Fill the expected Engram output."""
    tensors["out"][:] = golden_engram(
        tensors["hash_ids"],
        tensors["engram_table"],
        tensors["wkv_weight"],
        tensors["weight"],
        tensors["x"],
    )


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
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=4)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--masked-gate", action="store_true")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    if len(device_ids) != 1:
        raise ValueError(f"need exactly one device id, got {device_ids}")

    entry = l2_engram_masked_gate if args.masked_gate else l2_engram
    specs = (build_masked_gate_specs(args.batch * args.sequence) if args.masked_gate
             else build_engram_tensor_specs(args.batch, args.sequence))
    golden_fn = golden_masked_gate if args.masked_gate else golden_engram_case
    comparator = compare_masked_gate if args.masked_gate else ratio_allclose(atol=1e-3, rtol=1e-2)
    result = run(
        fn=entry,
        specs=specs,
        golden_fn=golden_fn,
        config={
            "backend_type": args.backend_type, "execution_mode": args.execution_mode,
            "device_id": device_ids[0],
            "enable_chip_swimlane": args.enable_chip_swimlane,
        },
        rtol=1e-3,
        atol=1e-3,
        compare_fn={"out": comparator},
        compile_only=args.compile_only,
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
