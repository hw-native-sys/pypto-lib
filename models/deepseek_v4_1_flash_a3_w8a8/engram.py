# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Engram conditional memory: host-offloaded INT8 group-32 rows, BF16 projection, residual gating.

The 384M-row tables stay on the host (A3 ``cpu_offload``).  The host gathers the
24 n-gram rows per token and dequantizes int8_sym group-32 rows to BF16; the
device projects the 6144-wide lookup with ``wkv`` into four key copies and one
value, computes the RMS-normalized gated dot product, and adds the gated value
to every hyper-connection stream.
"""

from __future__ import annotations

from pathlib import Path

import pypto.language as pl
import torch

from compressor import bf16_project
from config import ENGRAM_GROUP_SIZE, FLASH as M, T_DYN


# Dynamic shape variable.

# model config
D = M.hidden_size
HC_MULT = M.hc_mult
N_HASH_COLS = (M.engram_max_ngram_size - 1) * M.engram_n_heads
HEAD_DIM = M.engram_head_dim
ENGRAM_K = N_HASH_COLS * HEAD_DIM
KV_OUT = (HC_MULT + 1) * D
NORM_EPS = M.rms_norm_eps
D_INV = 1.0 / D
DOT_SCALE = D**-0.5
CLAMP = 1e-6

# tiling
T_TILE = 8
D_TILE = 512


@pl.jit.inline(auto_scope=False)
def engram_gate(
    kv: pl.Tensor,
    weight: pl.Tensor,
    rotation: pl.Tensor,
    x: pl.Tensor,
    output: pl.Tensor,
):
    """Restore original-basis hidden for gating and update the rotated FP32 residual."""
    t_dim = pl.tensor.dim(kv, 0)
    x_flat = pl.reshape(x, [t_dim, HC_MULT * D])
    rotation_rows = t_dim * HC_MULT * D // 32
    x_blocks = pl.reshape(x, [rotation_rows, 32])
    restored = pl.create_tensor([rotation_rows, 32], dtype=pl.FP32)
    for block in pl.spmd((rotation_rows + T_TILE - 1) // T_TILE, name_hint="engram_restore_basis"):
        begin = block * T_TILE
        active = pl.min(T_TILE, rotation_rows - begin)
        hidden_block = pl.tile.load(x_blocks, [begin, 0], [T_TILE, 32], valid_shape=[active, 32])
        hidden_block = pl.set_validshape(pl.fillpad(hidden_block, pad_value=pl.PadValue.zero), T_TILE, 32)
        original_block = pl.tile.full([T_TILE, 32], dtype=pl.FP32, value=0.0)
        for column in pl.range(32):
            rotation_row = pl.tile.load(rotation, [column, 0], [1, 32])
            products = pl.col_expand_mul(hidden_block, rotation_row)
            reduce_tmp = pl.tile.create([T_TILE, 32], dtype=pl.FP32)
            reduced = pl.reshape(pl.row_sum(products, tmp_tile=reduce_tmp), [T_TILE, 1])
            for lane in pl.unroll(T_TILE):
                pl.tile.write(original_block, [lane, column], pl.tile.read(reduced, [lane, 0]))
        pl.tile.store(pl.set_validshape(original_block, active, 32), [begin, 0], restored)
    restored_flat = pl.reshape(restored, [t_dim, HC_MULT * D])
    out_flat = pl.reshape(output, [t_dim, HC_MULT * D])
    for block in pl.spmd(((t_dim + T_TILE - 1) // T_TILE) * HC_MULT, name_hint="engram_gate"):
        t0 = (block // HC_MULT) * T_TILE
        copy = block % HC_MULT
        rows = pl.min(T_TILE, t_dim - t0)
        sq_x = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        sq_k = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        dot = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for d0 in pl.range(0, D, D_TILE):
            x_raw = pl.slice(restored_flat, [T_TILE, D_TILE], [t0, copy * D + d0], valid_shape=[rows, D_TILE])
            key_raw = pl.slice(kv, [T_TILE, D_TILE], [t0, copy * D + d0], valid_shape=[rows, D_TILE])
            x_tile = pl.set_validshape(pl.fillpad(x_raw, pad_value=pl.PadValue.zero), T_TILE, D_TILE)
            key_tile = pl.set_validshape(pl.fillpad(key_raw, pad_value=pl.PadValue.zero), T_TILE, D_TILE)
            key_tile = pl.cast(key_tile, target_type=pl.FP32)
            sq_x = pl.add(sq_x, pl.reshape(pl.row_sum(pl.mul(x_tile, x_tile)), [1, T_TILE]))
            sq_k = pl.add(sq_k, pl.reshape(pl.row_sum(pl.mul(key_tile, key_tile)), [1, T_TILE]))
            column_weight = pl.slice(weight, [1, D_TILE], [copy, d0])
            weighted = pl.col_expand_mul(x_tile, column_weight)
            dot = pl.add(dot, pl.reshape(pl.row_sum(pl.mul(weighted, key_tile)), [1, T_TILE]))
        inv_x = pl.rsqrt(pl.add(pl.mul(sq_x, D_INV), NORM_EPS), high_precision=True)
        inv_k = pl.rsqrt(pl.add(pl.mul(sq_k, D_INV), NORM_EPS), high_precision=True)
        dot = pl.mul(pl.mul(pl.mul(dot, inv_x), inv_k), DOT_SCALE)
        magnitude = pl.maximum(dot, pl.neg(dot))
        zero = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        nonnegative = pl.cmp(dot, zero, cmp_type=5)
        sign = pl.sub(pl.mul(nonnegative, 2.0), 1.0)
        root = pl.sqrt(pl.maximum(magnitude, CLAMP))
        gate = pl.reshape(pl.recip(pl.add(pl.exp(pl.neg(pl.mul(root, sign))), 1.0)), [T_TILE, 1])
        for d0 in pl.range(0, D, D_TILE):
            source = pl.slice(x_flat, [T_TILE, D_TILE], [t0, copy * D + d0], valid_shape=[rows, D_TILE])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), T_TILE, D_TILE)
            value = pl.slice(kv, [T_TILE, D_TILE], [t0, HC_MULT * D + d0], valid_shape=[rows, D_TILE])
            value = pl.set_validshape(pl.fillpad(value, pad_value=pl.PadValue.zero), T_TILE, D_TILE)
            value = pl.cast(value, target_type=pl.FP32)
            gated = pl.row_expand_mul(value, gate)
            mixed = pl.add(source, gated)
            out_flat[t0 : t0 + T_TILE, copy * D + d0 : copy * D + d0 + D_TILE] = pl.set_validshape(mixed, rows, D_TILE)
    return output


@pl.jit.inline(auto_scope=False)
def engram_from_rows(
    lookup: pl.Tensor,
    wkv_weight: pl.Tensor,
    weight: pl.Tensor,
    rotation: pl.Tensor,
    x: pl.Tensor,
    output: pl.Tensor,
):
    """Project host-gathered BF16 rows and gate them into the FP32 residual streams."""
    t_dim = pl.tensor.dim(lookup, 0)
    kv = pl.create_tensor([t_dim, KV_OUT], dtype=pl.BF16)
    kv = bf16_project(lookup, wkv_weight, kv, width=ENGRAM_K, output_width=KV_OUT, output_fp32=False)
    output = engram_gate(kv, weight, rotation, x, output)
    return output


@pl.jit
def engram_test(
    lookup: pl.Tensor[[T_DYN, 6144], pl.BF16],
    wkv_weight: pl.Tensor[[25600, 6144], pl.BF16],
    weight: pl.Tensor[[4, 5120], pl.FP32],
    rotation: pl.Tensor[[32, 32], pl.FP32],
    x: pl.Tensor[[T_DYN, 4, 5120], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, 4, 5120], pl.FP32]],
):
    lookup.bind_dynamic(0, T_DYN)
    x.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    result = engram_from_rows(lookup, wkv_weight, weight, rotation, x, output)
    return output


def dequantize_engram_rows(payload: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize int8_sym group-32 rows (payload ``[N, 256]``, scales ``[N, 8]``) to BF16."""
    groups = payload.shape[-1] // ENGRAM_GROUP_SIZE
    values = payload.float().reshape(-1, groups, ENGRAM_GROUP_SIZE) * scales.float().reshape(-1, groups, 1)
    return values.reshape(payload.shape[0], -1).to(torch.bfloat16)


class HostEngramTable:
    """Random row access to a host-offloaded Engram table without loading it whole."""

    def __init__(self, root: str | Path, prefix: str):
        from utils import SafeTensorReader

        self.reader = SafeTensorReader(root)
        self.weight_name = f"{prefix}.embed.weight"
        self.scale_name = f"{prefix}.embed.scale"
        self.rows, self.width = self.reader.shape_and_dtype(self.weight_name)[0]
        self.packed = self.reader.header(self.weight_name)
        self.packed_scale = self.reader.header(self.scale_name)

    @staticmethod
    def _row_bytes(shard: Path, header_size: int, begin: int, row: int, row_bytes: int) -> bytes:
        with shard.open("rb") as handle:
            handle.seek(8 + header_size + begin + row * row_bytes)
            return handle.read(row_bytes)

    def row(self, index: int) -> torch.Tensor:
        """Return one dequantized row as BF16 ``[256]``."""
        shard, header_size, header = self.packed
        begin, _ = header[self.weight_name]["data_offsets"]
        raw = self._row_bytes(shard, header_size, begin, index, self.width)
        payload = torch.frombuffer(bytearray(raw), dtype=torch.int8).reshape(1, self.width)
        shard_s, header_s, header_sv = self.packed_scale
        begin_s, _ = header_sv[self.scale_name]["data_offsets"]
        groups = self.width // ENGRAM_GROUP_SIZE
        raw_scale = self._row_bytes(shard_s, header_s, begin_s, index, groups * 4)
        scale = torch.frombuffer(bytearray(raw_scale), dtype=torch.float32).reshape(1, groups)
        return dequantize_engram_rows(payload, scale)[0]

    def lookup(self, hash_ids: torch.Tensor) -> torch.Tensor:
        """Gather and flatten one lookup row per token; ``hash_ids`` is ``[T, 24]``."""
        flat = hash_ids.reshape(-1).tolist()
        rows = torch.stack([self.row(int(index)) for index in flat])
        return rows.reshape(hash_ids.shape[0], ENGRAM_K)


def golden_engram(tensors):
    """Torch reference mirroring the released Engram gating computation."""
    rows = tensors["lookup"].float()
    kv = torch.matmul(rows, tensors["wkv_weight"].float().t()).to(torch.bfloat16).float()
    key = kv[:, : HC_MULT * D].unflatten(-1, (HC_MULT, D))
    value = kv[:, HC_MULT * D :]
    h = tensors["x"].float()
    original = (h.unflatten(-1, (-1, 32)) @ tensors["rotation"].float().t()).flatten(-2)
    source_weight = tensors["weight"].float().unsqueeze(0)
    rstd = torch.rsqrt(original.square().mean(-1, keepdim=True) + NORM_EPS) * torch.rsqrt(
        key.square().mean(-1, keepdim=True) + NORM_EPS
    )
    dot = (original * source_weight * key).sum(-1, keepdim=True) * rstd * DOT_SCALE
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(CLAMP).sqrt(), dot))
    tensors["output"][:] = h + gate * value.unsqueeze(1)


def fixture_rotation():
    """Return a nonidentity orthogonal block for checkpoint-free Engram fixtures."""
    return torch.block_diag(*[torch.tensor([[.6, -.8], [.8, .6]])] * 16)


def build_tensor_specs(tokens: int = 8, table_rows: int = 1024):
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(11)
    table = torch.randn(table_rows, HEAD_DIM, generator=generator).to(torch.bfloat16)
    hash_ids = torch.randint(0, table_rows, (tokens, N_HASH_COLS), generator=generator)
    lookup = table[hash_ids.long()].reshape(tokens, ENGRAM_K)

    return [
        TensorSpec("lookup", [tokens, ENGRAM_K], torch.bfloat16, init_value=lookup),
        TensorSpec("wkv_weight", [KV_OUT, ENGRAM_K], torch.bfloat16, init_value=lambda: torch.randn(KV_OUT, ENGRAM_K) * 0.02),
        TensorSpec("weight", [HC_MULT, D], torch.float32, init_value=lambda: torch.randn(HC_MULT, D) * 0.1),
        TensorSpec("rotation", [32, 32], torch.float32, init_value=fixture_rotation()),
        TensorSpec("x", [tokens, HC_MULT, D], torch.float32, init_value=lambda: torch.randn(tokens, HC_MULT, D)),
        TensorSpec("output", [tokens, HC_MULT, D], torch.float32),
    ]


def _host_dequant_check() -> None:
    """Check the INT8 group-32 dequant contract against a direct expression."""
    generator = torch.Generator().manual_seed(5)
    payload = torch.randint(-127, 128, (3, HEAD_DIM), generator=generator).to(torch.int8)
    scales = (torch.rand(3, HEAD_DIM // ENGRAM_GROUP_SIZE, generator=generator) * 0.05).float()
    actual = dequantize_engram_rows(payload, scales)
    expected = (
        payload.float().reshape(3, HEAD_DIM // ENGRAM_GROUP_SIZE, ENGRAM_GROUP_SIZE)
        * scales.reshape(3, HEAD_DIM // ENGRAM_GROUP_SIZE, 1)
    ).reshape(3, HEAD_DIM)
    assert torch.equal(actual.float(), expected.to(torch.bfloat16).float()), "engram dequant mismatch"


def build_semantic_specs(tokens):
    """Signed one-column projection with asymmetric keys, values and original channel weights."""
    from golden import TensorSpec

    lookup = torch.zeros(tokens, ENGRAM_K, dtype=torch.bfloat16)
    lookup[:, 0] = 1
    projection = torch.zeros(KV_OUT, ENGRAM_K, dtype=torch.bfloat16)
    projection[:, 0] = ((torch.arange(KV_OUT).remainder(31) - 15) * .125).to(torch.bfloat16)
    x = ((torch.arange(tokens * HC_MULT * D).reshape(tokens, HC_MULT, D).remainder(23) - 11) * .0625).float()
    weight = ((torch.arange(HC_MULT * D).reshape(HC_MULT, D).remainder(19) - 9) * .5).float()
    rotation = fixture_rotation()
    return [
        TensorSpec("lookup", list(lookup.shape), lookup.dtype, init_value=lookup),
        TensorSpec("wkv_weight", list(projection.shape), projection.dtype, init_value=projection),
        TensorSpec("weight", list(weight.shape), weight.dtype, init_value=weight),
        TensorSpec("rotation", [32, 32], rotation.dtype, init_value=rotation),
        TensorSpec("x", list(x.shape), x.dtype, init_value=x),
        TensorSpec("output", list(x.shape), torch.float32),
    ]


def golden_engram_semantic(tensors):
    """Closed-form one-column projection and paired inverse rotations in FP32."""
    h = tensors["x"].float()
    pairs = h.reshape(*h.shape[:-1], -1, 2)
    real = pairs[..., 0] * .6 - pairs[..., 1] * .8
    imag = pairs[..., 0] * .8 + pairs[..., 1] * .6
    original = torch.stack((real, imag), dim=-1).flatten(-2)
    kv = ((torch.arange(KV_OUT).remainder(31) - 15) * .125).to(torch.bfloat16).float()
    key = kv[:HC_MULT * D].reshape(HC_MULT, D)
    value = kv[HC_MULT * D:]
    dot = (original * tensors["weight"] * key).sum(-1, keepdim=True)
    inverse = torch.rsqrt(original.square().mean(-1, keepdim=True) + NORM_EPS)
    inverse = inverse * torch.rsqrt(key.square().mean(-1, keepdim=True) + NORM_EPS)
    dot = dot * inverse * DOT_SCALE
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(CLAMP).sqrt(), dot))
    tensors["output"][:] = h + gate * value


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    from config import add_parallel_arguments
    add_parallel_arguments(parser)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    _host_dequant_check()
    result = run(
        fn=engram_test,
        specs=build_tensor_specs(args.tokens),
        golden_fn=golden_engram,
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        rtol=1e-2,
        atol=1e-2,
        compare_fn={"output": ratio_allclose(atol=1e-2, rtol=1.0 / 128, max_error_ratio=5e-3)},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)

    for tokens, zero_gate in ((1, False), (7, False), (1, True)):
        print(f"[CASE] original-basis Engram sparse T={tokens}", flush=True)
        specs = build_semantic_specs(tokens)
        if zero_gate:
            next(s for s in specs if s.name == "weight").init_value = torch.zeros(HC_MULT, D)
        result = run(fn=engram_test, specs=specs, golden_fn=golden_engram_semantic,
                     config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
                     rtol=1e-4, atol=1e-4,
                     compare_fn={"output": ratio_allclose(atol=1e-4, rtol=1e-4, max_error_ratio=0)})
        if not result.passed:
            raise SystemExit(result.error or 1)

    torch.manual_seed(1907)
    print("[CASE] original BF16 projection / FP32 residual T=3", flush=True)
    result = run(fn=engram_test, specs=build_tensor_specs(3), golden_fn=golden_engram,
                 config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
                 compare_fn={"output": ratio_allclose(atol=1e-4, rtol=1e-4, max_error_ratio=.005)})
    if not result.passed:
        raise SystemExit(result.error or 1)
