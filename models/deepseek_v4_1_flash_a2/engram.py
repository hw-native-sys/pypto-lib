# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-card BF16 Engram in the unrotated checkpoint basis.

The caller supplies resident table rows and their indices. Host hashing,
large-table offload and cross-rank gathers remain outside this kernel.
"""
import torch
import pypto.language as pl

D, HC, COLS, ROW_WIDTH = 5120, 4, 24, 256
K, N = COLS * ROW_WIDTH, (HC + 1) * D
TILE, KT, NT, DT = 16, 128, 256, 512


def make_engram(capacity, table_rows):
    if capacity % TILE or table_rows < 1:
        raise ValueError('Engram needs padded token capacity and a nonempty resident table')

    @pl.jit
    def engram(
        hash_ids: pl.Tensor[[capacity, COLS], pl.INT32],
        table: pl.Tensor[[table_rows, ROW_WIDTH], pl.BF16],
        wkv: pl.Tensor[[K, N], pl.BF16],
        channel_weight: pl.Tensor[[HC, D], pl.FP32],
        hidden: pl.Tensor[[capacity, HC, D], pl.BF16],
        token_mask: pl.Tensor[[capacity], pl.INT32],
        lookup: pl.InOut[pl.Tensor[[capacity, K], pl.BF16]],
        kv: pl.InOut[pl.Tensor[[capacity, N], pl.BF16]],
        output: pl.InOut[pl.Tensor[[capacity, HC, D], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for token in pl.spmd(active, name_hint='a2_engram_lookup'):
            for column in pl.range(COLS):
                row = pl.read(hash_ids, [token, column])
                lookup[token:token + 1, column * ROW_WIDTH:(column + 1) * ROW_WIDTH] = pl.slice(table, [1, ROW_WIDTH], [row, 0])

        blocks = pl.system.available_cluster_count()
        for core in pl.spmd(blocks, name_hint='a2_engram_projection'):
            for task in pl.range(core, ((active + TILE - 1) // TILE) * (N // NT), blocks):
                t0 = (task // (N // NT)) * TILE
                n0 = (task % (N // NT)) * NT
                rows = pl.min(TILE, active - t0)
                a = pl.slice(lookup, [TILE, KT], [t0, 0], valid_shape=[rows, KT])
                b = wkv[:KT, n0:n0 + NT]
                acc = pl.matmul(a, b, out_dtype=pl.FP32)
                for k0 in pl.pipeline(KT, K, KT, stage=2):
                    a = pl.slice(lookup, [TILE, KT], [t0, k0], valid_shape=[rows, KT])
                    b = wkv[k0:k0 + KT, n0:n0 + NT]
                    acc = pl.matmul_acc(acc, a, b)
                # ReplicatedLinear returns BF16 before the FP32 gate arithmetic.
                kv[t0:t0 + TILE, n0:n0 + NT] = pl.set_validshape(pl.cast(acc, pl.BF16, mode='rint'), rows, NT)

        flat_hidden = pl.reshape(hidden, [capacity, HC * D])
        flat_output = pl.reshape(output, [capacity, HC * D])
        mask = pl.reshape(token_mask, [1, capacity])
        for task in pl.spmd(((active + TILE - 1) // TILE) * HC, name_hint='a2_engram_gate'):
            t0, copy = (task // HC) * TILE, task % HC
            rows = pl.min(TILE, active - t0)
            sx = pl.full([1, TILE], dtype=pl.FP32, value=0.)
            sk = pl.full([1, TILE], dtype=pl.FP32, value=0.)
            dot = pl.full([1, TILE], dtype=pl.FP32, value=0.)
            for d0 in pl.range(0, D, DT):
                x = pl.cast(pl.slice(flat_hidden, [TILE, DT], [t0, copy * D + d0], valid_shape=[rows, DT]), pl.FP32)
                key = pl.cast(pl.slice(kv, [TILE, DT], [t0, copy * D + d0], valid_shape=[rows, DT]), pl.FP32)
                weight = channel_weight[copy:copy + 1, d0:d0 + DT]
                sx = pl.add(sx, pl.reshape(pl.row_sum(pl.mul(x, x)), [1, TILE]))
                sk = pl.add(sk, pl.reshape(pl.row_sum(pl.mul(key, key)), [1, TILE]))
                dot = pl.add(dot, pl.reshape(pl.row_sum(pl.mul(pl.col_expand_mul(x, weight), key)), [1, TILE]))
            inv = pl.mul(pl.rsqrt(pl.add(pl.mul(sx, 1. / D), 1e-20)), pl.rsqrt(pl.add(pl.mul(sk, 1. / D), 1e-20)))
            dot = pl.mul(pl.mul(dot, inv), D**-0.5)
            magnitude = pl.sqrt(pl.maximum(pl.abs(dot), 1e-6))
            # Signed FP32 bits preserve -0 without A2-unsupported INT32 bitwise AND.
            bits = pl.cast(pl.reinterpret_view(dot, pl.INT32), pl.FP32)
            negative = pl.maximum(pl.minimum(bits, 0.), -1.)
            signed = pl.mul(magnitude, pl.add(pl.mul(negative, 2.), 1.))
            gate = pl.recip(pl.add(pl.exp(pl.neg(signed)), 1.))
            eligible = pl.cast(pl.slice(mask, [1, TILE], [0, t0], valid_shape=[1, rows]), pl.FP32)
            gate = pl.reshape(pl.mul(gate, eligible), [TILE, 1])
            for d0 in pl.range(0, D, DT):
                x = pl.cast(pl.slice(flat_hidden, [TILE, DT], [t0, copy * D + d0], valid_shape=[rows, DT]), pl.FP32)
                value = pl.cast(pl.slice(kv, [TILE, DT], [t0, HC * D + d0], valid_shape=[rows, DT]), pl.FP32)
                result = pl.cast(pl.add(x, pl.row_expand_mul(value, gate)), pl.BF16, mode='rint')
                flat_output[t0:t0 + TILE, copy * D + d0:copy * D + d0 + DT] = pl.set_validshape(result, rows, DT)
        return lookup, kv, output

    return engram


# Decoder aliases retain the standalone A2/A3 contract names while sharing
# the same dimensions as the prefill Engram implementation above.
HIDDEN, HASH_COLS, HEAD_DIM = D, COLS, ROW_WIDTH
LOOKUP_WIDTH, KV_WIDTH = K, N
M_TILE, GATE_TILE = TILE, DT


def engram_reference(
    hash_ids: torch.Tensor,
    table: torch.Tensor,
    wkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    hidden: torch.Tensor,
    token_mask: torch.Tensor,
    eps: float = 1.0e-20,
) -> dict[str, torch.Tensor]:
    """CPU reference mirroring vLLM-Ascend's unrotated decoder Engram path."""

    lookup = table[hash_ids.long()].reshape(hash_ids.shape[0], LOOKUP_WIDTH)
    kv = torch.matmul(lookup.float(), wkv.float()).to(torch.bfloat16)
    key = kv[:, :HC * HIDDEN].reshape(kv.shape[0], HC, HIDDEN).float()
    value = kv[:, HC * HIDDEN:]
    source = hidden.float()
    channel_weight = q_weight.float() * k_weight.float()
    source_rstd = torch.rsqrt(source.square().mean(-1, keepdim=True) + eps)
    key_rstd = torch.rsqrt(key.square().mean(-1, keepdim=True) + eps)
    dot = (source * channel_weight.reshape(1, HC, HIDDEN) * key).sum(-1, keepdim=True)
    dot = dot * source_rstd * key_rstd * (HIDDEN ** -0.5)
    signed_root = torch.copysign(dot.abs().clamp_min(1.0e-6).sqrt(), dot)
    gate = torch.sigmoid(signed_root) * token_mask.reshape(-1, 1, 1).float()
    output = (source + gate * value.float().unsqueeze(1)).to(torch.bfloat16)
    return {"lookup": lookup, "kv": kv, "output": output}


def make_lookup(capacity: int, table_rows: int):
    @pl.jit
    def lookup(
        hash_ids: pl.Tensor[[capacity, HASH_COLS], pl.INT32],
        table: pl.Tensor[[table_rows, HEAD_DIM], pl.BF16],
        output: pl.InOut[pl.Tensor[[capacity, LOOKUP_WIDTH], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for token in pl.spmd(capacity, name_hint="dsv41_a2a3_engram_lookup"):
            if token < active:
                for column in pl.range(HASH_COLS):
                    row = pl.read(hash_ids, [token, column])
                    output[token:token + 1, column * HEAD_DIM:(column + 1) * HEAD_DIM] = table[
                        row:row + 1, :
                    ]
        return output

    return lookup


def make_gate(capacity: int, eps: float = 1.0e-20):
    @pl.jit
    def gate(
        hidden: pl.Tensor[[capacity, HC, HIDDEN], pl.BF16],
        kv: pl.Tensor[[capacity, KV_WIDTH], pl.BF16],
        channel_weight: pl.Tensor[[HC, HIDDEN], pl.FP32],
        token_mask: pl.Tensor[[capacity], pl.INT32],
        output: pl.InOut[pl.Tensor[[capacity, HC, HIDDEN], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        hidden_flat = pl.reshape(hidden, [capacity, HC * HIDDEN])
        output_flat = pl.reshape(output, [capacity, HC * HIDDEN])
        mask_flat = pl.reshape(token_mask, [1, capacity])
        for block in pl.spmd(((active + M_TILE - 1) // M_TILE) * HC,
                             name_hint="dsv41_a2a3_engram_gate"):
            token = (block // HC) * M_TILE
            copy = block % HC
            rows = pl.min(M_TILE, active - token)
            source_rms = pl.full([1, M_TILE], dtype=pl.FP32, value=0.0)
            key_rms = pl.full([1, M_TILE], dtype=pl.FP32, value=0.0)
            dot = pl.full([1, M_TILE], dtype=pl.FP32, value=0.0)
            for d0 in pl.range(0, HIDDEN, GATE_TILE):
                x = pl.cast(pl.slice(hidden_flat, [M_TILE, GATE_TILE], [token, copy * HIDDEN + d0], valid_shape=[rows, GATE_TILE]), pl.FP32)
                key = pl.cast(pl.slice(kv, [M_TILE, GATE_TILE], [token, copy * HIDDEN + d0], valid_shape=[rows, GATE_TILE]), pl.FP32)
                weight = channel_weight[copy:copy + 1, d0:d0 + GATE_TILE]
                source_rms = pl.add(source_rms, pl.reshape(pl.row_sum(pl.mul(x, x)), [1, M_TILE]))
                key_rms = pl.add(key_rms, pl.reshape(pl.row_sum(pl.mul(key, key)), [1, M_TILE]))
                dot = pl.add(dot, pl.reshape(pl.row_sum(pl.col_expand_mul(pl.mul(x, key), weight)), [1, M_TILE]))
            inv = pl.mul(
                pl.rsqrt(pl.add(pl.mul(source_rms, 1.0 / HIDDEN), eps)),
                pl.rsqrt(pl.add(pl.mul(key_rms, 1.0 / HIDDEN), eps)),
            )
            dot = pl.mul(pl.mul(dot, inv), HIDDEN ** -0.5)
            magnitude = pl.sqrt(pl.maximum(pl.abs(dot), 1.0e-6))
            bits = pl.cast(pl.reinterpret_view(dot, pl.INT32), pl.FP32)
            negative = pl.maximum(pl.minimum(bits, 0.0), -1.0)
            signed = pl.mul(magnitude, pl.add(pl.mul(negative, 2.0), 1.0))
            gate_value = pl.recip(pl.add(pl.exp(pl.neg(signed)), 1.0))
            mask = pl.cast(pl.slice(mask_flat, [1, M_TILE], [0, token], valid_shape=[1, rows]), pl.FP32)
            gate_value = pl.reshape(pl.mul(gate_value, pl.reshape(mask, [1, M_TILE])), [M_TILE, 1])
            for d0 in pl.range(0, HIDDEN, GATE_TILE):
                x = pl.cast(pl.slice(hidden_flat, [M_TILE, GATE_TILE], [token, copy * HIDDEN + d0], valid_shape=[rows, GATE_TILE]), pl.FP32)
                value = pl.cast(pl.slice(kv, [M_TILE, GATE_TILE], [token, HC * HIDDEN + d0], valid_shape=[rows, GATE_TILE]), pl.FP32)
                result = pl.cast(pl.add(x, pl.row_expand_mul(value, gate_value)), pl.BF16, mode="rint")
                output_flat[token:token + M_TILE, copy * HIDDEN + d0:copy * HIDDEN + d0 + GATE_TILE] = pl.set_validshape(result, rows, GATE_TILE)
        return output

    return gate
