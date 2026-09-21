# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4.1-Flash Engram block: gated n-gram lookup into the residual stream.

Single-rank mirror of the released ``inference/model.py`` ``Engram.forward``
(world_size == 1). The n-gram hashing runs host-side; this kernel starts from
the precomputed per-position table row ids and evaluates, per token tile,

    rows  = engram_table[hash_ids]                 # 24 gathered rows of 256
    kv    = flatten(rows) @ wkv_weight             # [T,6144] @ [6144, 25600]
    key, value = split(kv, [HC_MULT*D, D])         # key per hc copy, one value
    dot   = sum(x * weight * key, -1) * rms(x) * rms(key) * D**-0.5
    gate  = sigmoid(sign(dot) * sqrt(max(|dot|, clamp)))
    out   = x + gate * value                       # value shared across copies

``weight`` is the precomputed per-channel scale ``q_weight * k_weight`` from the
released module (folded host-side into a single FP32 tensor so the kernel does
not pay a BF16 rounding on the gate path). ``token_mask`` is a no-op here
(text-only, no image spans), so the gate is never forced to zero. The table
stays BF16 (a stand-in for the released FP8 rows; the golden reference
dequantizes the same BF16 values, and the acceptance budget covers the
FP8-vs-BF16 gap).

Tensor-parallel mode (``--tp P``): the table is row-sharded across ``P`` ranks
(``ParallelEmbedding`` style, ``ROWS_PER_RANK = NUM_EMBEDDINGS // P`` rows per
rank). Every rank gathers the full hash-id set against its own shard, writes
zeros for off-shard rows, publishes the partial lookup to its HCCL window, and
all-reduces it chunk-wise inside the projection loop, so every rank produces
the full output locally.
"""

import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# A5-only; intentionally excluded from the A2/A3 device sweep. `ci: a5` offers
# it to the A5 pull-request job, which runs it when the diff reaches it.
# ci: no-sim
# ci: a5

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from models.deepseek_v4_1_flash.config import D, FLASH, HC_MULT, T_DYN


N_HASH_COLS = (FLASH.engram_max_ngram_size - 1) * FLASH.engram_n_heads  # 24
HEAD_DIM = FLASH.engram_head_dim  # 256
ENGRAM_K = N_HASH_COLS * HEAD_DIM  # 6144 flattened lookup width
KV_OUT = (HC_MULT + 1) * D  # 5 * 5120 = 25600
# Validation-sized table; the released layer-1 table (384006168 rows) does not fit
# in host memory for a standalone case, so this exercises the same kernel against a
# much smaller row count. The kernel only reads gathered rows, so any size works.
NUM_EMBEDDINGS = 1 << 20  # 1048576 rows (~512 MiB as BF16)
CLAMP = 1e-6
NORM_EPS = FLASH.rms_norm_eps
D_INV = 1.0 / D
DOT_SCALE = D**-0.5

T_TILE = 16
K_TILE = 128
N_TILE = 256
D_TILE = 512


def _parse_tp_size() -> int:
    """Read ``--tp`` from the command line (default 1 = single-rank)."""
    for index, argument in enumerate(sys.argv):
        if argument == "--tp" and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if argument.startswith("--tp="):
            return int(argument.split("=", 1)[1])
    return 1


TP_SIZE = _parse_tp_size()
if TP_SIZE not in (1, 2, 4, 8):
    raise ValueError(f"--tp must be one of (1, 2, 4, 8), got {TP_SIZE}")
if NUM_EMBEDDINGS % TP_SIZE:
    raise ValueError(f"NUM_EMBEDDINGS={NUM_EMBEDDINGS} not divisible by TP{TP_SIZE}")
ROWS_PER_RANK = NUM_EMBEDDINGS // TP_SIZE
# Static row capacity of the per-rank lookup window used by the TP all-reduce.
TP_MAX_TOKENS = 256


@pl.jit.inline
def engram_gate(
    kv: pl.Tensor[[T_DYN, KV_OUT], pl.FP32],
    weight: pl.Tensor[[HC_MULT, D], pl.FP32],
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    out: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
):
    """Per (token block, hc copy) gate and residual add on a projected kv."""
    t_dim = pl.tensor.dim(kv, 0)
    x_flat = pl.reshape(x, [t_dim, HC_MULT * D])
    out_flat = pl.reshape(out, [t_dim, HC_MULT * D])
    t_blocks = (t_dim + T_TILE - 1) // T_TILE

    for block in pl.spmd(t_blocks * HC_MULT, name_hint="engram_gate"):
        t0 = (block // HC_MULT) * T_TILE
        c = block % HC_MULT
        valid_rows = pl.min(T_TILE, t_dim - t0)

        # Pass A: walk D in D_TILE chunks, accumulate sq_x / sq_k / dot.
        # Row-byte alignment: a [T_TILE, 1] fp32 tile has a 4-byte row and the
        # allocator requires 32; keep accumulators as [1, T_TILE] instead.
        sq_x = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        sq_k = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        dot = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for d0 in pl.range(0, D, D_TILE):
            x_d = pl.cast(
                pl.slice(
                    x_flat,
                    [T_TILE, D_TILE],
                    [t0, c * D + d0],
                    valid_shape=[valid_rows, D_TILE],
                ),
                target_type=pl.FP32,
            )
            key_d = pl.slice(
                kv,
                [T_TILE, D_TILE],
                [t0, c * D + d0],
                valid_shape=[valid_rows, D_TILE],
            )
            sq_x = pl.add(sq_x, pl.reshape(pl.row_sum(pl.mul(x_d, x_d)), [1, T_TILE]))
            sq_k = pl.add(sq_k, pl.reshape(pl.row_sum(pl.mul(key_d, key_d)), [1, T_TILE]))
            w_d = pl.slice(weight, [1, D_TILE], [c, d0])
            xw_d = pl.col_expand_mul(x_d, w_d)
            dot = pl.add(dot, pl.reshape(pl.row_sum(pl.mul(xw_d, key_d)), [1, T_TILE]))

        # gate = sigmoid(sign(dot) * sqrt(max(|dot|, clamp)))
        # sign(dot)*sqrt(|dot|) == dot / sqrt(|dot|); the clamp keeps the
        # denominator positive, matching copysign(sqrt(|dot|), dot).
        inv_x = pl.rsqrt(pl.add(pl.mul(sq_x, D_INV), NORM_EPS))
        inv_k = pl.rsqrt(pl.add(pl.mul(sq_k, D_INV), NORM_EPS))
        dot = pl.mul(pl.mul(pl.mul(dot, inv_x), inv_k), DOT_SCALE)
        mag = pl.maximum(pl.abs(dot), CLAMP)
        signed = pl.div(dot, pl.sqrt(mag))
        gate = pl.reshape(
            pl.recip(pl.add(pl.exp(pl.neg(signed)), 1.0)), [T_TILE, 1]
        )

        # Pass B: walk D again, compute y = x + gate * value
        for d0 in pl.range(0, D, D_TILE):
            x_d = pl.cast(
                pl.slice(
                    x_flat,
                    [T_TILE, D_TILE],
                    [t0, c * D + d0],
                    valid_shape=[valid_rows, D_TILE],
                ),
                target_type=pl.FP32,
            )
            value_d = pl.slice(
                kv,
                [T_TILE, D_TILE],
                [t0, HC_MULT * D + d0],
                valid_shape=[valid_rows, D_TILE],
            )
            gated = pl.row_expand_mul(value_d, gate)
            y = pl.cast(pl.add(x_d, gated), target_type=pl.BF16, mode="rint")
            out_flat[t0 : t0 + T_TILE, c * D + d0 : c * D + d0 + D_TILE] = (
                pl.set_validshape(y, valid_rows, D_TILE)
            )
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
    engram_gate(kv, weight, x, out)
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
    rstd = torch.rsqrt(h.square().mean(-1, keepdim=True) + FLASH.rms_norm_eps) * torch.rsqrt(
        key.square().mean(-1, keepdim=True) + FLASH.rms_norm_eps
    )
    dot = (h * w * key).sum(-1, keepdim=True) * rstd * (D**-0.5)  # [T, HC_MULT, 1]
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(CLAMP).sqrt(), dot))
    return (h + gate * value.unsqueeze(1)).to(torch.bfloat16)


@pl.jit
def engram_test(
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


@pl.jit.inline
def engram_tp(
    hash_ids: pl.Tensor[[T_DYN, N_HASH_COLS], pl.INT32],
    engram_table: pl.Tensor[[ROWS_PER_RANK, HEAD_DIM], pl.BF16],
    wkv_weight: pl.Tensor[[ENGRAM_K, KV_OUT], pl.BF16],
    weight: pl.Tensor[[HC_MULT, D], pl.FP32],
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    out: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    lookup_window: pld.DistributedTensor[[TP_MAX_TOKENS, ENGRAM_K], pl.BF16],
    signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """One TP rank: gather against the local table shard, all-reduce the lookup.

    The gathered rows are linear in the table, so summing every rank's
    zero-masked partial lookup reproduces the full-table gather. The reduction
    is fused chunk-wise into the projection loop to keep tiles small.
    """
    t_dim = pl.tensor.dim(hash_ids, 0)
    t_blocks = (t_dim + T_TILE - 1) // T_TILE
    n_blocks = KV_OUT // N_TILE

    # ---- Stage 0: masked gather from this rank's shard (off-shard rows -> 0)
    lookup_partial = pl.create_tensor([t_dim, ENGRAM_K], dtype=pl.BF16)
    with pl.spmd(t_blocks, name_hint="engram_tp_gather") as gather_tid:
        t0 = pl.tile.get_block_idx() * T_TILE
        valid_rows = pl.min(T_TILE, t_dim - t0)
        lookup = pl.create_tensor([T_TILE, ENGRAM_K], dtype=pl.BF16)
        for c in pl.range(N_HASH_COLS):
            for tt in pl.range(T_TILE):
                if t0 + tt < t_dim:
                    row = pl.read(hash_ids, [t0 + tt, c])
                    local = row - my_rank * ROWS_PER_RANK
                    clamped = pl.max(0, pl.min(ROWS_PER_RANK - 1, local))
                    if local >= 0 and local < ROWS_PER_RANK:
                        lookup[tt : tt + 1, c * HEAD_DIM : (c + 1) * HEAD_DIM] = engram_table[
                            clamped : clamped + 1, 0:HEAD_DIM
                        ]
                    else:
                        lookup[tt : tt + 1, c * HEAD_DIM : (c + 1) * HEAD_DIM] = pl.full(
                            [1, HEAD_DIM], dtype=pl.BF16, value=0.0
                        )
                else:
                    lookup[tt : tt + 1, c * HEAD_DIM : (c + 1) * HEAD_DIM] = pl.full(
                        [1, HEAD_DIM], dtype=pl.BF16, value=0.0
                    )
        lookup_partial[t0 : t0 + T_TILE, 0:ENGRAM_K] = pl.set_validshape(
            lookup, valid_rows, ENGRAM_K
        )

    # Publish this rank's partial lookup into its own window slice, then barrier.
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="engram_tp_publish", deps=[gather_tid]
    ) as publish_tid:
        pld.tensor.put(
            dst=lookup_window,
            peer=my_rank,
            src=lookup_partial,
            dst_offsets=[0, 0],
            src_offsets=[0, 0],
            shape=[t_dim, ENGRAM_K],
            chunk_rows=1,
            chunk_cols=2048,
        )
        for peer in pl.range(TP_SIZE):
            if peer != my_rank:
                pld.system.notify(
                    target=signal,
                    peer=peer,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="engram_tp_wait",
        deps=[publish_tid],
        allow_early_resolve=False,
    ) as wait_tid:
        for src in pl.range(TP_SIZE):
            if src != my_rank:
                pld.system.wait(
                    signal=signal,
                    offsets=[src, 0],
                    expected=1,
                    cmp=pld.WaitCmp.Ge,
                )

    # ---- Stage 1: all-reduce the lookup chunk-wise, project to key|value.
    # Every element has exactly one non-zero contributor across the ranks, so
    # the BF16 adds are exact and the matmul sees the full-table gather. The
    # first K chunk goes through pl.matmul so the accumulator is produced in
    # Acc memory; later chunks accumulate onto it.
    kv = pl.create_tensor([t_dim, KV_OUT], dtype=pl.FP32)
    with pl.spmd(t_blocks * n_blocks, name_hint="engram_tp_matmul", deps=[wait_tid]):
        block = pl.tile.get_block_idx()
        t0 = (block // n_blocks) * T_TILE
        n0 = (block % n_blocks) * N_TILE
        valid_rows = pl.min(T_TILE, t_dim - t0)
        a0 = pl.load(
            lookup_window,
            [t0, 0],
            [T_TILE, K_TILE],
            target_memory=pl.MemorySpace.Vec,
        )
        for peer in pl.range(TP_SIZE):
            if peer != my_rank:
                a0 = pl.add(
                    a0,
                    pld.tile.remote_load(
                        lookup_window, peer=peer, offsets=[t0, 0], shape=[T_TILE, K_TILE]
                    ),
                )
        w0 = pl.load(wkv_weight, [0, n0], [K_TILE, N_TILE])
        acc = pl.matmul(a0, w0, out_dtype=pl.FP32)
        for kb in pl.range(1, ENGRAM_K // K_TILE):
            k0 = kb * K_TILE
            a_tile = pl.load(
                lookup_window,
                [t0, k0],
                [T_TILE, K_TILE],
                target_memory=pl.MemorySpace.Vec,
            )
            for peer in pl.range(TP_SIZE):
                if peer != my_rank:
                    a_tile = pl.add(
                        a_tile,
                        pld.tile.remote_load(
                            lookup_window,
                            peer=peer,
                            offsets=[t0, k0],
                            shape=[T_TILE, K_TILE],
                        ),
                    )
            w_tile = pl.load(wkv_weight, [k0, n0], [K_TILE, N_TILE])
            acc = pl.matmul_acc(acc, a_tile, w_tile)
        pl.store(pl.set_validshape(acc, valid_rows, N_TILE), [t0, n0], kv)

    # ---- Stage 2: per (token block, hc copy) gate and residual add
    engram_gate(kv, weight, x, out)
    return out


@pl.jit
def engram_tp_rank(
    hash_ids: pl.Tensor[[T_DYN, N_HASH_COLS], pl.INT32],
    engram_table: pl.Tensor[[ROWS_PER_RANK, HEAD_DIM], pl.BF16],
    wkv_weight: pl.Tensor[[ENGRAM_K, KV_OUT], pl.BF16],
    weight: pl.Tensor[[HC_MULT, D], pl.FP32],
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]],
    lookup_window: pld.DistributedTensor[[TP_MAX_TOKENS, ENGRAM_K], pl.BF16],
    signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Run the Engram block on one TP rank for standalone validation."""
    hash_ids.bind_dynamic(0, T_DYN)
    x.bind_dynamic(0, T_DYN)
    out.bind_dynamic(0, T_DYN)
    engram_tp(
        hash_ids, engram_table, wkv_weight, weight, x, out,
        lookup_window, signal, my_rank,
    )
    return out


@pl.jit.host
def engram_tp_group(
    hash_ids: pl.Tensor[[TP_SIZE, T_DYN, N_HASH_COLS], pl.INT32],
    engram_table: pl.Tensor[[TP_SIZE, ROWS_PER_RANK, HEAD_DIM], pl.BF16],
    wkv_weight: pl.Tensor[[TP_SIZE, ENGRAM_K, KV_OUT], pl.BF16],
    weight: pl.Tensor[[TP_SIZE, HC_MULT, D], pl.FP32],
    x: pl.Tensor[[TP_SIZE, T_DYN, HC_MULT, D], pl.BF16],
    out: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, HC_MULT, D], pl.BF16]],
):
    """Launch one Engram TP group, every rank sharing the window buffers."""
    hash_ids.bind_dynamic(1, T_DYN)
    x.bind_dynamic(1, T_DYN)
    out.bind_dynamic(1, T_DYN)

    lookup_window_buf = pld.alloc_window_buffer([TP_MAX_TOKENS, ENGRAM_K], dtype=pl.BF16)
    signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        lookup_window = pld.window(lookup_window_buf, [TP_MAX_TOKENS, ENGRAM_K], dtype=pl.BF16)
        signal = pld.window(signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        engram_tp_rank(
            hash_ids[rank],
            engram_table[rank],
            wkv_weight[rank],
            weight[rank],
            x[rank],
            out[rank],
            lookup_window,
            signal,
            rank,
            device=rank,
        )


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


def build_engram_tp_tensor_specs(batch: int = 2, sequence: int = 4):
    """Build stacked per-rank inputs for TP validation.

    Ids, weights, and activations are identical on every rank; only the table
    differs, holding each rank's row shard of the full table.
    """
    from golden import TensorSpec

    tokens = batch * sequence
    generator = torch.Generator().manual_seed(3)

    def init_hash_ids():
        ids = torch.randint(0, NUM_EMBEDDINGS, (tokens, N_HASH_COLS), generator=generator)
        return ids.unsqueeze(0).repeat(TP_SIZE, 1, 1)

    def init_table():
        full = (torch.randn(NUM_EMBEDDINGS, HEAD_DIM, generator=generator) * 0.02).to(
            torch.bfloat16
        )
        return full.reshape(TP_SIZE, ROWS_PER_RANK, HEAD_DIM)

    def init_wkv():
        w = (torch.randn(ENGRAM_K, KV_OUT, generator=generator) / math.sqrt(ENGRAM_K)).to(
            torch.bfloat16
        )
        return w.unsqueeze(0).repeat(TP_SIZE, 1, 1)

    def init_weight():
        q = torch.rand(HC_MULT, D, generator=generator) + 0.5
        k = torch.rand(HC_MULT, D, generator=generator) + 0.5
        w = q * k
        return w.unsqueeze(0).repeat(TP_SIZE, 1, 1)

    def init_x():
        v = torch.randn(tokens, HC_MULT, D, generator=generator).to(torch.bfloat16)
        return v.unsqueeze(0).repeat(TP_SIZE, 1, 1, 1)

    return [
        TensorSpec("hash_ids", [TP_SIZE, tokens, N_HASH_COLS], torch.int32, init_value=init_hash_ids),
        TensorSpec("engram_table", [TP_SIZE, ROWS_PER_RANK, HEAD_DIM], torch.bfloat16, init_value=init_table),
        TensorSpec("wkv_weight", [TP_SIZE, ENGRAM_K, KV_OUT], torch.bfloat16, init_value=init_wkv),
        TensorSpec("weight", [TP_SIZE, HC_MULT, D], torch.float32, init_value=init_weight),
        TensorSpec("x", [TP_SIZE, tokens, HC_MULT, D], torch.bfloat16, init_value=init_x),
        TensorSpec("out", [TP_SIZE, tokens, HC_MULT, D], torch.bfloat16),
    ]


def golden_engram_tp_case(tensors):
    """Fill the expected TP output: full-table result, identical on every rank."""
    full_table = tensors["engram_table"].reshape(NUM_EMBEDDINGS, HEAD_DIM)
    out = golden_engram(
        tensors["hash_ids"][0],
        full_table,
        tensors["wkv_weight"][0],
        tensors["weight"][0],
        tensors["x"][0],
    )
    tensors["out"][:] = out.unsqueeze(0).expand_as(tensors["out"])


def _precision_compare(name, compare):
    """Report achieved precision before applying the tensor's acceptance budget."""

    def compare_and_report(actual, expected, **kwargs):
        actual_f = actual.double()
        expected_f = expected.double()
        diff = actual_f - expected_f
        rel_l2 = diff.norm() / expected_f.norm().clamp_min(1e-12)
        max_abs = diff.abs().max()
        print(f"[PRECISION] {name} rel_l2={rel_l2.item():.8g} max_abs={max_abs.item():.8g}")
        return compare(actual, expected, **kwargs)

    return compare_and_report


def validate(argv=None):
    """Validate the Engram block on A5 (or its simulator), single-rank or TP."""
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 Engram validation")
    parser.add_argument("-p", "--platform", default="a5sim", choices=["a5", "a5sim"])
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="0",
        help="comma-separated device ids; must provide exactly --tp ids",
    )
    parser.add_argument("--tp", type=int, default=1, choices=[1, 2, 4, 8])
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=4)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    args = parser.parse_args(argv)

    if args.tp != TP_SIZE:
        raise ValueError(f"--tp {args.tp} disagrees with module TP_SIZE {TP_SIZE}")
    device_ids = [int(d) for d in args.device.split(",")]
    if len(device_ids) != TP_SIZE:
        raise ValueError(f"need exactly {TP_SIZE} device ids for tp={TP_SIZE}, got {device_ids}")
    if TP_SIZE > 1 and args.batch * args.sequence > TP_MAX_TOKENS:
        raise ValueError(
            f"batch*sequence={args.batch * args.sequence} exceeds the TP window "
            f"capacity {TP_MAX_TOKENS}"
        )

    if TP_SIZE == 1:
        result = run(
            fn=engram_test,
            specs=build_engram_tensor_specs(args.batch, args.sequence),
            golden_fn=golden_engram_case,
            config={
                "platform": args.platform,
                "device_id": device_ids[0],
                "enable_chip_swimlane": args.enable_chip_swimlane,
            },
            rtol=1e-3,
            atol=1e-3,
            compare_fn={"out": _precision_compare("out", ratio_allclose(atol=1e-3, rtol=1e-2))},
            compile_only=args.compile_only,
        )
    else:
        from pypto.ir import DistributedConfig

        result = run(
            fn=engram_tp_group,
            specs=build_engram_tp_tensor_specs(args.batch, args.sequence),
            golden_fn=golden_engram_tp_case,
            config={
                "platform": args.platform,
                "enable_chip_swimlane": args.enable_chip_swimlane,
                "distributed_config": DistributedConfig(
                    device_ids=device_ids,
                    num_sub_workers=0,
                ),
            },
            rtol=1e-3,
            atol=1e-3,
            compare_fn={"out": _precision_compare("out", ratio_allclose(atol=1e-3, rtol=1e-2))},
            compile_only=args.compile_only,
        )
    return result


__all__ = [
    "build_engram_tensor_specs",
    "build_engram_tp_tensor_specs",
    "engram",
    "engram_gate",
    "engram_test",
    "engram_tp",
    "engram_tp_group",
    "engram_tp_rank",
    "golden_engram",
    "golden_engram_case",
    "golden_engram_tp_case",
]


_SCRIPT_ENTRY_POINT = "__" + "main__"


def main():
    """Run local validation and return a failing exit status on precision errors."""
    result = validate()
    if not result.passed:
        raise SystemExit(result.error or 1)


if "pytest" in sys.modules:
    import pytest

    @pytest.mark.parametrize("tp", [1, 4])
    def test_precision(tp, a5_args):
        """Validate the operator against its golden reference on A5."""
        result = validate(a5_args(tp=tp))
        assert result.passed, result.error

if __name__ == _SCRIPT_ENTRY_POINT:
    main()
