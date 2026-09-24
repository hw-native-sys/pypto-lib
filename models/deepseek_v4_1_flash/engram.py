# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4.1-Flash Engram block: gated n-gram lookup into the residual stream.

Serving mirror of the released ``inference/model.py`` ``Engram.forward``. The
engram tables stay host-resident (each layer is close to a hundred gigabytes);
for every launch the host n-gram hasher turns the token window into
per-position table row ids, gathers each rank's zero-padded TP partial
straight from the mapped checkpoint, and ships the raw FP8E4M3FN payload rows
plus the raw E8M0 scale byte per 32-column group, laid out transposed as
[SCALE_COLS, T] because the A5 load path rejects a ColMajor [T_TILE, 1] Vec
tile from an ND global tensor. A5 tcvt has no e8m0 -> fp32 edge, so the
kernel relays the byte through fp16 to int32 and rebuilds the fp32 bits
(byte << 23, the byte being the biased fp32 exponent field); every legal
scale is a power of two, so the rebuild is bit-exact.
Per token tile the kernel then evaluates

    rows  = dequant(embed_weight, embed_scale)  # this rank's zero-padded partial
    kv    = allreduce(rows) @ wkv_weight         # [T,6144] -> [T, 25600]
    key, value = split(kv, [HC_MULT*D, D])       # key per hc copy, one value
    dot   = sum(x * weight * key, -1) * rms(x) * rms(key) * D**-0.5
    gate  = sigmoid(sign(dot) * sqrt(max(|dot|, clamp)))
    out   = x + gate * value                     # value shared across copies

``weight`` is the precomputed per-channel scale ``q_weight * k_weight`` from
the released module (folded host-side into a single FP32 tensor so the kernel
does not pay a BF16 rounding on the gate path). ``token_mask`` is a no-op here
(text-only, no image spans), so the gate is never forced to zero.

Tensor-parallel mode (``--tp P``): the table is row-sharded across ``P`` ranks
(``ParallelEmbedding`` style). Every rank's partial gathers only its own
shard's rows with zeros elsewhere, publishes the dequantized partial to its
HCCL window, and all-reduces it chunk-wise inside the projection loop, so
every rank produces the full output locally. The zero-padded columns are
mathematically inert (an e4m3 0x00 byte is +0.0 whatever the scale), so the
partial sums reproduce a full-table gather exactly.
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

from models.deepseek_v4_1_flash.config import D, FLASH, HC_MULT, PREFILL_MAX_TOKENS, T_DYN


N_HASH_COLS = (FLASH.engram_max_ngram_size - 1) * FLASH.engram_n_heads  # 24
HEAD_DIM = FLASH.engram_head_dim  # 256
ENGRAM_K = N_HASH_COLS * HEAD_DIM  # 6144 flattened lookup width
SCALE_GROUP = 32  # columns sharing one E8M0 scale, as in the released table
SCALE_COLS = ENGRAM_K // SCALE_GROUP  # 192
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
# Leading epoch axis of the repeat-epoch validation program.
E_DYN = pl.dynamic("ENGRAM_EPOCHS_DYN")


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
    embed_weight: pl.Tensor[[T_DYN, ENGRAM_K], pl.FP8E4M3FN],
    embed_scale: pl.Tensor[[SCALE_COLS, T_DYN], pl.FP8E8M0],
    wkv_weight: pl.Tensor[[ENGRAM_K, KV_OUT], pl.BF16],
    weight: pl.Tensor[[HC_MULT, D], pl.FP32],
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    out: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    lookup_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, ENGRAM_K], pl.BF16],
    signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    """One TP rank: dequantize the host-gathered partial, all-reduce, gate.

    The host gathers this rank's zero-padded TP partial straight from the
    mapped checkpoint, so the kernel starts from the raw FP8E4M3FN payload
    bytes plus the raw E8M0 scale byte per 32-column group. The partial
    is dequantized, published to this rank's HCCL window slice, then reduced
    chunk-wise inside the projection loop so tiles stay small. The zero-padded
    off-shard columns stay exactly zero (an e4m3 0x00 byte is +0.0 whatever
    the scale), so the all-reduce reproduces the full lookup exactly.

    The scale wire is transposed -- [SCALE_COLS, T] instead of [T, SCALE_COLS]
    -- because the A5 load path rejects a ColMajor [T_TILE, 1] Vec tile from
    an ND global tensor ("Src and dst layout must be same"). A [1, T_TILE]
    slice loads RowMajor, and the in-register reshape back to [T_TILE, 1] is
    the same pattern engram_gate uses for its gate vector.

    ``epoch`` is the 1-based dispatch counter over the persistent window and
    signal, following the other V4.1 communication paths: every rank bumps
    each peer's signal slot twice per epoch (publish, then read-done), so
    waiting for ``2*(epoch-1)`` proves the previous round drained before this
    round reuses the window, and ``2*epoch-1`` orders this round's reads
    behind every peer's publish. A fixed threshold would pass early on the
    second execution (the signal never resets) and read a window mid-rewrite.
    """
    t_dim = pl.tensor.dim(embed_weight, 0)
    t_blocks = (t_dim + T_TILE - 1) // T_TILE
    n_blocks = KV_OUT // N_TILE

    # ---- Stage 0: a later epoch must not overwrite the window slice a peer
    # is still reading. Every peer bumps this rank's signal slot twice per
    # epoch (publish, then read-done), so waiting for 2*(epoch-1) proves the
    # previous round fully drained. On the first epoch the threshold is zero
    # and the wait is a no-op.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="engram_previous_epoch",
        allow_early_resolve=False,
    ) as prev_tid:
        for peer in pl.range(TP_SIZE):
            if peer != my_rank:
                pld.system.wait(
                    signal=signal,
                    offsets=[peer, 0],
                    expected=(epoch - 1) * 2,
                    cmp=pld.WaitCmp.Ge,
                )

    # ---- Stage 1: dequantize this rank's partial (fp8 x e8m0, 32-column groups)
    lookup_partial = pl.create_tensor([t_dim, ENGRAM_K], dtype=pl.BF16)
    with pl.spmd(t_blocks, name_hint="engram_dequant"):
        t0 = pl.tile.get_block_idx() * T_TILE
        valid_rows = pl.min(T_TILE, t_dim - t0)
        for g in pl.range(SCALE_COLS):
            # Transposed wire: a [1, T_TILE] E8M0 row loads RowMajor (32-byte
            # tile row); reshape to [T_TILE, 1] in-register for the row scale.
            scale_g = pl.slice(
                embed_scale, [1, T_TILE], [g, t0], valid_shape=[1, valid_rows]
            )
            payload_g = pl.slice(
                embed_weight,
                [T_TILE, SCALE_GROUP],
                [t0, g * SCALE_GROUP],
                valid_shape=[valid_rows, SCALE_GROUP],
            )
            payload_f = pl.cast(payload_g, target_type=pl.FP32)
            # A5 tcvt has no e8m0 -> fp32 edge and uint8 -> int32 has no
            # admissible relay (every 2-hop bridge provably narrows), but
            # uint8 -> fp16 is native (exact for 0..255) and so is
            # fp16 -> int32, so relay the byte through fp16. The e8m0 byte
            # IS the biased fp32 exponent field of its power-of-two scale
            # (2**(b-127) has exponent field b and zero mantissa), so the
            # fp32 bits are just b << 23: bit-exact for every normal byte
            # (1..254; quantized tables never emit the 2**-127 / NaN corner
            # bytes).
            scale_bits = pl.reinterpret_view(scale_g, pl.UINT8)
            scale_byte = pl.cast(scale_bits, target_type=pl.FP16)
            scale_field = pl.cast(scale_byte, target_type=pl.INT32)
            scale_f = pl.reshape(
                pl.reinterpret_view(pl.shls(scale_field, 23), pl.FP32), [T_TILE, 1]
            )
            dequant = pl.row_expand_mul(payload_f, scale_f)
            lookup_partial[t0 : t0 + T_TILE, g * SCALE_GROUP : (g + 1) * SCALE_GROUP] = (
                pl.cast(dequant, target_type=pl.BF16, mode="rint")
            )

    # ---- Stage 2: publish the partial into this rank's window slice. The
    # publish is gated on the previous epoch's reads having drained, then the
    # first per-epoch notify lets peers order their reads behind it.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="engram_publish",
        deps=[prev_tid],
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
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="engram_publish_ready",
        deps=[publish_tid],
    ) as ready_tid:
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
        name_hint="engram_wait",
        deps=[ready_tid],
        allow_early_resolve=False,
    ) as wait_tid:
        for src in pl.range(TP_SIZE):
            if src != my_rank:
                pld.system.wait(
                    signal=signal,
                    offsets=[src, 0],
                    expected=epoch * 2 - 1,
                    cmp=pld.WaitCmp.Ge,
                )

    # ---- Stage 3: chunk-wise all-reduce, project to key|value. Every element
    # has exactly one non-zero contributor across the ranks, so the BF16 adds
    # are exact and the matmul sees the full lookup. The first K chunk goes
    # through pl.matmul so the accumulator is produced in Acc memory; later
    # chunks accumulate onto it.
    kv = pl.create_tensor([t_dim, KV_OUT], dtype=pl.FP32)
    with pl.spmd(
        t_blocks * n_blocks, name_hint="engram_matmul", deps=[wait_tid]
    ) as reduce_tid:
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

    # ---- Stage 3b: read-done. The remote loads above consumed every peer's
    # slice, so the second per-epoch notify frees this rank's window for the
    # next epoch; the trailing wait proves the whole group drained before the
    # gate retires, mirroring prefill_tp_output_all_reduce.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="engram_read_done",
        deps=[reduce_tid],
    ) as release_tid:
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
        name_hint="engram_window_drained",
        deps=[release_tid],
        allow_early_resolve=False,
    ):
        for peer in pl.range(TP_SIZE):
            if peer != my_rank:
                pld.system.wait(
                    signal=signal,
                    offsets=[peer, 0],
                    expected=epoch * 2,
                    cmp=pld.WaitCmp.Ge,
                )

    # ---- Stage 4: per (token block, hc copy) gate and residual add
    engram_gate(kv, weight, x, out)
    return out


@pl.jit
def engram_rank(
    embed_weight: pl.Tensor[[T_DYN, ENGRAM_K], pl.FP8E4M3FN],
    embed_scale: pl.Tensor[[SCALE_COLS, T_DYN], pl.FP8E8M0],
    wkv_weight: pl.Tensor[[ENGRAM_K, KV_OUT], pl.BF16],
    weight: pl.Tensor[[HC_MULT, D], pl.FP32],
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]],
    lookup_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, ENGRAM_K], pl.BF16],
    signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    """Run the Engram block on one TP rank for standalone validation."""
    embed_weight.bind_dynamic(0, T_DYN)
    embed_scale.bind_dynamic(1, T_DYN)
    x.bind_dynamic(0, T_DYN)
    out.bind_dynamic(0, T_DYN)
    engram(
        embed_weight, embed_scale, wkv_weight, weight, x, out,
        lookup_window, signal, my_rank, epoch,
    )
    return out


@pl.jit.host
def engram_group(
    embed_weight: pl.Tensor[[TP_SIZE, T_DYN, ENGRAM_K], pl.FP8E4M3FN],
    embed_scale: pl.Tensor[[TP_SIZE, SCALE_COLS, T_DYN], pl.FP8E8M0],
    wkv_weight: pl.Tensor[[TP_SIZE, ENGRAM_K, KV_OUT], pl.BF16],
    weight: pl.Tensor[[TP_SIZE, HC_MULT, D], pl.FP32],
    x: pl.Tensor[[TP_SIZE, T_DYN, HC_MULT, D], pl.BF16],
    out: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, HC_MULT, D], pl.BF16]],
    epoch: pl.Scalar[pl.INT32],
):
    """Launch one Engram TP group, every rank sharing the window buffers.

    ``epoch`` is the 1-based count of prior executions over the persistent
    window/signal pair; callers advancing across dispatches increment it.
    """
    embed_weight.bind_dynamic(1, T_DYN)
    embed_scale.bind_dynamic(2, T_DYN)
    x.bind_dynamic(1, T_DYN)
    out.bind_dynamic(1, T_DYN)

    lookup_window_buf = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, ENGRAM_K], dtype=pl.BF16)
    signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        lookup_window = pld.window(lookup_window_buf, [PREFILL_MAX_TOKENS, ENGRAM_K], dtype=pl.BF16)
        signal = pld.window(signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        engram_rank(
            embed_weight[rank],
            embed_scale[rank],
            wkv_weight[rank],
            weight[rank],
            x[rank],
            out[rank],
            lookup_window,
            signal,
            rank,
            epoch,
            device=rank,
        )


@pl.jit.host
def engram_group_repeat(
    embed_weight: pl.Tensor[[E_DYN, TP_SIZE, T_DYN, ENGRAM_K], pl.FP8E4M3FN],
    embed_scale: pl.Tensor[[E_DYN, TP_SIZE, SCALE_COLS, T_DYN], pl.FP8E8M0],
    wkv_weight: pl.Tensor[[TP_SIZE, ENGRAM_K, KV_OUT], pl.BF16],
    weight: pl.Tensor[[TP_SIZE, HC_MULT, D], pl.FP32],
    x: pl.Tensor[[E_DYN, TP_SIZE, T_DYN, HC_MULT, D], pl.BF16],
    out: pl.Out[pl.Tensor[[E_DYN, TP_SIZE, T_DYN, HC_MULT, D], pl.BF16]],
    epochs: pl.Scalar[pl.INT32],
):
    """Run several epochs of the Engram block over one persistent window.

    Every epoch gets its own inputs and a fresh 1-based epoch counter while
    the window and signal buffers are allocated once, which is exactly the
    production cadence: dispatch after dispatch reuses the same transport
    storage and the signal never resets.
    """
    embed_weight.bind_dynamic(2, T_DYN)
    embed_scale.bind_dynamic(3, T_DYN)
    x.bind_dynamic(2, T_DYN)
    out.bind_dynamic(2, T_DYN)

    lookup_window_buf = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, ENGRAM_K], dtype=pl.BF16)
    signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for step in pl.range(epochs):
        for rank in pl.range(pld.world_size()):
            lookup_window = pld.window(lookup_window_buf, [PREFILL_MAX_TOKENS, ENGRAM_K], dtype=pl.BF16)
            signal = pld.window(signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
            engram_rank(
                embed_weight[step, rank],
                embed_scale[step, rank],
                wkv_weight[rank],
                weight[rank],
                x[step, rank],
                out[step, rank],
                lookup_window,
                signal,
                rank,
                step + 1,
                device=rank,
            )


def _quantize_rows(full_table: torch.Tensor, hash_ids: torch.Tensor, rank: int):
    """Build one rank's host-gathered partial exactly as serving would.

    Rows whose hash id falls in this rank's shard are quantized per
    32-column group with a power-of-two E8M0 scale (as the released table
    stores them); every other column is a zero payload byte, so the dequantized
    partial is zero there whatever the scale byte says.
    """
    tokens = hash_ids.shape[0]
    per_row_groups = HEAD_DIM // SCALE_GROUP
    generator = torch.Generator().manual_seed(1000 + rank)
    # Powers of two in a modest band keep e4m3 in range after the division.
    exponents = torch.randint(-3, 4, (tokens, SCALE_COLS), generator=generator)
    embed_scale = torch.pow(2.0, exponents.float())  # exact E8M0 values
    embed_weight = torch.zeros(tokens, ENGRAM_K, dtype=torch.uint8)
    for c in range(N_HASH_COLS):
        hit = (hash_ids[:, c] >= rank * ROWS_PER_RANK) & (
            hash_ids[:, c] < (rank + 1) * ROWS_PER_RANK
        )
        if not hit.any():
            continue
        rows = full_table[hash_ids[hit, c].long()].float()  # [nnz, HEAD_DIM]
        group_scales = embed_scale[hit, c * per_row_groups : (c + 1) * per_row_groups]
        grouped = rows.reshape(-1, per_row_groups, SCALE_GROUP) / group_scales.unsqueeze(-1)
        quantized = grouped.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        embed_weight[hit, c * HEAD_DIM : (c + 1) * HEAD_DIM] = (
            quantized.reshape(-1, HEAD_DIM).view(torch.uint8)
        )
    # Exact E8M0 bytes (powers of two); transposed to [SCALE_COLS, T] because
    # the A5 load path rejects a ColMajor [T_TILE, 1] Vec tile from ND global.
    return (
        embed_weight.view(torch.float8_e4m3fn),
        embed_scale.to(torch.float8_e8m0fnu).t().contiguous(),
    )


def _engram_case(batch: int, sequence: int, seed: int):
    """Draw one epoch's hash ids, table, and activations host-side."""
    tokens = batch * sequence
    generator = torch.Generator().manual_seed(seed)
    hash_ids = torch.randint(0, NUM_EMBEDDINGS, (tokens, N_HASH_COLS), generator=generator)
    # small-magnitude rows keep the projection in a numerically tame range
    table = (torch.randn(NUM_EMBEDDINGS, HEAD_DIM, generator=generator) * 0.02).to(
        torch.bfloat16
    )
    x = torch.randn(tokens, HC_MULT, D, generator=generator).to(torch.bfloat16)
    return hash_ids, table, x


def _engram_weights(seed: int):
    """Draw the projection and gate weights shared by every rank."""
    generator = torch.Generator().manual_seed(seed)
    wkv = (torch.randn(ENGRAM_K, KV_OUT, generator=generator) / math.sqrt(ENGRAM_K)).to(
        torch.bfloat16
    )
    # released module keeps q_weight and k_weight separate; fold them
    # host-side so the kernel takes a single per-channel scale
    q = torch.rand(HC_MULT, D, generator=generator) + 0.5
    k = torch.rand(HC_MULT, D, generator=generator) + 0.5
    return wkv, q * k


def build_engram_tensor_specs(batch: int = 2, sequence: int = 4):
    """Build stacked per-rank inputs for Engram validation.

    One full table and one hash-id set are drawn host-side; every rank's
    embed_weight/embed_scale partial is the host-gathered, zero-padded shard
    of that table, and weights/activations are identical on every rank.
    """
    from golden import ScalarSpec, TensorSpec

    tokens = batch * sequence
    hash_ids, table, x = _engram_case(batch, sequence, seed=3)
    wkv, weight = _engram_weights(seed=3)
    embed_weight, embed_scale = [], []
    for rank in range(TP_SIZE):
        embed_weight_rank, embed_scale_rank = _quantize_rows(table, hash_ids, rank)
        embed_weight.append(embed_weight_rank)
        embed_scale.append(embed_scale_rank)
    return [
        TensorSpec(
            "embed_weight",
            [TP_SIZE, tokens, ENGRAM_K],
            torch.float8_e4m3fn,
            init_value=lambda: torch.stack(embed_weight),
        ),
        TensorSpec(
            "embed_scale",
            [TP_SIZE, SCALE_COLS, tokens],
            torch.float8_e8m0fnu,
            init_value=lambda: torch.stack(embed_scale),
        ),
        TensorSpec(
            "wkv_weight",
            [TP_SIZE, ENGRAM_K, KV_OUT],
            torch.bfloat16,
            init_value=lambda: wkv.unsqueeze(0).repeat(TP_SIZE, 1, 1),
        ),
        TensorSpec(
            "weight",
            [TP_SIZE, HC_MULT, D],
            torch.float32,
            init_value=lambda: weight.unsqueeze(0).repeat(TP_SIZE, 1, 1),
        ),
        TensorSpec(
            "x",
            [TP_SIZE, tokens, HC_MULT, D],
            torch.bfloat16,
            init_value=lambda: x.unsqueeze(0).repeat(TP_SIZE, 1, 1, 1),
        ),
        TensorSpec("out", [TP_SIZE, tokens, HC_MULT, D], torch.bfloat16),
        # Runtime ABI scalar: the persistent signal never resets, so every
        # dispatch passes its own 1-based epoch instead of a fixed threshold.
        ScalarSpec("epoch", torch.int32, 1, compile_runtime=True),
    ]


def build_engram_repeat_specs(batch: int = 2, sequence: int = 4, epochs: int = 2):
    """Build per-epoch stacked inputs for the repeat-epoch validation.

    Each epoch draws a different hash-id set, table, and activations so a
    stale window or an early-passing wait cannot hide behind identical
    inputs; the projection and gate weights are shared like production.
    """
    from golden import ScalarSpec, TensorSpec

    tokens = batch * sequence
    wkv, weight = _engram_weights(seed=3)
    embed_weight_epochs, embed_scale_epochs, x_epochs = [], [], []
    for epoch in range(epochs):
        hash_ids, table, x = _engram_case(batch, sequence, seed=100 + epoch)
        embed_weight, embed_scale = [], []
        for rank in range(TP_SIZE):
            embed_weight_rank, embed_scale_rank = _quantize_rows(table, hash_ids, rank)
            embed_weight.append(embed_weight_rank)
            embed_scale.append(embed_scale_rank)
        embed_weight_epochs.append(torch.stack(embed_weight))
        embed_scale_epochs.append(torch.stack(embed_scale))
        x_epochs.append(x)
    return [
        TensorSpec(
            "embed_weight",
            [epochs, TP_SIZE, tokens, ENGRAM_K],
            torch.float8_e4m3fn,
            init_value=lambda: torch.stack(embed_weight_epochs),
        ),
        TensorSpec(
            "embed_scale",
            [epochs, TP_SIZE, SCALE_COLS, tokens],
            torch.float8_e8m0fnu,
            init_value=lambda: torch.stack(embed_scale_epochs),
        ),
        TensorSpec(
            "wkv_weight",
            [TP_SIZE, ENGRAM_K, KV_OUT],
            torch.bfloat16,
            init_value=lambda: wkv.unsqueeze(0).repeat(TP_SIZE, 1, 1),
        ),
        TensorSpec(
            "weight",
            [TP_SIZE, HC_MULT, D],
            torch.float32,
            init_value=lambda: weight.unsqueeze(0).repeat(TP_SIZE, 1, 1),
        ),
        TensorSpec(
            "x",
            [epochs, TP_SIZE, tokens, HC_MULT, D],
            torch.bfloat16,
            init_value=lambda: torch.stack(x_epochs).unsqueeze(1).repeat(1, TP_SIZE, 1, 1, 1),
        ),
        TensorSpec("out", [epochs, TP_SIZE, tokens, HC_MULT, D], torch.bfloat16),
        ScalarSpec("epochs", torch.int32, epochs),
    ]


def golden_engram(tensors):
    """Fill the expected output from the kernel-wire tensors (harness golden).

    Consumes the same tensors the kernel ABI defines -- the stacked per-rank
    FP8E4M3FN payload embed_weight [TP, T, ENGRAM_K] plus the transposed E8M0
    scale wire embed_scale [TP, SCALE_COLS, T] -- so the reference covers the
    whole kernel: the zero-padded partials dequantize and sum to the
    full-table gather exactly, then project, gate, and add the residual.
    Fills tensors["out"] in place, the golden-harness contract.
    """
    embed_weight = tensors["embed_weight"]
    embed_scale = tensors["embed_scale"].float()  # [TP, SCALE_COLS, T] (transposed wire)
    lookup = torch.zeros(embed_weight.shape[1], ENGRAM_K, dtype=torch.float32)
    for rank in range(TP_SIZE):
        partial = embed_weight[rank].float() * embed_scale[rank].t().repeat_interleave(
            SCALE_GROUP, dim=1
        )
        lookup += partial
    kv = lookup @ tensors["wkv_weight"][0].float()  # [T, KV_OUT]
    key, value = kv.split([HC_MULT * D, D], dim=-1)
    key = key.unflatten(-1, (HC_MULT, D))  # [T, HC_MULT, D]

    h = tensors["x"][0].float()
    w = tensors["weight"][0].float().unsqueeze(0)  # [1, HC_MULT, D]
    rstd = torch.rsqrt(h.square().mean(-1, keepdim=True) + FLASH.rms_norm_eps) * torch.rsqrt(
        key.square().mean(-1, keepdim=True) + FLASH.rms_norm_eps
    )
    dot = (h * w * key).sum(-1, keepdim=True) * rstd * (D**-0.5)  # [T, HC_MULT, 1]
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(CLAMP).sqrt(), dot))
    out = (h + gate * value.unsqueeze(1)).to(torch.bfloat16)
    tensors["out"][:] = out.unsqueeze(0).expand_as(tensors["out"])


def golden_engram_repeat_case(tensors):
    """Fill each epoch's expected output from its own inputs.

    Epochs draw different hash ids and activations, so a stale window read or
    a wait that passed early shows up as a mismatch in that epoch's output.
    """
    for epoch in range(tensors["embed_weight"].shape[0]):
        golden_engram(
            {
                "embed_weight": tensors["embed_weight"][epoch],
                "embed_scale": tensors["embed_scale"][epoch],
                "wkv_weight": tensors["wkv_weight"],
                "weight": tensors["weight"],
                "x": tensors["x"][epoch],
                "out": tensors["out"][epoch],
            }
        )


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
    from pypto.ir import DistributedConfig

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
    if args.batch * args.sequence > PREFILL_MAX_TOKENS:
        raise ValueError(
            f"batch*sequence={args.batch * args.sequence} exceeds the window "
            f"capacity {PREFILL_MAX_TOKENS}"
        )

    result = run(
        fn=engram_group,
        specs=build_engram_tensor_specs(args.batch, args.sequence),
        golden_fn=golden_engram,
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


def validate_repeat(argv=None):
    """Validate repeated Engram epochs over one persistent window on A5."""
    import argparse

    from golden import ratio_allclose, run
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 Engram repeat-epoch validation")
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
    parser.add_argument("--epochs", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    args = parser.parse_args(argv)

    if args.tp != TP_SIZE:
        raise ValueError(f"--tp {args.tp} disagrees with module TP_SIZE {TP_SIZE}")
    device_ids = [int(d) for d in args.device.split(",")]
    if len(device_ids) != TP_SIZE:
        raise ValueError(f"need exactly {TP_SIZE} device ids for tp={TP_SIZE}, got {device_ids}")
    if args.batch * args.sequence > PREFILL_MAX_TOKENS:
        raise ValueError(
            f"batch*sequence={args.batch * args.sequence} exceeds the window "
            f"capacity {PREFILL_MAX_TOKENS}"
        )

    result = run(
        fn=engram_group_repeat,
        specs=build_engram_repeat_specs(args.batch, args.sequence, epochs=args.epochs),
        golden_fn=golden_engram_repeat_case,
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
    "build_engram_repeat_specs",
    "build_engram_tensor_specs",
    "engram",
    "engram_gate",
    "engram_group",
    "engram_group_repeat",
    "engram_rank",
    "golden_engram",
    "golden_engram_repeat_case",
    "validate_repeat",
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

    @pytest.mark.parametrize("tp", [1, 4])
    def test_large_batch_precision(tp, a5_args):
        """T past the old 256-row window cap exercises multi-tile publish/reduce."""
        result = validate(a5_args(tp=tp) + ["--batch", "8", "--sequence", "64"])
        assert result.passed, result.error

    @pytest.mark.parametrize("tp", [4])
    def test_repeat_epoch_precision(tp, a5_args):
        """Two epochs over one persistent window must not read stale signal state."""
        result = validate_repeat(a5_args(tp=tp))
        assert result.passed, result.error

if __name__ == _SCRIPT_ENTRY_POINT:
    main()
