# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: 2-card run; borrows 2 cards via task-submit --device-num
"""DeepSeek-V4 LM head projection with DP-owned hidden and TP vocab shards.

The input hidden states are expected to have already passed the final RMSNorm,
matching cann-recipes' ``DeepseekV3Model.forward`` + ``forward_lm_head`` split.

After attention DP and MoE EP, each rank owns a different set of hidden rows.
For LM head TP, every TP rank owns one contiguous vocabulary shard of
``lm_head_weight``. Each owner rank first publishes its hidden rows to all TP
ranks. Every TP rank computes its local vocabulary shard for every owner, then
routes that shard back to the owner so the owner has full-vocabulary logits for
token selection.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

from config import DECODE_TOKENS, FLASH as M, LM_HEAD_TP_SIZE, PREFILL_TOKENS


T_DYN = pl.dynamic("LM_HEAD_T_DYN")


# Tensor shapes and loop trip counts are static in the frontend, so the TP and
# owner worlds are build-time constants.  Standalone lm_head keeps
# OWNER_SIZE == TP_SIZE; when composed into decode_fwd, OWNER_SIZE comes from
# --ep and may be larger than TP_SIZE.
_TP_CHOICES = (2, 4, 8)
_TP_DEFAULT = 2


def _parse_int_argv(name, default=None):
    for i, tok in enumerate(sys.argv):
        if tok == name and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if tok.startswith(f"{name}="):
            return int(tok.split("=", 1)[1])
    return default


TP_SIZE = _parse_int_argv("--tp", None)
if TP_SIZE is None:
    TP_SIZE = _parse_int_argv("--ep", _TP_DEFAULT)
OWNER_SIZE = _parse_int_argv("--ep", TP_SIZE)

T = DECODE_TOKENS
T_MAX = max(DECODE_TOKENS, PREFILL_TOKENS)
D = M.hidden_size  # 4096 hidden size.
VOCAB = M.vocab_size  # 129280 vocabulary size.

LM_HEAD_K_CHUNK = 128  # K tile width; D / 128 = 32 matmul accumulation blocks.
HIDDEN_COMM_CHUNK = 512  # Hidden publish tile width; D / 512 = 8 copy blocks.
VOCAB_CHUNK = 256  # Main vocab tile width; tail is handled separately.
T_TILE = 8  # Token tile height.
MATMUL_T_TILE = 16
DONE_VALUE = 1
PREFILL_LOGITS_ROWS = 1

assert D % LM_HEAD_K_CHUNK == 0
assert D % HIDDEN_COMM_CHUNK == 0
assert DECODE_TOKENS % T_TILE == 0 and PREFILL_TOKENS % T_TILE == 0
assert DECODE_TOKENS <= MATMUL_T_TILE and PREFILL_TOKENS % MATMUL_T_TILE == 0
assert VOCAB % TP_SIZE == 0
assert TP_SIZE in _TP_CHOICES, f"--tp must be one of {_TP_CHOICES} (got {TP_SIZE})"
assert OWNER_SIZE in _TP_CHOICES, f"--ep must be one of {_TP_CHOICES} (got {OWNER_SIZE})"
assert TP_SIZE <= OWNER_SIZE, f"--tp must be <= --ep/OWNER_SIZE, got tp={TP_SIZE}, owner={OWNER_SIZE}"
assert TP_SIZE <= LM_HEAD_TP_SIZE

K_BLOCKS = D // LM_HEAD_K_CHUNK  # 32.
HIDDEN_COMM_BLOCKS = D // HIDDEN_COMM_CHUNK  # 8.
VOCAB_PER_TP = VOCAB // TP_SIZE
VOCAB_FULL_BLOCKS_PER_TP = VOCAB_PER_TP // VOCAB_CHUNK
VOCAB_TAIL = VOCAB_PER_TP % VOCAB_CHUNK


@pl.jit.inline
def lm_head(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logits_shard: pl.Out[pl.Tensor[[T_MAX, VOCAB_PER_TP], pl.FP32]],
) -> pl.Tensor[[T_MAX, VOCAB_PER_TP], pl.FP32]:
    t_dim = pl.tensor.dim(hidden_states, 0)
    t_matmul = pl.max(t_dim, MATMUL_T_TILE)
    logits_pad = pl.create_tensor([T_MAX, VOCAB_PER_TP], dtype=pl.FP32)

    for t0 in pl.parallel(0, t_matmul, MATMUL_T_TILE):
        hid_rows = pl.min(MATMUL_T_TILE, t_dim - t0)
        for ob in pl.parallel(VOCAB_FULL_BLOCKS_PER_TP):
            o0 = ob * VOCAB_CHUNK
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head"):
                # The peeled (kb==0) matmul tiles must use names distinct from the
                # loop-body tiles below; reusing one name across the peel and the
                # carry loop collides under SSA and corrupts the accumulation.
                hidden0 = pl.slice(hidden_states, [MATMUL_T_TILE, LM_HEAD_K_CHUNK], [t0, 0], valid_shape=[hid_rows, LM_HEAD_K_CHUNK])
                weight0 = lm_head_weight[o0 : o0 + VOCAB_CHUNK, 0:LM_HEAD_K_CHUNK]
                acc = pl.matmul(hidden0, weight0, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.range(1, K_BLOCKS):
                    k0 = kb * LM_HEAD_K_CHUNK
                    hidden_chunk = pl.slice(hidden_states, [MATMUL_T_TILE, LM_HEAD_K_CHUNK], [t0, k0], valid_shape=[hid_rows, LM_HEAD_K_CHUNK])
                    weight_chunk = lm_head_weight[o0 : o0 + VOCAB_CHUNK, k0 : k0 + LM_HEAD_K_CHUNK]
                    acc = pl.matmul_acc(acc, hidden_chunk, weight_chunk, b_trans=True)
                logits_pad[t0 : t0 + MATMUL_T_TILE, o0 : o0 + VOCAB_CHUNK] = acc

        if VOCAB_TAIL != 0:
            tail_o0 = VOCAB_FULL_BLOCKS_PER_TP * VOCAB_CHUNK
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_tail"):
                # Every tile/accumulator here uses a name distinct from the main
                # block above: they live in the same inlined function scope, and a
                # shared name carrying a different shape (e.g. VOCAB_TAIL vs
                # VOCAB_CHUNK) collides under SSA and corrupts the result.
                hidden_t0 = pl.slice(hidden_states, [MATMUL_T_TILE, LM_HEAD_K_CHUNK], [t0, 0], valid_shape=[hid_rows, LM_HEAD_K_CHUNK])
                weight_t0 = lm_head_weight[tail_o0 : tail_o0 + VOCAB_TAIL, 0:LM_HEAD_K_CHUNK]
                acc_tail = pl.matmul(hidden_t0, weight_t0, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.range(1, K_BLOCKS):
                    k0 = kb * LM_HEAD_K_CHUNK
                    hidden_tk = pl.slice(hidden_states, [MATMUL_T_TILE, LM_HEAD_K_CHUNK], [t0, k0], valid_shape=[hid_rows, LM_HEAD_K_CHUNK])
                    weight_tk = lm_head_weight[tail_o0 : tail_o0 + VOCAB_TAIL, k0 : k0 + LM_HEAD_K_CHUNK]
                    acc_tail = pl.matmul_acc(acc_tail, hidden_tk, weight_tk, b_trans=True)
                logits_pad[t0 : t0 + MATMUL_T_TILE, tail_o0 : tail_o0 + VOCAB_TAIL] = acc_tail

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_output"):
        for t0 in pl.range(0, t_dim, T_TILE):
            for ob in pl.pipeline(VOCAB_FULL_BLOCKS_PER_TP, stage=2):
                o0 = ob * VOCAB_CHUNK
                logits_shard[t0:t0 + T_TILE, o0:o0 + VOCAB_CHUNK] = \
                    logits_pad[t0:t0 + T_TILE, o0:o0 + VOCAB_CHUNK]
            if VOCAB_TAIL != 0:
                tail_o0 = VOCAB_FULL_BLOCKS_PER_TP * VOCAB_CHUNK
                logits_shard[t0:t0 + T_TILE, tail_o0:tail_o0 + VOCAB_TAIL] = \
                    logits_pad[t0:t0 + T_TILE, tail_o0:tail_o0 + VOCAB_TAIL]
    return logits_shard


@pl.jit.incore
def publish_hidden_step(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    hidden_window: pld.DistributedTensor[[TP_SIZE * T_MAX, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    t_dim = pl.tensor.dim(hidden_states, 0)
    row_base = my_rank * T_MAX

    for peer in pl.range(TP_SIZE):
        if peer != my_rank:
            # Use the communication-level put primitive for hidden publish.
            # Tile remote_store lowers to a plain remote TSTORE and can expose
            # a short-publish visibility window before the peer sees notify.
            for t0 in pl.range(0, t_dim, T_TILE):
                for kb in pl.range(HIDDEN_COMM_BLOCKS):
                    k0 = kb * HIDDEN_COMM_CHUNK
                    pld.tensor.put(
                        dst=hidden_window,
                        peer=peer,
                        src=hidden_states,
                        dst_offsets=[row_base + t0, k0],
                        src_offsets=[t0, k0],
                        shape=[T_TILE, HIDDEN_COMM_CHUNK],
                    )

    for peer in pl.range(TP_SIZE):
        if peer != my_rank:
            pld.system.notify(
                target=hidden_done,
                peer=peer,
                offsets=[my_rank, 0],
                value=DONE_VALUE,
                op=pld.NotifyOp.Set,
            )

    for src in pl.range(TP_SIZE):
        if src != my_rank:
            pld.system.wait(
                signal=hidden_done,
                offsets=[src, 0],
                expected=DONE_VALUE,
                cmp=pld.WaitCmp.Ge,
            )


@pl.jit.incore
def load_owner_hidden_step(
    owner_hidden: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    hidden_window: pld.DistributedTensor[[TP_SIZE * T_MAX, D], pl.BF16],
    owner_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T_DYN, D], pl.BF16]:
    t_dim = pl.tensor.dim(owner_hidden, 0)
    row_base = owner_rank * T_MAX

    for t0 in pl.range(0, t_dim, T_TILE):
        for kb in pl.range(HIDDEN_COMM_BLOCKS):
            k0 = kb * HIDDEN_COMM_CHUNK
            tile = pl.load(hidden_window, [row_base + t0, k0], [T_TILE, HIDDEN_COMM_CHUNK])
            pl.store(tile, [t0, k0], owner_hidden)
    return owner_hidden


@pl.jit.incore
def route_logits_shard_step(
    logits_shard: pl.Tensor[[T_MAX, VOCAB_PER_TP], pl.FP32],
    logits: pl.Out[pl.Tensor[[T_DYN, VOCAB], pl.FP32]],
    logits_window: pld.DistributedTensor[[T_MAX, VOCAB], pl.FP32],
    owner_rank: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T_DYN, VOCAB], pl.FP32]:
    t_dim = pl.tensor.dim(logits, 0)
    vocab_base = my_rank * VOCAB_PER_TP

    for t0 in pl.range(0, t_dim, T_TILE):
        for ob in pl.range(VOCAB_FULL_BLOCKS_PER_TP):
            o0 = ob * VOCAB_CHUNK
            tile = pl.load(logits_shard, [t0, o0], [T_TILE, VOCAB_CHUNK])
            if owner_rank == my_rank:
                pl.store(tile, [t0, vocab_base + o0], logits)
            else:
                pld.tile.remote_store(
                    tile,
                    target=logits_window,
                    peer=owner_rank,
                    offsets=[t0, vocab_base + o0],
                )

        if VOCAB_TAIL != 0:
            tail_o0 = VOCAB_FULL_BLOCKS_PER_TP * VOCAB_CHUNK
            tile = pl.load(logits_shard, [t0, tail_o0], [T_TILE, VOCAB_TAIL])
            if owner_rank == my_rank:
                pl.store(tile, [t0, vocab_base + tail_o0], logits)
            else:
                pld.tile.remote_store(
                    tile,
                    target=logits_window,
                    peer=owner_rank,
                    offsets=[t0, vocab_base + tail_o0],
                )
    return logits


@pl.jit.incore
def finish_logits_step(
    logits: pl.Out[pl.Tensor[[T_DYN, VOCAB], pl.FP32]],
    logits_window: pld.DistributedTensor[[T_MAX, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T_DYN, VOCAB], pl.FP32]:
    t_dim = pl.tensor.dim(logits, 0)
    for peer in pl.range(TP_SIZE):
        if peer != my_rank:
            pld.system.notify(
                target=logits_done,
                peer=peer,
                offsets=[my_rank, 0],
                value=DONE_VALUE,
                op=pld.NotifyOp.Set,
            )

    for src in pl.range(TP_SIZE):
        if src != my_rank:
            pld.system.wait(
                signal=logits_done,
                offsets=[src, 0],
                expected=DONE_VALUE,
                cmp=pld.WaitCmp.Ge,
            )

    for t0 in pl.range(0, t_dim, T_TILE):
        for src in pl.range(TP_SIZE):
            if src != my_rank:
                src_vocab_base = src * VOCAB_PER_TP
                for ob in pl.range(VOCAB_FULL_BLOCKS_PER_TP):
                    o0 = ob * VOCAB_CHUNK
                    tile = pl.load(
                        logits_window,
                        [t0, src_vocab_base + o0],
                        [T_TILE, VOCAB_CHUNK],
                    )
                    pl.store(tile, [t0, src_vocab_base + o0], logits)

                if VOCAB_TAIL != 0:
                    tail_o0 = VOCAB_FULL_BLOCKS_PER_TP * VOCAB_CHUNK
                    tile = pl.load(
                        logits_window,
                        [t0, src_vocab_base + tail_o0],
                        [T_TILE, VOCAB_TAIL],
                    )
                    pl.store(tile, [t0, src_vocab_base + tail_o0], logits)
    return logits


@pl.jit
def lm_head_tp(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logits: pl.Out[pl.Tensor[[T_DYN, VOCAB], pl.FP32]],
    hidden_window: pld.DistributedTensor[[TP_SIZE * T_MAX, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[T_MAX, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    # scalars trailing — runtime TaskArgs requires all tensor args before any
    # scalar args (#1603-adjacent constraint).
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T_DYN, VOCAB], pl.FP32]:
    t_dim = pl.tensor.dim(hidden_states, 0)
    publish_hidden_step(hidden_states, hidden_window, hidden_done, my_rank)

    for owner_rank in pl.range(TP_SIZE):
        logits_shard = pl.create_tensor([T_MAX, VOCAB_PER_TP], dtype=pl.FP32)
        if owner_rank == my_rank:
            logits_shard = lm_head(hidden_states, lm_head_weight, logits_shard)
            logits = route_logits_shard_step(logits_shard, logits, logits_window, owner_rank, my_rank)
        else:
            owner_hidden = pl.create_tensor([t_dim, D], dtype=pl.BF16)
            owner_hidden = load_owner_hidden_step(owner_hidden, hidden_window, owner_rank)
            logits_shard = lm_head(owner_hidden, lm_head_weight, logits_shard)
            logits = route_logits_shard_step(logits_shard, logits, logits_window, owner_rank, my_rank)
    logits = finish_logits_step(logits, logits_window, logits_done, my_rank)
    return logits


@pl.jit.incore
def publish_hidden_to_tp_decoupled(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    hidden_window: pld.DistributedTensor[[OWNER_SIZE * T_MAX, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[OWNER_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    wait_for_owners: pl.Scalar[pl.INT32],
):
    t_dim = pl.tensor.dim(hidden_states, 0)
    row_base = my_rank * T_MAX

    for peer in pl.range(TP_SIZE):
        if peer != my_rank:
            for t0 in pl.range(0, t_dim, T_TILE):
                for kb in pl.range(HIDDEN_COMM_BLOCKS):
                    k0 = kb * HIDDEN_COMM_CHUNK
                    pld.tensor.put(
                        dst=hidden_window,
                        peer=peer,
                        src=hidden_states,
                        dst_offsets=[row_base + t0, k0],
                        src_offsets=[t0, k0],
                        shape=[T_TILE, HIDDEN_COMM_CHUNK],
                    )

    for peer in pl.range(TP_SIZE):
        if peer != my_rank:
            pld.system.notify(
                target=hidden_done,
                peer=peer,
                offsets=[my_rank, 0],
                value=DONE_VALUE,
                op=pld.NotifyOp.Set,
            )

    if wait_for_owners != 0:
        for src in pl.range(OWNER_SIZE):
            if src != my_rank:
                pld.system.wait(
                    signal=hidden_done,
                    offsets=[src, 0],
                    expected=DONE_VALUE,
                    cmp=pld.WaitCmp.Ge,
                )


@pl.jit.incore
def load_owner_hidden_decoupled(
    owner_hidden: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    hidden_window: pld.DistributedTensor[[OWNER_SIZE * T_MAX, D], pl.BF16],
    owner_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T_DYN, D], pl.BF16]:
    t_dim = pl.tensor.dim(owner_hidden, 0)
    row_base = owner_rank * T_MAX

    for t0 in pl.range(0, t_dim, T_TILE):
        for kb in pl.range(HIDDEN_COMM_BLOCKS):
            k0 = kb * HIDDEN_COMM_CHUNK
            tile = pl.load(hidden_window, [row_base + t0, k0], [T_TILE, HIDDEN_COMM_CHUNK])
            pl.store(tile, [t0, k0], owner_hidden)
    return owner_hidden


@pl.jit.incore
def route_logits_shard_decoupled(
    logits_shard: pl.Tensor[[T_MAX, VOCAB_PER_TP], pl.FP32],
    logits: pl.Out[pl.Tensor[[T_DYN, VOCAB], pl.FP32]],
    logits_window: pld.DistributedTensor[[T_MAX, VOCAB], pl.FP32],
    owner_rank: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T_DYN, VOCAB], pl.FP32]:
    t_dim = pl.tensor.dim(logits, 0)
    vocab_base = tp_rank * VOCAB_PER_TP

    for t0 in pl.range(0, t_dim, T_TILE):
        for ob in pl.range(VOCAB_FULL_BLOCKS_PER_TP):
            o0 = ob * VOCAB_CHUNK
            if owner_rank == tp_rank:
                tile = pl.load(logits_shard, [t0, o0], [T_TILE, VOCAB_CHUNK])
                pl.store(tile, [t0, vocab_base + o0], logits)
            else:
                pld.tensor.put(
                    dst=logits_window,
                    peer=owner_rank,
                    src=logits_shard,
                    dst_offsets=[t0, vocab_base + o0],
                    src_offsets=[t0, o0],
                    shape=[T_TILE, VOCAB_CHUNK],
                )

        if VOCAB_TAIL != 0:
            tail_o0 = VOCAB_FULL_BLOCKS_PER_TP * VOCAB_CHUNK
            if owner_rank == tp_rank:
                tile_tail = pl.load(logits_shard, [t0, tail_o0], [T_TILE, VOCAB_TAIL])
                pl.store(tile_tail, [t0, vocab_base + tail_o0], logits)
            else:
                pld.tensor.put(
                    dst=logits_window,
                    peer=owner_rank,
                    src=logits_shard,
                    dst_offsets=[t0, vocab_base + tail_o0],
                    src_offsets=[t0, tail_o0],
                    shape=[T_TILE, VOCAB_TAIL],
                )
    return logits


@pl.jit.incore
def finish_logits_decoupled(
    logits: pl.Out[pl.Tensor[[T_DYN, VOCAB], pl.FP32]],
    logits_window: pld.DistributedTensor[[T_MAX, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    publish_done: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T_DYN, VOCAB], pl.FP32]:
    t_dim = pl.tensor.dim(logits, 0)
    if publish_done != 0:
        for owner in pl.range(OWNER_SIZE):
            if owner != my_rank:
                pld.system.notify(
                    target=logits_done,
                    peer=owner,
                    offsets=[my_rank, 0],
                    value=DONE_VALUE,
                    op=pld.NotifyOp.Set,
                )

    for src in pl.range(TP_SIZE):
        if src != my_rank:
            pld.system.wait(
                signal=logits_done,
                offsets=[src, 0],
                expected=DONE_VALUE,
                cmp=pld.WaitCmp.Ge,
            )

    for t0 in pl.range(0, t_dim, T_TILE):
        for src in pl.range(TP_SIZE):
            if src != my_rank:
                src_vocab_base = src * VOCAB_PER_TP
                for ob in pl.range(VOCAB_FULL_BLOCKS_PER_TP):
                    o0 = ob * VOCAB_CHUNK
                    tile = pl.load(
                        logits_window,
                        [t0, src_vocab_base + o0],
                        [T_TILE, VOCAB_CHUNK],
                    )
                    pl.store(tile, [t0, src_vocab_base + o0], logits)

                if VOCAB_TAIL != 0:
                    tail_o0 = VOCAB_FULL_BLOCKS_PER_TP * VOCAB_CHUNK
                    tile = pl.load(
                        logits_window,
                        [t0, src_vocab_base + tail_o0],
                        [T_TILE, VOCAB_TAIL],
                    )
                    pl.store(tile, [t0, src_vocab_base + tail_o0], logits)
    return logits


@pl.jit.incore
def select_prefill_last_hidden(
    hidden_states: pl.Tensor[[PREFILL_TOKENS, D], pl.BF16],
    last_hidden: pl.Out[pl.Tensor[[T_TILE, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T_TILE, D], pl.BF16]:
    active_tokens = pl.max(pl.min(num_tokens, PREFILL_TOKENS), 1)
    last_idx = active_tokens - 1
    for kb in pl.range(HIDDEN_COMM_BLOCKS):
        k0 = kb * HIDDEN_COMM_CHUNK
        last_hidden[0:T_TILE, k0:k0 + HIDDEN_COMM_CHUNK] = pl.full(
            [T_TILE, HIDDEN_COMM_CHUNK], dtype=pl.BF16, value=0.0,
        )
        hidden_tile = pl.load(hidden_states, [last_idx, k0], [1, HIDDEN_COMM_CHUNK])
        pl.store(hidden_tile, [0, k0], last_hidden)
    return last_hidden


@pl.jit.incore
def select_prefill_logits_row(
    transport_logits: pl.Tensor[[T_TILE, VOCAB], pl.FP32],
    logits: pl.Out[pl.Tensor[[PREFILL_LOGITS_ROWS, VOCAB], pl.FP32]],
) -> pl.Tensor[[PREFILL_LOGITS_ROWS, VOCAB], pl.FP32]:
    for ob in pl.range(VOCAB // VOCAB_CHUNK):
        o0 = ob * VOCAB_CHUNK
        tile = pl.load(transport_logits, [0, o0], [1, VOCAB_CHUNK])
        pl.store(tile, [0, o0], logits)
    return logits


@pl.jit.inline
def lm_head_tp_decoupled_core(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logits: pl.Out[pl.Tensor[[T_DYN, VOCAB], pl.FP32]],
    hidden_window: pld.DistributedTensor[[OWNER_SIZE * T_MAX, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[OWNER_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[T_MAX, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T_DYN, VOCAB], pl.FP32]:
    t_dim = pl.tensor.dim(hidden_states, 0)
    publish_hidden_to_tp_decoupled(
        hidden_states, hidden_window, hidden_done, my_rank, 1,
    )

    for owner_rank in pl.range(OWNER_SIZE):
        logits_shard = pl.create_tensor([T_MAX, VOCAB_PER_TP], dtype=pl.FP32)
        if owner_rank == my_rank:
            logits_shard = lm_head(hidden_states, lm_head_weight, logits_shard)
            logits = route_logits_shard_decoupled(
                logits_shard, logits, logits_window, owner_rank, my_rank,
            )
        else:
            owner_hidden = pl.create_tensor([t_dim, D], dtype=pl.BF16)
            owner_hidden = load_owner_hidden_decoupled(
                owner_hidden, hidden_window, owner_rank,
            )
            logits_shard = lm_head(owner_hidden, lm_head_weight, logits_shard)
            logits = route_logits_shard_decoupled(
                logits_shard, logits, logits_window, owner_rank, my_rank,
            )

    logits = finish_logits_decoupled(
        logits, logits_window, logits_done, my_rank, 1,
    )
    return logits


@pl.jit.inline
def lm_head_owner_only_decoupled_core(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    logits: pl.Out[pl.Tensor[[T_DYN, VOCAB], pl.FP32]],
    hidden_window: pld.DistributedTensor[[OWNER_SIZE * T_MAX, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[OWNER_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[T_MAX, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T_DYN, VOCAB], pl.FP32]:
    publish_hidden_to_tp_decoupled(
        hidden_states, hidden_window, hidden_done, my_rank, 0,
    )
    logits = finish_logits_decoupled(
        logits, logits_window, logits_done, my_rank, 0,
    )
    return logits


@pl.jit
def lm_head_tp_decoupled_worker(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logits: pl.Out[pl.Tensor[[T_DYN, VOCAB], pl.FP32]],
    hidden_window: pld.DistributedTensor[[OWNER_SIZE * T_MAX, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[OWNER_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[T_MAX, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    # scalars trailing — runtime TaskArgs requires all tensor args before any
    # scalar args (#1603-adjacent constraint).
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T_DYN, VOCAB], pl.FP32]:
    return lm_head_tp_decoupled_core(
        hidden_states, lm_head_weight, logits,
        hidden_window, hidden_done, logits_window, logits_done, my_rank,
    )


@pl.jit
def lm_head_owner_only_decoupled(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    logits: pl.Out[pl.Tensor[[T_DYN, VOCAB], pl.FP32]],
    hidden_window: pld.DistributedTensor[[OWNER_SIZE * T_MAX, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[OWNER_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[T_MAX, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    # scalars trailing — runtime TaskArgs requires all tensor args before any
    # scalar args (#1603-adjacent constraint).
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T_DYN, VOCAB], pl.FP32]:
    return lm_head_owner_only_decoupled_core(
        hidden_states, logits,
        hidden_window, hidden_done, logits_window, logits_done, my_rank,
    )


@pl.jit
def prefill_lm_head_tp_decoupled_worker(
    hidden_states: pl.Tensor[[PREFILL_TOKENS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logits: pl.Out[pl.Tensor[[PREFILL_LOGITS_ROWS, VOCAB], pl.FP32]],
    hidden_window: pld.DistributedTensor[[OWNER_SIZE * T_MAX, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[OWNER_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[T_MAX, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[PREFILL_LOGITS_ROWS, VOCAB], pl.FP32]:
    last_hidden = pl.create_tensor([T_TILE, D], dtype=pl.BF16)
    last_hidden = select_prefill_last_hidden(hidden_states, last_hidden, num_tokens)
    transport_logits = pl.create_tensor([T_TILE, VOCAB], dtype=pl.FP32)
    transport_logits = lm_head_tp_decoupled_core(
        last_hidden, lm_head_weight, transport_logits,
        hidden_window, hidden_done, logits_window, logits_done, my_rank,
    )
    return select_prefill_logits_row(transport_logits, logits)


@pl.jit
def prefill_lm_head_owner_only_decoupled(
    hidden_states: pl.Tensor[[PREFILL_TOKENS, D], pl.BF16],
    logits: pl.Out[pl.Tensor[[PREFILL_LOGITS_ROWS, VOCAB], pl.FP32]],
    hidden_window: pld.DistributedTensor[[OWNER_SIZE * T_MAX, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[OWNER_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[T_MAX, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[PREFILL_LOGITS_ROWS, VOCAB], pl.FP32]:
    last_hidden = pl.create_tensor([T_TILE, D], dtype=pl.BF16)
    last_hidden = select_prefill_last_hidden(hidden_states, last_hidden, num_tokens)
    transport_logits = pl.create_tensor([T_TILE, VOCAB], dtype=pl.FP32)
    transport_logits = lm_head_owner_only_decoupled_core(
        last_hidden, transport_logits,
        hidden_window, hidden_done, logits_window, logits_done, my_rank,
    )
    return select_prefill_logits_row(transport_logits, logits)


@pl.jit.host
def l3_lm_head(
    hidden_states: pl.Tensor[[TP_SIZE, T, D], pl.BF16],
    lm_head_weight: pl.Tensor[[TP_SIZE, VOCAB_PER_TP, D], pl.BF16],
    logits: pl.Out[pl.Tensor[[TP_SIZE, T, VOCAB], pl.FP32]],
):
    hidden_window_buf = pld.alloc_window_buffer([TP_SIZE * T_MAX, D], dtype=pl.BF16)
    hidden_done_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    logits_window_buf = pld.alloc_window_buffer([T_MAX, VOCAB], dtype=pl.FP32)
    logits_done_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)

    for r in pl.range(pld.world_size()):
        hidden_window = pld.window(hidden_window_buf, [TP_SIZE * T_MAX, D], dtype=pl.BF16)
        hidden_done = pld.window(hidden_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        logits_window = pld.window(logits_window_buf, [T_MAX, VOCAB], dtype=pl.FP32)
        logits_done = pld.window(logits_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        lm_head_tp(
            hidden_states[r],
            lm_head_weight[r],
            logits[r],
            hidden_window,
            hidden_done,
            logits_window,
            logits_done,
            r,
            device=r,
        )


@pl.jit.host
def l3_prefill_lm_head(
    hidden_states: pl.Tensor[[OWNER_SIZE, PREFILL_TOKENS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[TP_SIZE, VOCAB_PER_TP, D], pl.BF16],
    logits: pl.Out[pl.Tensor[[OWNER_SIZE, PREFILL_LOGITS_ROWS, VOCAB], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    hidden_window_buf = pld.alloc_window_buffer(OWNER_SIZE * T_MAX * D * 2)
    hidden_done_buf = pld.alloc_window_buffer(OWNER_SIZE * 4)
    logits_window_buf = pld.alloc_window_buffer(T_MAX * VOCAB * 4)
    logits_done_buf = pld.alloc_window_buffer(TP_SIZE * 4)

    for r in pl.range(TP_SIZE):
        hidden_window = pld.window(hidden_window_buf, [OWNER_SIZE * T_MAX, D], dtype=pl.BF16)
        hidden_done = pld.window(hidden_done_buf, [OWNER_SIZE, 1], dtype=pl.INT32)
        logits_window = pld.window(
            logits_window_buf, [T_MAX, VOCAB], dtype=pl.FP32,
        )
        logits_done = pld.window(logits_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        prefill_lm_head_tp_decoupled_worker(
            hidden_states[r],
            lm_head_weight[r],
            logits[r],
            hidden_window,
            hidden_done,
            logits_window,
            logits_done,
            num_tokens,
            r,
            device=r,
        )

    for r in pl.range(TP_SIZE, pld.world_size()):
        hidden_window = pld.window(hidden_window_buf, [OWNER_SIZE * T_MAX, D], dtype=pl.BF16)
        hidden_done = pld.window(hidden_done_buf, [OWNER_SIZE, 1], dtype=pl.INT32)
        logits_window = pld.window(
            logits_window_buf, [T_MAX, VOCAB], dtype=pl.FP32,
        )
        logits_done = pld.window(logits_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        prefill_lm_head_owner_only_decoupled(
            hidden_states[r],
            logits[r],
            hidden_window,
            hidden_done,
            logits_window,
            logits_done,
            num_tokens,
            r,
            device=r,
        )


def golden_lm_head(tensors):
    import torch

    hidden = tensors["hidden_states"].float()
    weight = tensors["lm_head_weight"].float()
    full_logits = []
    for owner_rank in range(hidden.shape[0]):
        shard_logits = []
        for tp_rank in range(weight.shape[0]):
            shard_weight = weight[tp_rank]
            shard_logits.append(torch.matmul(hidden[owner_rank], shard_weight.t()))
        full_logits.append(torch.cat(shard_logits, dim=-1))
    tensors["logits"][:] = torch.stack(full_logits, dim=0)


def golden_prefill_lm_head(tensors):
    import torch

    hidden = tensors["hidden_states"].float()
    weight = tensors["lm_head_weight"].float()
    last_idx = max(min(int(tensors["num_tokens"]), PREFILL_TOKENS), 1) - 1
    full_logits = []
    for owner_rank in range(hidden.shape[0]):
        last_hidden = hidden[owner_rank, last_idx:last_idx + 1]
        shard_logits = [torch.matmul(last_hidden, shard.float().t()) for shard in weight]
        full_logits.append(torch.cat(shard_logits, dim=-1))
    tensors["logits"][:] = torch.stack(full_logits, dim=0)


def compare_prefill_logits(actual, expected, **_):
    import torch

    close = torch.isclose(actual, expected, rtol=1e-3, atol=1e-3)
    if bool(close.all()):
        return True, ""
    lines = []
    for owner in range(actual.shape[0]):
        for shard in range(TP_SIZE):
            start = shard * VOCAB_PER_TP
            end = start + VOCAB_PER_TP
            shard_actual = actual[owner, 0, start:end]
            shard_close = close[owner, 0, start:end]
            lines.append(
                f"    owner={owner} shard={shard}: "
                f"bad={int((~shard_close).sum())}/{VOCAB_PER_TP} "
                f"zeros={int((shard_actual == 0).sum())}"
            )
    return False, "\n".join(lines)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_hidden_states():
        # Each source rank owns different post-DP/post-EP token rows.
        return torch.randn(TP_SIZE, T, D) * 0.1

    def init_lm_head_weight():
        return (torch.randn(TP_SIZE, VOCAB_PER_TP, D) / D ** 0.5).to(torch.bfloat16)

    return [
        TensorSpec("hidden_states", [TP_SIZE, T, D], torch.bfloat16, init_value=init_hidden_states),
        # Static LM-head weight, sharded per TP rank (leading dim TP_SIZE). Kept
        # device-resident (child_memory): each shard uploaded to its card once and
        # reused across dispatches, skipping the per-dispatch H2D/D2H. Resident is
        # input-only, which this weight is (is_output=False).
        TensorSpec(
            "lm_head_weight",
            [TP_SIZE, VOCAB_PER_TP, D],
            torch.bfloat16,
            init_value=init_lm_head_weight,
            resident="stacked",
        ),
        TensorSpec("logits", [TP_SIZE, T, VOCAB], torch.float32, is_output=True),
    ]


def build_prefill_tensor_specs(num_tokens=PREFILL_TOKENS):
    import torch
    from golden import ScalarSpec, TensorSpec

    def init_hidden_states():
        return (torch.randn(OWNER_SIZE, PREFILL_TOKENS, D) * 0.1).to(torch.bfloat16)

    def init_lm_head_weight():
        return (torch.randn(TP_SIZE, VOCAB_PER_TP, D) / D ** 0.5).to(torch.bfloat16)

    return [
        TensorSpec(
            "hidden_states",
            [OWNER_SIZE, PREFILL_TOKENS, D],
            torch.bfloat16,
            init_value=init_hidden_states,
        ),
        TensorSpec(
            "lm_head_weight",
            [TP_SIZE, VOCAB_PER_TP, D],
            torch.bfloat16,
            init_value=init_lm_head_weight,
            resident="stacked",
        ),
        TensorSpec(
            "logits",
            [OWNER_SIZE, PREFILL_LOGITS_ROWS, VOCAB],
            torch.float32,
            is_output=True,
        ),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=list(_TP_CHOICES),
                        help="LM-head tensor-parallel world size")
    parser.add_argument("--ep", type=int, default=OWNER_SIZE, choices=list(_TP_CHOICES),
                        help="Hidden-owner world size; must be >= --tp")
    parser.add_argument("--prefill-last-token", action="store_true", default=False,
                        help="Project only the last active packed-prefill token")
    parser.add_argument("--num-tokens", type=int, default=PREFILL_TOKENS,
                        help="Active packed-prefill rows used by --prefill-last-token")
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(TP_SIZE)),
                        help=f"comma-separated device ids; need at least {TP_SIZE}")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    required_devices = OWNER_SIZE if args.prefill_last_token else TP_SIZE
    assert len(device_ids) >= required_devices, (
        f"need at least {required_devices} devices, got {device_ids}"
    )
    assert args.tp == TP_SIZE and args.ep == OWNER_SIZE
    assert TP_SIZE <= OWNER_SIZE
    assert 1 <= args.num_tokens <= PREFILL_TOKENS

    fn = l3_prefill_lm_head if args.prefill_last_token else l3_lm_head
    specs = (
        build_prefill_tensor_specs(args.num_tokens)
        if args.prefill_last_token else build_tensor_specs()
    )
    golden_fn = golden_prefill_lm_head if args.prefill_last_token else golden_lm_head
    compare_fn = {"logits": compare_prefill_logits} if args.prefill_last_token else None

    result = run_jit(
        fn=fn,
        specs=specs,
        golden_fn=golden_fn,
        compare_fn=compare_fn,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:required_devices],
                num_sub_workers=0,
            ),
        ),
        runtime_cfg=dict(
            platform=args.platform,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
