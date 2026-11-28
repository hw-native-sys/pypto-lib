# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Vocabulary-sharded scorer for the native five-step Markov chain.

The base-logit producer can retain its existing participants. This scorer takes
that producer's local vocabulary slice and publishes one score/token pair per
rank and step. The device pipeline must fund the new windows and epochs before
using it.
"""

import pypto.language as pl
import pypto.language.distributed as pld

from .entry import CONFIDENCE_PAD, D, DSPARK_MARKOV_RANK, GREEDY_ARGMAX_ROWS, HIDDEN_TILE, MARKOV_M_TILE, NEG_INF

RANKS = 8
STEPS = 5
VOCAB_CHUNK = 256
CHUNKS_PER_SHARD = 64
SHARD_ROWS = CHUNKS_PER_SHARD * VOCAB_CHUNK
PAIR_ROW = 16


@pl.jit.inline
def score_shard_chunks(
    embedding: pl.Tensor[[1, DSPARK_MARKOV_RANK], pl.BF16],
    base_logits: pl.Tensor[[STEPS, SHARD_ROWS], pl.FP32],
    w2: pl.Tensor[[SHARD_ROWS, DSPARK_MARKOV_RANK], pl.BF16],
    winners: pl.Tensor[[CHUNKS_PER_SHARD, PAIR_ROW], pl.FP32],
    active_chunks: pl.Scalar[pl.INT32],
    vocab_start: pl.Scalar[pl.INT32],
    step: pl.Scalar[pl.INT32],
    base_ready: pl.Scalar[pl.TASK_ID],
):
    """Replace a full vocabulary scan with 63 or 64 exact chunk maxima."""
    with pl.spmd(
        24, name_hint="markov_shard_bias_argmax", deps=[base_ready],
        optimizations=[pl.cross_core_slot(slot_num=2)],
    ) as scored:
        pl.set_cache_policy(w2, pl.CachePolicy.BYPASS)
        part = pl.tile.get_block_idx()
        embed = pl.slice(embedding, [MARKOV_M_TILE, DSPARK_MARKOV_RANK], [0, 0],
                         valid_shape=[1, DSPARK_MARKOV_RANK])
        for chunk in pl.range(part, active_chunks, 24):
            start = chunk * VOCAB_CHUNK
            weight = pl.slice(w2, [VOCAB_CHUNK, DSPARK_MARKOV_RANK], [start, 0])
            bias = pl.matmul(embed, weight, b_trans=True, out_dtype=pl.FP32)
            for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                bias_tile = pl.aiv_shard(bias)
                if lane == 0:
                    bias_row = pl.slice(bias_tile, [1, VOCAB_CHUNK], [0, 0], valid_shape=[1, VOCAB_CHUNK])
                    base_row = pl.slice(base_logits, [1, VOCAB_CHUNK], [step, start])
                    values = pl.full([GREEDY_ARGMAX_ROWS, VOCAB_CHUNK], dtype=pl.FP32, value=NEG_INF)
                    values[0:1, :] = pl.add(base_row, bias_row)
                    best = pl.cast(pl.read(pl.row_argmax(values), [0, 0]), pl.INDEX)
                    value = pl.read(values, [0, best])
                    pl.write(winners, [chunk, 0], value)
                    token = vocab_start + pl.cast(start + best, pl.INT32)
                    pl.write(winners, [chunk, 1], pl.cast(token, pl.FP32))

    return scored


@pl.jit.inline
def score_shard(
    embedding: pl.Tensor[[1, DSPARK_MARKOV_RANK], pl.BF16],
    base_logits: pl.Tensor[[STEPS, SHARD_ROWS], pl.FP32],
    w2: pl.Tensor[[SHARD_ROWS, DSPARK_MARKOV_RANK], pl.BF16],
    rank_pair: pl.Tensor[[1, PAIR_ROW], pl.FP32],
    active_chunks: pl.Scalar[pl.INT32],
    vocab_start: pl.Scalar[pl.INT32],
    step: pl.Scalar[pl.INT32],
    base_ready: pl.Scalar[pl.TASK_ID],
):
    """Replace a full vocabulary scan with 63 or 64 exact chunk maxima."""
    winners = pl.create_tensor([CHUNKS_PER_SHARD, PAIR_ROW], dtype=pl.FP32)
    scored = score_shard_chunks(embedding, base_logits, w2, winners, active_chunks,
                                vocab_start, step, base_ready)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="markov_shard_fold", deps=[scored]) as local_ready:
        best_score = pl.read(winners, [0, 0])
        best_id = pl.read(winners, [0, 1])
        for chunk in pl.range(1, active_chunks):
            score = pl.read(winners, [chunk, 0])
            if score > best_score:
                best_score = score
                best_id = pl.read(winners, [chunk, 1])
        row = pl.create_tensor([1, PAIR_ROW], dtype=pl.FP32)
        row[:, :] = pl.full([1, PAIR_ROW], dtype=pl.FP32, value=0.0)
        pl.write(row, [0, 0], best_score)
        # Vocabulary IDs are below 2^24 and exactly representable as FP32.
        pl.write(row, [0, 1], best_id)
        rank_pair[:, :] = row
    return local_ready


@pl.jit.inline
def publish_shard_winner(
    rank_pair: pl.Tensor[[1, PAIR_ROW], pl.FP32],
    top_ids: pl.Tensor[[1, STEPS], pl.INT32],
    candidates: pld.DistributedTensor[[STEPS * RANKS, PAIR_ROW], pl.FP32],
    arrivals: pld.DistributedTensor[[STEPS * RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    step: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    """Make each step's winner visible to all scorers and to the confidence path."""
    step_base = step * RANKS
    owned_slot = step_base + my_rank
    with pl.spmd(RANKS, name_hint="markov_publish_local_winner", deps=[ready]) as sent:
        peer = pl.tile.get_block_idx()
        pld.tensor.put(
            dst=candidates, peer=peer, src=rank_pair, dst_offsets=[owned_slot, 0],
            src_offsets=[0, 0], shape=[1, PAIR_ROW],
        )
        pld.system.notify(target=arrivals, peer=peer, offsets=[owned_slot, 0],
                          value=1, op=pld.NotifyOp.AtomicAdd)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="markov_merge_rank_winners", deps=[sent]) as done:
        for rank in pl.range(RANKS):
            pld.system.wait(signal=arrivals, offsets=[step_base + rank, 0], expected=epoch, cmp=pld.WaitCmp.Ge)
        best_score = pl.read(candidates, [step_base, 0])
        best_id = pl.read(candidates, [step_base, 1])
        # Strict comparison and ascending rank order preserve the lowest-ID tie break.
        for rank in pl.range(1, RANKS):
            score = pl.read(candidates, [step_base + rank, 0])
            if score > best_score:
                best_score = score
                best_id = pl.read(candidates, [step_base + rank, 1])
        pl.write(top_ids, [0, step], pl.cast(best_id, pl.INT32))
    return done


@pl.jit.inline
def run_sharded_steps(
    first_token: pl.Scalar[pl.INT64],
    w1: pl.Tensor[[129280, DSPARK_MARKOV_RANK], pl.BF16],
    w2: pl.Tensor[[SHARD_ROWS, DSPARK_MARKOV_RANK], pl.BF16],
    base_logits: pl.Tensor[[STEPS, SHARD_ROWS], pl.FP32],
    ids: pl.Tensor[[1, STEPS], pl.INT32],
    candidates: pld.DistributedTensor[[STEPS * RANKS, PAIR_ROW], pl.FP32],
    arrivals: pld.DistributedTensor[[STEPS * RANKS, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    """Run the five dependent steps on all ranks, with one winner fold per step."""
    active = pl.const(63, pl.INT32)
    start = pl.cast((rank * 63 + 1) * VOCAB_CHUNK, pl.INT32)
    if rank == 0:
        active = pl.const(64, pl.INT32)
        start = pl.const(0, pl.INT32)
    step_embedding = pl.create_tensor([1, DSPARK_MARKOV_RANK], dtype=pl.BF16)
    local_pair = pl.create_tensor([1, PAIR_ROW], dtype=pl.FP32)
    previous_step = ready
    for step in pl.range(STEPS):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="markov_winner_embedding", deps=[previous_step]) as embedded:
            token = pl.cast(first_token, pl.INDEX)
            if step > 0:
                token = pl.cast(pl.read(ids, [0, step - 1]), pl.INDEX)
            step_embedding[:, :] = w1[token : token + 1, :]
        scored = score_shard(
            step_embedding, base_logits, w2, local_pair, active, start, pl.cast(step, pl.INT32), embedded,
        )
        previous_step = publish_shard_winner(local_pair, ids, candidates, arrivals, rank, epoch,
                                             pl.cast(step, pl.INT32), scored)
    return previous_step


@pl.jit.inline
def distribute_base_logits(
    logits: pl.Tensor[[32, 129280], pl.FP32],
    shard_window: pld.DistributedTensor[[2 * STEPS, SHARD_ROWS], pl.FP32],
    shard_epoch: pld.DistributedTensor[[1, 1], pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    logits_ready: pl.Scalar[pl.TASK_ID],
):
    """The TP4 owner publishes each rank's nonoverlapping B1 logits slice."""
    lane = pl.cast((epoch - 1) % 2, pl.INDEX) * STEPS
    with pl.spmd(RANKS, name_hint="markov_publish_base_slices", deps=[logits_ready]) as ready:
        peer = pl.tile.get_block_idx()
        if peer == 0:
            pld.tensor.put(
                dst=shard_window, peer=peer, src=logits,
                dst_offsets=[lane, 0], src_offsets=[0, 0], shape=[STEPS, SHARD_ROWS],
                chunk_rows=1, chunk_cols=2048, pipeline=True,
            )
        else:
            col = (peer * 63 + 1) * VOCAB_CHUNK
            pld.tensor.put(
                dst=shard_window, peer=peer, src=logits,
                dst_offsets=[lane, 0], src_offsets=[0, col],
                shape=[STEPS, 63 * VOCAB_CHUNK],
                chunk_rows=1, chunk_cols=2048, pipeline=True,
            )
        pld.system.notify(target=shard_epoch, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)
    return ready


@pl.jit.inline
def receive_base_logits(
    shard_window: pld.DistributedTensor[[2 * STEPS, SHARD_ROWS], pl.FP32],
    shard_epoch: pld.DistributedTensor[[1, 1], pl.INT32],
    local_logits: pl.Tensor[[STEPS, SHARD_ROWS], pl.FP32],
    epoch: pl.Scalar[pl.INT32],
    entry_ready: pl.Scalar[pl.TASK_ID],
):
    """Receive a slice before local scores, using only bounded Vec tiles."""
    lane = pl.cast((epoch - 1) % 2, pl.INDEX) * STEPS
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="markov_wait_base_slice", deps=[entry_ready]) as waited:
        pld.system.wait(signal=shard_epoch, offsets=[0, 0], expected=epoch, cmp=pld.WaitCmp.Ge)
    with pl.spmd(SHARD_ROWS // 2048, name_hint="markov_load_base_slice", deps=[waited]) as loaded:
        col = pl.tile.get_block_idx() * 2048
        block = pl.load(shard_window, [lane, col], [STEPS, 2048])
        pl.store(block, [0, col], local_logits)
    return loaded


@pl.jit.inline
def m8_publish_inputs(
    logits: pl.Tensor[[32, 129280], pl.FP32],
    last: pl.Tensor[[1], pl.INT64],
    sampled: pl.Tensor[[1], pl.INT32],
    initial: pl.Tensor[[1], pl.INT64],
    phase_epoch: pl.Tensor[[1], pl.INT32],
    shard: pld.DistributedTensor[[2 * STEPS, SHARD_ROWS], pl.FP32],
    seed_window: pld.DistributedTensor[[2, 8], pl.INT64],
    ready_epoch: pld.DistributedTensor[[2, 32], pl.INT32],
    slot: pl.Scalar[pl.INDEX],
    logits_ready: pl.Scalar[pl.TASK_ID],
):
    """Publish FP32 base logits and the actual rank-0 initial token."""
    seed = pl.create_tensor([1, 8], dtype=pl.INT64)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="markov_native_seed", deps=[logits_ready]) as seed_ready:
        token = pl.read(initial, [0])
        if pl.read(sampled, [0]) > 0:
            token = pl.read(last, [0])
        for column in pl.range(8):
            pl.write(seed, [0, column], pl.const(0, pl.INT64))
        pl.write(seed, [0, 0], token)
        pl.system.cacheinvalid()
        pl.system.fence()
    with pl.spmd(RANKS, name_hint="markov_native_publish_inputs", deps=[seed_ready]) as sent:
        peer = pl.tile.get_block_idx()
        if peer == 0:
            pld.tensor.put(dst=shard, peer=peer, src=logits, dst_offsets=[slot * STEPS, 0],
                           src_offsets=[0, 0], shape=[STEPS, SHARD_ROWS],
                           chunk_rows=1, chunk_cols=2048, pipeline=True)
        else:
            off = (peer * 63 + 1) * VOCAB_CHUNK
            pld.tensor.put(dst=shard, peer=peer, src=logits, dst_offsets=[slot * STEPS, 0],
                           src_offsets=[0, off], shape=[STEPS, 63 * VOCAB_CHUNK],
                           chunk_rows=1, chunk_cols=2048, pipeline=True)
        pld.tensor.put(dst=seed_window, peer=peer, src=seed, dst_offsets=[slot, 0],
                       src_offsets=[0, 0], shape=[1, 8])
        pld.system.notify(target=ready_epoch, peer=peer, offsets=[slot, 0],
                          value=pl.read(phase_epoch, [0]), op=pld.NotifyOp.Set)
    return sent


@pl.jit.inline
def m8_publish_winner(
    pair: pl.Tensor[[1, PAIR_ROW], pl.FP32],
    ids: pl.Tensor[[1, STEPS], pl.INT32],
    candidates: pld.DistributedTensor[[2 * STEPS * RANKS, PAIR_ROW], pl.FP32],
    arrivals: pld.DistributedTensor[[2 * STEPS * RANKS, 32], pl.INT32],
    phase_epoch: pl.Tensor[[1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    slot: pl.Scalar[pl.INDEX],
    step: pl.Scalar[pl.INDEX],
    ready: pl.Scalar[pl.TASK_ID],
):
    base = (slot * STEPS + step) * RANKS
    owned = base + rank
    with pl.spmd(RANKS, name_hint="markov_native_publish_rank_pair", deps=[ready]) as published:
        peer = pl.tile.get_block_idx()
        pld.tensor.put(dst=candidates, peer=peer, src=pair, dst_offsets=[owned, 0],
                       src_offsets=[0, 0], shape=[1, PAIR_ROW])
        pld.system.notify(target=arrivals, peer=peer, offsets=[owned, 0],
                          value=pl.read(phase_epoch, [0]), op=pld.NotifyOp.Set)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="markov_native_wait_rank_pairs", deps=[published]) as waited:
        epoch = pl.read(phase_epoch, [0])
        for peer in pl.range(RANKS):
            pld.system.defer_wait(signal=arrivals, offsets=[base + peer, 0], expected=epoch, cmp=pld.WaitCmp.Ge)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="markov_native_fold_rank_pairs", deps=[waited]) as done:
        pl.system.cacheinvalid()
        best_score = pl.read(candidates, [base, 0])
        best_id = pl.read(candidates, [base, 1])
        for peer in pl.range(1, RANKS):
            score = pl.read(candidates, [base + peer, 0])
            if score > best_score:
                best_score = score
                best_id = pl.read(candidates, [base + peer, 1])
        pl.write(ids, [0, step], pl.cast(best_id, pl.INT32))
    return done


@pl.jit.inline
def m8_fold_and_publish_chunks(
    winners: pl.Tensor[[CHUNKS_PER_SHARD, PAIR_ROW], pl.FP32],
    active_chunks: pl.Scalar[pl.INT32],
    candidates: pld.DistributedTensor[[2 * STEPS * RANKS, PAIR_ROW], pl.FP32],
    arrivals: pld.DistributedTensor[[2 * STEPS * RANKS, 32], pl.INT32],
    phase_epoch: pl.Tensor[[1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    slot: pl.Scalar[pl.INDEX],
    step: pl.Scalar[pl.INDEX],
    scored: pl.Scalar[pl.TASK_ID],
):
    """Each peer block folds the ordered chunk winners and sends its own row."""
    owned = (slot * STEPS + step) * RANKS + rank
    with pl.spmd(RANKS, name_hint="markov_native_fold_and_publish_peer", deps=[scored]) as sent:
        peer = pl.tile.get_block_idx()
        best_score = pl.read(winners, [0, 0])
        best_id = pl.read(winners, [0, 1])
        for chunk in pl.range(1, active_chunks):
            score = pl.read(winners, [chunk, 0])
            if score > best_score:
                best_score = score
                best_id = pl.read(winners, [chunk, 1])
        row = pl.tile.full([1, PAIR_ROW], dtype=pl.FP32, value=0.0)
        pl.tile.write(row, [0, 0], best_score)
        pl.tile.write(row, [0, 1], best_id)
        epoch = pl.read(phase_epoch, [0])
        pld.tile.remote_store(row, candidates, peer, [owned, 0])
        pld.system.notify(target=arrivals, peer=peer, offsets=[owned, 0],
                          value=epoch, op=pld.NotifyOp.Set)
    return sent


@pl.jit.inline
def m8_wait_fold_and_embed(
    candidates: pld.DistributedTensor[[2 * STEPS * RANKS, PAIR_ROW], pl.FP32],
    arrivals: pld.DistributedTensor[[2 * STEPS * RANKS, 32], pl.INT32],
    ids: pl.Tensor[[1, STEPS], pl.INT32],
    w1: pl.Tensor[[129280, DSPARK_MARKOV_RANK], pl.BF16],
    embedding_rows: pl.Tensor[[STEPS, DSPARK_MARKOV_RANK], pl.BF16],
    phase_epoch: pl.Tensor[[1], pl.INT32],
    slot: pl.Scalar[pl.INDEX],
    step: pl.Scalar[pl.INDEX],
    sent: pl.Scalar[pl.TASK_ID],
):
    """Observe arrivals on one AIV, retain rank-first ties and load the next embedding."""
    base = (slot * STEPS + step) * RANKS
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="markov_native_wait_fold_and_embed",
               deps=[sent], allow_early_resolve=True) as done:
        epoch = pl.read(phase_epoch, [0])
        for peer in pl.range(RANKS):
            pld.system.wait(signal=arrivals, offsets=[base + peer, 0], expected=epoch, cmp=pld.WaitCmp.Ge)
        pl.system.cacheinvalid()
        best_score = pl.read(candidates, [base, 0])
        best_id = pl.read(candidates, [base, 1])
        for peer in pl.range(1, RANKS):
            score = pl.read(candidates, [base + peer, 0])
            if score > best_score:
                best_score = score
                best_id = pl.read(candidates, [base + peer, 1])
        token = pl.cast(best_id, pl.INT32)
        pl.write(ids, [0, step], token)
        if step + 1 < STEPS:
            location = pl.cast(token, pl.INDEX)
            embedding_rows[step + 1:step + 2, :] = w1[location:location + 1, :]
    return done


@pl.jit.inline
def m8_verify_steps(
    w1: pl.Tensor[[129280, DSPARK_MARKOV_RANK], pl.BF16],
    w2: pl.Tensor[[SHARD_ROWS, DSPARK_MARKOV_RANK], pl.BF16],
    ids: pl.Tensor[[1, STEPS], pl.INT32],
    phase_epoch: pl.Tensor[[1], pl.INT32],
    shard: pld.DistributedTensor[[2 * STEPS, SHARD_ROWS], pl.FP32],
    seed_window: pld.DistributedTensor[[2, 8], pl.INT64],
    ready_epoch: pld.DistributedTensor[[2, 32], pl.INT32],
    candidates: pld.DistributedTensor[[2 * STEPS * RANKS, PAIR_ROW], pl.FP32],
    arrivals: pld.DistributedTensor[[2 * STEPS * RANKS, 32], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    slot: pl.Scalar[pl.INDEX],
    metadata_ready: pl.Scalar[pl.TASK_ID],
):
    local = pl.create_tensor([STEPS, SHARD_ROWS], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="markov_native_wait_logits", deps=[metadata_ready]) as waited:
        pld.system.defer_wait(signal=ready_epoch, offsets=[slot, 0], expected=pl.read(phase_epoch, [0]),
                              cmp=pld.WaitCmp.Ge)
    with pl.spmd(SHARD_ROWS // 2048, name_hint="markov_native_load_logits", deps=[waited]) as loaded:
        col = pl.tile.get_block_idx() * 2048
        tile = pl.load(shard, [slot * STEPS, col], [STEPS, 2048])
        pl.store(tile, [0, col], local)

    active = pl.const(63, pl.INT32)
    start = pl.cast((rank * 63 + 1) * VOCAB_CHUNK, pl.INT32)
    if rank == 0:
        active = pl.const(64, pl.INT32)
        start = pl.const(0, pl.INT32)
    previous = loaded
    for step in pl.range(STEPS):
        embed = pl.create_tensor([1, DSPARK_MARKOV_RANK], dtype=pl.BF16)
        pair = pl.create_tensor([1, PAIR_ROW], dtype=pl.FP32)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="markov_native_embed_step", deps=[previous]) as embedded:
            token = pl.cast(0, pl.INDEX)
            if step == 0:
                pl.system.cacheinvalid()
                token = pl.cast(pl.read(seed_window, [slot, 0]), pl.INDEX)
            else:
                token = pl.cast(pl.read(ids, [0, step - 1]), pl.INDEX)
            embed[:, :] = w1[token : token + 1, :]
        scored = score_shard(embed, local, w2, pair, active, start, pl.cast(step, pl.INT32), embedded)
        previous = m8_publish_winner(pair, ids, candidates, arrivals, phase_epoch, rank, slot, step, scored)
    return previous


@pl.jit.inline
def m8_confidence_outputs(
    head: pl.Tensor[[1, STEPS, 4096], pl.BF16],
    w1: pl.Tensor[[129280, DSPARK_MARKOV_RANK], pl.BF16],
    confidence_weight: pl.Tensor[[1, 4352], pl.FP32],
    all_ids: pl.Tensor[[1, STEPS], pl.INT32],
    root_ids: pl.Tensor[[1, STEPS], pl.INT32],
    root_probs: pl.Tensor[[1, STEPS], pl.FP32],
    seed_window: pld.DistributedTensor[[2, 8], pl.INT64],
    slot: pl.Scalar[pl.INDEX],
    rank: pl.Scalar[pl.INT32],
    all_ready: pl.Scalar[pl.TASK_ID],
):
    """Keep the original Markov confidence math on the bridge owner only."""
    scratch = pl.create_tensor([STEPS, PAIR_ROW], dtype=pl.FP32)
    with pl.spmd(STEPS, name_hint="markov_native_confidence", deps=[all_ready]) as confidence_ready:
        step = pl.tile.get_block_idx()
        if rank == 0:
            token = pl.cast(0, pl.INDEX)
            if step == 0:
                pl.system.cacheinvalid()
                token = pl.cast(pl.read(seed_window, [slot, 0]), pl.INDEX)
            else:
                token = pl.cast(pl.read(all_ids, [0, step - 1]), pl.INDEX)
            confidence_logit = pl.full([1, CONFIDENCE_PAD], dtype=pl.FP32, value=0.0)
            for block in pl.range(D // HIDDEN_TILE):
                offset = block * HIDDEN_TILE
                hidden = pl.reshape(
                    pl.slice(head, [CONFIDENCE_PAD, 1, HIDDEN_TILE], [0, step, offset],
                             valid_shape=[1, 1, HIDDEN_TILE]),
                    [CONFIDENCE_PAD, HIDDEN_TILE],
                )
                w = confidence_weight[0:1, offset:offset + HIDDEN_TILE]
                scaled = pl.col_expand_mul(pl.cast(hidden, pl.FP32), w)
                confidence_logit = pl.add(confidence_logit,
                                         pl.reshape(pl.row_sum(scaled), [1, CONFIDENCE_PAD]))
            emb = pl.slice(w1, [CONFIDENCE_PAD, DSPARK_MARKOV_RANK], [token, 0],
                           valid_shape=[1, DSPARK_MARKOV_RANK])
            extra = confidence_weight[0:1, D:D + DSPARK_MARKOV_RANK]
            emb_scaled = pl.col_expand_mul(pl.cast(emb, pl.FP32), extra)
            confidence_logit = pl.add(confidence_logit, pl.reshape(pl.row_sum(emb_scaled), [1, CONFIDENCE_PAD]))
            probability = pl.recip(pl.add(pl.exp(pl.neg(confidence_logit)), 1.0))
            pl.write(scratch, [step, 0], pl.read(probability, [0, 0]))
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="markov_native_bridge_outputs", deps=[confidence_ready]) as done:
        for step in pl.range(STEPS):
            out_token = pl.const(0, pl.INT32)
            confidence = pl.const(0.0, pl.FP32)
            if rank == 0:
                out_token = pl.read(all_ids, [0, step])
                confidence = pl.read(scratch, [step, 0])
            pl.write(root_ids, [0, step], out_token)
            pl.write(root_probs, [0, step], confidence)
    return done
