# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Drafter base logits with the original K order and one vocabulary shard per rank."""

import pypto.language as pl
import pypto.language.distributed as pld

from .tp8_score import (
    CHUNKS_PER_SHARD, DSPARK_MARKOV_RANK, PAIR_ROW, STEPS, VOCAB_CHUNK,
    m8_fold_and_publish_chunks, m8_wait_fold_and_embed, score_shard_chunks,
)


D = 4096
ROWS = 5
M_TILE = 16
K_TILE = 256
N_TILE = 256
VOCAB_SHARD = 16384
FULL = VOCAB_SHARD // N_TILE
RANKS = 8


@pl.jit.inline(auto_scope=False)
def publish_base_rows(
    normalized_head: pl.Tensor[[32, D], pl.BF16],
    logit_row_indices: pl.Tensor[[32], pl.INT32],
    last_sampled: pl.Tensor[[1], pl.INT64],
    num_sampled: pl.Tensor[[1], pl.INT32],
    next_prefill: pl.Tensor[[1], pl.INT64],
    phase_epoch: pl.Tensor[[1], pl.INT32],
    hidden_window: pld.DistributedTensor[[2*M_TILE, D], pl.BF16],
    seed_window: pld.DistributedTensor[[2, 8], pl.INT64],
    ready_epoch: pld.DistributedTensor[[2, 32], pl.INT32],
    slot: pl.Scalar[pl.INDEX],
    normalized_ready: pl.Scalar[pl.TASK_ID],
):
    """Call on the existing rank-zero request owner at the head producer."""
    selected = pl.create_tensor([ROWS,D],dtype=pl.BF16)
    seed = pl.create_tensor([1,8],dtype=pl.INT64)
    with pl.spmd(ROWS,name_hint="head8_select_logit_rows",deps=[normalized_ready]) as data:
        row = pl.tile.get_block_idx()
        source = pl.read(logit_row_indices,[row])
        selected[row:row+1,:] = pl.full([1,D],dtype=pl.BF16,value=0.0)
        if source >= 0:
            used = pl.min(source,31)
            selected[row:row+1,:] = normalized_head[used:used+1,:]
    with pl.at(level=pl.Level.CORE_GROUP,name_hint="head8_select_seed",deps=[normalized_ready]) as seeded:
        token = pl.read(next_prefill,[0])
        if pl.read(num_sampled,[0]) > 0:
            token = pl.read(last_sampled,[0])
        for col in pl.range(8):
            pl.write(seed,[0,col],pl.const(0,pl.INT64))
        pl.write(seed,[0,0],token)
        pl.system.cacheinvalid()
        pl.system.fence()
    with pl.spmd(RANKS,name_hint="head8_publish_input",deps=[data,seeded]) as sent:
        peer = pl.tile.get_block_idx()
        pld.tensor.put(dst=hidden_window,peer=peer,src=selected,dst_offsets=[slot*M_TILE,0],
                       src_offsets=[0,0],shape=[ROWS,D],chunk_rows=1,chunk_cols=1024,pipeline=True)
        pld.tensor.put(dst=seed_window,peer=peer,src=seed,dst_offsets=[slot,0],
                       src_offsets=[0,0],shape=[1,8])
        pld.system.notify(target=ready_epoch,peer=peer,offsets=[slot,0],
                          value=pl.read(phase_epoch,[0]),op=pld.NotifyOp.Set)
    return sent


@pl.jit.inline(auto_scope=False)
def wait_base_rows(
    ready_epoch: pld.DistributedTensor[[2,32],pl.INT32],
    phase_epoch: pl.Tensor[[1],pl.INT32],
    slot: pl.Scalar[pl.INDEX],
    metadata_ready: pl.Scalar[pl.TASK_ID],
):
    with pl.at(level=pl.Level.CORE_GROUP,name_hint="head8_wait_current_rows",deps=[metadata_ready]) as ready:
        pld.system.defer_wait(signal=ready_epoch,offsets=[slot,0],expected=pl.read(phase_epoch,[0]),
                               cmp=pld.WaitCmp.Ge)
    return ready


@pl.jit.inline(auto_scope=False)
def base_logits_shard(
    hidden_window: pld.DistributedTensor[[2*M_TILE,D],pl.BF16],
    weight: pl.Tensor[[VOCAB_SHARD,D],pl.BF16,pl.NZ],
    logits: pl.Tensor[[ROWS,VOCAB_SHARD],pl.FP32],
    slot: pl.Scalar[pl.INDEX],
    features_ready: pl.Scalar[pl.TASK_ID],
    cube_blocks: pl.constexpr,
):
    with pl.spmd(cube_blocks,name_hint="head8_vocab_local",deps=[features_ready]) as done:
        pl.set_cache_policy(weight,pl.CachePolicy.BYPASS)
        block=pl.tile.get_block_idx()
        for nt in pl.range(block,FULL,cube_blocks):
            n0=nt*N_TILE
            acc=pl.create_tensor([M_TILE,N_TILE],dtype=pl.FP32)
            for kt in pl.pipeline(D//K_TILE,stage=2):
                k0=kt*K_TILE
                x=pl.slice(hidden_window,[M_TILE,K_TILE],[slot*M_TILE,k0],valid_shape=[ROWS,K_TILE])
                w=weight[n0:n0+N_TILE,k0:k0+K_TILE]
                acc=pl.matmul_acc(acc,x,w,b_trans=True,init_cond=(kt==0))
            logits[0:ROWS,n0:n0+N_TILE]=pl.set_validshape(acc,ROWS,N_TILE)
    return done

@pl.jit.inline
def m8_verify_local_steps(
    w1: pl.Tensor[[129280, DSPARK_MARKOV_RANK], pl.BF16],
    w2: pl.Tensor[[VOCAB_SHARD, DSPARK_MARKOV_RANK], pl.BF16],
    ids: pl.Tensor[[1, STEPS], pl.INT32],
    phase_epoch: pl.Tensor[[1], pl.INT32],
    local: pl.Tensor[[STEPS, VOCAB_SHARD], pl.FP32],
    seed_window: pld.DistributedTensor[[2, 8], pl.INT64],
    candidates: pld.DistributedTensor[[2 * STEPS * RANKS, PAIR_ROW], pl.FP32],
    arrivals: pld.DistributedTensor[[2 * STEPS * RANKS, 32], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    slot: pl.Scalar[pl.INDEX],
    base_ready: pl.Scalar[pl.TASK_ID],
):
    active = pl.const(63, pl.INT32)
    start = pl.cast((rank * 63 + 1) * VOCAB_CHUNK, pl.INT32)
    if rank == 0:
        active = pl.const(64, pl.INT32)
        start = pl.const(0, pl.INT32)
    embedding_rows = pl.create_tensor([STEPS, DSPARK_MARKOV_RANK], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="markov_native_initial_embedding",
               deps=[base_ready]) as previous:
        pl.system.cacheinvalid()
        token = pl.cast(pl.read(seed_window, [slot, 0]), pl.INDEX)
        embedding_rows[0:1, :] = w1[token:token + 1, :]
    for step in pl.range(STEPS):
        embedding = embedding_rows[step:step + 1, :]
        winners = pl.create_tensor([CHUNKS_PER_SHARD, PAIR_ROW], dtype=pl.FP32)
        scored = score_shard_chunks(embedding, local, w2, winners, active,
                                    start, pl.cast(step, pl.INT32), previous)
        sent = m8_fold_and_publish_chunks(winners, active, candidates, arrivals,
                                          phase_epoch, rank, slot, step, scored)
        previous = m8_wait_fold_and_embed(candidates, arrivals, ids, w1, embedding_rows,
                                          phase_epoch, slot, step, sent)
    return previous


@pl.jit.inline(auto_scope=False)
def run_head8_steps(
    weight: pl.Tensor[[16384,4096],pl.BF16,pl.NZ],
    hidden_window: pld.DistributedTensor[[32,4096],pl.BF16],
    w1: pl.Tensor[[129280,256],pl.BF16],
    w2: pl.Tensor[[16384,256],pl.BF16],
    ids: pl.Tensor[[1,5],pl.INT32],
    epoch: pl.Tensor[[1],pl.INT32],
    seed: pld.DistributedTensor[[2,8],pl.INT64],
    seed_ready: pld.DistributedTensor[[2,32],pl.INT32],
    pairs: pld.DistributedTensor[[80,16],pl.FP32],
    arrivals: pld.DistributedTensor[[80,32],pl.INT32],
    rank: pl.Scalar[pl.INT32],
    slot: pl.Scalar[pl.INDEX],
    metadata_ready: pl.Scalar[pl.TASK_ID],
):
    features = wait_base_rows(seed_ready,epoch,slot,metadata_ready)
    local_m8_logits = pl.create_tensor([STEPS,VOCAB_SHARD],dtype=pl.FP32)
    base_ready = base_logits_shard(hidden_window,weight,local_m8_logits,slot,features,24)
    return m8_verify_local_steps(w1,w2,ids,epoch,local_m8_logits,seed,pairs,arrivals,rank,slot,base_ready)
