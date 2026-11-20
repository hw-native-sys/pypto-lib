# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""V4.1 DSpark five-step Markov sampling."""

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from config import (
    B_DYN, DP_SIZE, MOE_TOKENS,
    DSPARK_MARKOV_RANK, DSPARK_QUERY_WIDTH,
    EP_SIZE, FLASH, DECODE_BATCH,
    SUPPORTED_EP_SIZES, SUPPORTED_LM_HEAD_TP_SIZES, LM_HEAD_TP_SIZE,
)
from lm_head import VOCAB_PER_TP, clear_lm_head_signals, lm_head
from rmsnorm import rms_norm
from hc import hc_head


# model config
D = FLASH.hidden_size
HC_MULT = FLASH.hc_mult
VOCAB = FLASH.vocab_size
MARKOV_RANK = DSPARK_MARKOV_RANK

# Shared logit capacity.
LOGIT_ROWS = (DECODE_BATCH * DSPARK_QUERY_WIDTH + 15) // 16 * 16
GROUP_LOGIT_ROWS = LM_HEAD_TP_SIZE * LOGIT_ROWS

# tiling
VOCAB_TILE = 256
RING_SLOTS = 4
SLOT_BYTES = 8 * VOCAB_TILE * 4
RING_BYTES = RING_SLOTS * SLOT_BYTES
PARTIAL_STRIDE = 16
PARTIAL_TILE = 64 * PARTIAL_STRIDE  # One gather lane per cluster.

if (D, MARKOV_RANK, DSPARK_QUERY_WIDTH) != (5120, 256, 5):
    raise ValueError("Markov recurrence requires hidden_size=5120, rank=256, and five steps")
if VOCAB % VOCAB_TILE != 0:
    raise ValueError("Markov recurrence requires complete 256-column vocabulary panels")
if EP_SIZE % LM_HEAD_TP_SIZE:
    raise ValueError("EP must be divisible by LM head TP")


@pl.jit.inline(auto_scope=False)
def _send_markov_bias(
    full_left: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left],
    panel_l1: pl.Tile[[VOCAB_TILE, 256], pl.BF16, pl.Mem.Mat],
):
    transposed = pl.tile.transpose_view(panel_l1)
    acc = pl.create_tile([16, VOCAB_TILE], dtype=pl.FP32, target_memory=pl.Mem.Acc)
    for k in pl.pipeline(0, 256, 64, stage=2):
        left_k = pl.tile.slice(full_left, [16, 64], [0, k])
        right_k = pl.tile.extract(transposed, k, 0, [64, VOCAB_TILE], target_memory=pl.Mem.Right)
        acc = pl.tile.matmul_acc(acc, left_k, right_k, init_cond=(k == 0))
    pl.system.tpush_to_aiv(acc, split=1, id=0)


@pl.jit.inline(auto_scope=False)
def _store_markov_confidence(
    hidden: pl.Tensor,
    embed: pl.Tensor,
    confidence_weight: pl.Tensor,
    confidence_state: pl.Out[pl.Tensor],
    row: pl.Scalar[pl.INDEX],
    step: pl.Scalar[pl.INDEX],
):
    hidden_vec_tile = pl.load(hidden, [row * 5 + step, 0], [1, D], target_memory=pl.Mem.Vec)
    hidden_vec = pl.reshape(hidden_vec_tile, [D])
    embedding_vec_tile = pl.load(embed, [row, 0], [1, 256], target_memory=pl.Mem.Vec)
    embedding_vec = pl.reshape(embedding_vec_tile, [256])
    weight_tile = pl.load(confidence_weight, [0, 0], [1, D + 256], target_memory=pl.Mem.Vec)
    weight = pl.reshape(weight_tile, [D + 256])
    conf_tile = pl.create_tile([64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
    with pl.reg_scope():
        dot_mask = pl.reg.full_mask(lanes=64)
        dot_out = pl.reg.active_mask(16, lanes=64)
        total = pl.reg.broadcast(0.0, lanes=64, dtype=pl.FP32)
        for chunk in pl.hardware_loop(D // 64):
            bf = pl.reg.load(hidden_vec, chunk * 64, lanes=64)
            fp = pl.reg.convert(bf, dtype=pl.FP32)
            wt = pl.reg.load(weight, chunk * 64, lanes=64)
            prod = pl.reg.mul(fp, wt, dot_mask)
            total = pl.reg.add(total, prod, dot_mask)
        for chunk in pl.hardware_loop(4):
            bf = pl.reg.load(embedding_vec, chunk * 64, lanes=64)
            fp = pl.reg.convert(bf, dtype=pl.FP32)
            wt = pl.reg.load(weight, D + chunk * 64, lanes=64)
            prod = pl.reg.mul(fp, wt, dot_mask)
            total = pl.reg.add(total, prod, dot_mask)
        dot = pl.reg.reduce_sum(total, dot_mask, reassoc=True)
        val = pl.reg.broadcast(dot, lanes=64, dtype=pl.FP32)
        negval = pl.reg.muls(val, -1.0, dot_mask)
        denom = pl.reg.adds(pl.reg.exp(negval, dot_mask), 1.0, dot_mask)
        one = pl.reg.broadcast(1.0, lanes=64, dtype=pl.FP32)
        probability = pl.reg.div(one, denom, dot_mask)
        _conf = pl.reg.store(conf_tile, 0, probability, dot_out)
    conf_view = pl.reshape(conf_tile, [1, 64])
    conf_line = pl.tile.set_validshape(conf_view, 1, 16)
    confidence_state = pl.store(conf_line, [row, step * 16], confidence_state)
    return confidence_state


@pl.jit.incore(level=pl.Level.AIC)
def _recurrence_cube(
    logits: pl.Tensor,
    embed: pl.Out[pl.Tensor],
    w2: pl.Tensor,
    partials: pl.Out[pl.Tensor],
    hidden: pl.Tensor,
    w1: pl.Tensor,
    confidence_weight: pl.Tensor,
    draft_state: pl.Out[pl.Tensor],
    confidence_state: pl.Out[pl.Tensor],
    ids: pl.Out[pl.Tensor],
    probs: pl.Out[pl.Tensor],
    anchors: pl.Tensor,
):
    pl.func_attr({"split_aiv": True})
    peer = pl.system.import_peer_buffer(name="markov_bias", peer_func="_recurrence_vector")
    pl.system.aic_initialize_pipe(peer, dir_mask=1, slot_size=SLOT_BYTES, slot_num=RING_SLOTS, id=0)
    block = pl.tile.get_block_idx()
    blocks = pl.tile.get_block_num()
    batch = pl.tensor.dim(anchors, 0)
    # Reuse the first two W2 panels across all five steps.
    first_weight = pl.load(w2, [block * VOCAB_TILE, 0], [VOCAB_TILE, 256], target_memory=pl.Mem.Mat)
    second_weight = pl.load(w2, [(block + blocks) * VOCAB_TILE, 0], [VOCAB_TILE, 256], target_memory=pl.Mem.Mat)
    pl.system.syncall(core_type=pl.KernelType.MIX)
    for step in pl.range(5):
        for first in pl.range(0, batch, 16):
            embed_l1 = pl.load(embed, [first, 0], [16, 256], target_memory=pl.Mem.Mat)
            full_left = pl.tile.extract(embed_l1, 0, 0, [16, 256], target_memory=pl.Mem.Left)
            _send_markov_bias(full_left, first_weight)
            _send_markov_bias(full_left, second_weight)
            for wave in pl.pipeline((VOCAB // VOCAB_TILE + blocks - 1 - block) // blocks - 2, stage=2):
                panel = block + (wave + 2) * blocks
                col = panel * VOCAB_TILE
                panel_l1 = pl.load(w2, [col, 0], [VOCAB_TILE, 256], target_memory=pl.Mem.Mat)
                _send_markov_bias(full_left, panel_l1)
        pl.system.syncall(core_type=pl.KernelType.MIX)
    pl.system.syncall(core_type=pl.KernelType.MIX)
    return ids


@pl.jit.incore(level=pl.Level.AIV)
def _recurrence_vector(
    logits: pl.Tensor,
    embed: pl.Out[pl.Tensor],
    w2: pl.Tensor,
    partials: pl.Out[pl.Tensor],
    hidden: pl.Tensor,
    w1: pl.Tensor,
    confidence_weight: pl.Tensor,
    draft_state: pl.Out[pl.Tensor],
    confidence_state: pl.Out[pl.Tensor],
    ids: pl.Out[pl.Tensor],
    probs: pl.Out[pl.Tensor],
    anchors: pl.Tensor,
):
    pl.func_attr({"split_aiv": True, "dual_aiv_dispatch": True})
    ring = pl.system.reserve_buffer(name="markov_bias", size=RING_BYTES)
    pl.system.aiv_initialize_pipe(ring, dir_mask=1, slot_size=SLOT_BYTES, slot_num=RING_SLOTS, id=0)
    block = pl.tile.get_block_idx()
    blocks = pl.tile.get_block_num()
    lane = pl.tile.get_subblock_idx()
    worker = block * 2 + lane
    batch = pl.tensor.dim(anchors, 0)
    padded_rows = pl.tensor.dim(embed, 0)
    for row in pl.range(worker, padded_rows, blocks * 2):
        token = pl.read(anchors, [pl.min(row, batch - 1)])
        initial = pl.load(w1, [token, 0], [1, 256], target_memory=pl.Mem.Vec)
        embed = pl.store(initial, [row, 0], embed)
    pl.system.barrier(pipe=pl.PipeType.ALL)
    pl.system.syncall(core_type=pl.KernelType.MIX)
    for step in pl.range(5):
        # High-numbered lanes compute confidence from the preceding embedding.
        for row in pl.range(blocks * 2 - 1 - worker, batch, blocks * 2):
            confidence_state = _store_markov_confidence(
                hidden, embed, confidence_weight,
                confidence_state, row, step,
            )
        for first in pl.range(0, batch, 16):
            row0 = first + lane * 8
            valid = pl.max(0, pl.min(batch - row0, 8))
            best_tile = pl.create_tile([512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            id_tile = pl.create_tile([512], dtype=pl.INT32, target_memory=pl.Mem.Vec)
            with pl.reg_scope():
                init_mask = pl.reg.full_mask(lanes=64)
                negative_inf_bits = pl.reg.broadcast(-8388608, lanes=64, dtype=pl.INT32)
                neginf = pl.reg.interpret_cast(negative_inf_bits, dtype=pl.FP32)
                zero = pl.reg.broadcast(0, lanes=64, dtype=pl.INT32)
                for init_row in pl.hardware_loop(valid):
                    _init_scores = pl.reg.store(best_tile, init_row * 64, neginf, init_mask)
                    _init_ids = pl.reg.store(id_tile, init_row * 64, zero, init_mask)
            if valid > 0:
                for panel in pl.range(block, VOCAB // VOCAB_TILE, blocks):
                    col = panel * VOCAB_TILE
                    logit_col = step * VOCAB + col
                    base = pl.load(logits, [row0, logit_col], [8, VOCAB_TILE], valid_shape=[valid, VOCAB_TILE], target_memory=pl.Mem.Vec)
                    bias = pl.system.tpop_from_aic(shape=[8, VOCAB_TILE], dtype=pl.FP32, split=1, id=0)
                    base_full = pl.tile.set_validshape(base, 8, VOCAB_TILE)
                    summed = pl.add(base_full, bias)
                    pl.system.tfree_to_aic(bias, id=0)
                    panel_scores = pl.reshape(summed, [8 * VOCAB_TILE])
                    start = pl.cast(col, pl.INT32)
                    with pl.reg_scope():
                        score_mask = pl.reg.full_mask(lanes=64)
                        for score_row in pl.hardware_loop(valid):
                            best = pl.reg.load(best_tile, score_row * 64, lanes=64)
                            winner = pl.reg.load(id_tile, score_row * 64, lanes=64)
                            token_col = pl.reg.index_gen(start, lanes=64, dtype=pl.INT32)
                            for seg in pl.hardware_loop(VOCAB_TILE // 64):
                                score = pl.reg.load(panel_scores, score_row * VOCAB_TILE + seg * 64, lanes=64)
                                better = pl.reg.compare(score, best, score_mask, predicate="gt")
                                best = pl.reg.select(better, score, best)
                                winner = pl.reg.select(better, token_col, winner)
                                token_col = pl.reg.adds(token_col, 64, score_mask)
                            _save_scores = pl.reg.store(best_tile, score_row * 64, best, score_mask)
                            _save_ids = pl.reg.store(id_tile, score_row * 64, winner, score_mask)
            else:
                for panel in pl.range(block, VOCAB // VOCAB_TILE, blocks):
                    discard = pl.system.tpop_from_aic(shape=[8, VOCAB_TILE], dtype=pl.FP32, split=1, id=0)
                    pl.system.tfree_to_aic(discard, id=0)
            if valid > 0:
                block_tile = pl.create_tile([8 * PARTIAL_STRIDE], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                with pl.reg_scope():
                    partial_mask = pl.reg.full_mask(lanes=64)
                    half_line = pl.reg.active_mask(8, lanes=64)
                    invalid_id = pl.reg.broadcast(2147483647, lanes=64, dtype=pl.INT32)
                    for partial_row in pl.hardware_loop(valid):
                        lane_scores = pl.reg.load(best_tile, partial_row * 64, lanes=64)
                        lane_ids = pl.reg.load(id_tile, partial_row * 64, lanes=64)
                        highest = pl.reg.reduce_max(lane_scores, partial_mask)
                        block_max = pl.reg.broadcast(highest, lanes=64, dtype=pl.FP32)
                        ties = pl.reg.compare(lane_scores, block_max, partial_mask, predicate="eq")
                        selected = pl.reg.select(ties, lane_ids, invalid_id)
                        lowest_id = pl.reg.reduce_min(selected, partial_mask)
                        chosen = pl.reg.broadcast(lowest_id, lanes=64, dtype=pl.INT32)
                        # One 64-byte record per block. Each half occupies one aligned vector store.
                        score_bits = pl.reg.interpret_cast(block_max, dtype=pl.INT32)
                        _partial_scores = pl.reg.store(block_tile, partial_row * PARTIAL_STRIDE, score_bits, half_line)
                        _partial_ids = pl.reg.store(block_tile, partial_row * PARTIAL_STRIDE + 8, chosen, half_line)
                active_partials = pl.tile.set_validshape(pl.reshape(block_tile, [8, PARTIAL_STRIDE]), valid, PARTIAL_STRIDE)
                partials = pl.store(active_partials, [row0, block * PARTIAL_STRIDE], partials)
        pl.system.barrier(pipe=pl.PipeType.ALL)
        pl.system.syncall(core_type=pl.KernelType.AIV)
        for row in pl.range(worker, batch, blocks * 2):
            block_results_tile = pl.load(
                partials, [row, 0], [1, PARTIAL_TILE],
                valid_shape=[1, blocks * PARTIAL_STRIDE], target_memory=pl.Mem.Vec,
            )
            block_results_full = pl.tile.set_validshape(block_results_tile, 1, PARTIAL_TILE)
            block_results = pl.reshape(block_results_full, [PARTIAL_TILE])
            token_tile = pl.create_tile([64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
            with pl.reg_scope():
                winner_mask = pl.reg.full_mask(lanes=64)
                active = pl.reg.active_mask(blocks, lanes=64)
                winner_out = pl.reg.active_mask(16, lanes=64)
                ix = pl.reg.index_gen(0, lanes=64, dtype=pl.UINT32)
                offsets = pl.reg.muls(ix, PARTIAL_STRIDE, winner_mask)
                score_words = pl.reg.gather(block_results, offsets, active)
                vals = pl.reg.interpret_cast(score_words, dtype=pl.FP32)
                id_offsets = pl.reg.adds(offsets, 8, winner_mask)
                candidates = pl.reg.gather(block_results, id_offsets, active)
                high = pl.reg.reduce_max(vals, active)
                global_max = pl.reg.broadcast(high, lanes=64, dtype=pl.FP32)
                eq = pl.reg.compare(vals, global_max, active, predicate="eq")
                invalid_winner = pl.reg.broadcast(2147483647, lanes=64, dtype=pl.INT32)
                tied = pl.reg.select(eq, candidates, invalid_winner)
                low = pl.reg.reduce_min(tied, active)
                token_reg = pl.reg.broadcast(low, lanes=64, dtype=pl.INT32)
                _id = pl.reg.store(token_tile, 0, token_reg, winner_out)
            token_view = pl.reshape(token_tile, [1, 64])
            token_line = pl.tile.set_validshape(token_view, 1, 16)
            draft_state = pl.store(token_line, [row, step * 16], draft_state)
            if step < 4:
                next_id = pl.tile.read(token_tile, [0])
                next_e = pl.load(w1, [next_id, 0], [1, 256], target_memory=pl.Mem.Vec)
                embed = pl.store(next_e, [row, 0], embed)
        pl.system.barrier(pipe=pl.PipeType.ALL)
        pl.system.syncall(core_type=pl.KernelType.MIX)
    # Eight rows of five output words give disjoint aligned output ranges.
    for first_row in pl.range(worker * 8, batch, blocks * 16):
        rows = pl.min(8, batch - first_row)
        draft_tile = pl.load(draft_state, [first_row, 0], [8, 80], valid_shape=[rows, 80], target_memory=pl.Mem.Vec)
        draft_rows_full = pl.tile.set_validshape(draft_tile, 8, 80)
        draft_rows = pl.reshape(draft_rows_full, [640])
        conf_tile = pl.load(confidence_state, [first_row, 0], [8, 80], valid_shape=[rows, 80], target_memory=pl.Mem.Vec)
        confidence_rows_full = pl.tile.set_validshape(conf_tile, 8, 80)
        confidence_rows = pl.reshape(confidence_rows_full, [640])
        pack_i = pl.create_tile([64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
        pack_c = pl.create_tile([64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        with pl.reg_scope():
            pack_mask = pl.reg.active_mask(5, lanes=64)
            lane_idx = pl.reg.index_gen(0, lanes=64, dtype=pl.UINT32)
            col_idx = pl.reg.muls(lane_idx, 16, pack_mask)
            for publish_row in pl.hardware_loop(rows):
                pack_offset = pl.reg.adds(col_idx, pl.cast(publish_row * 80, pl.UINT32), pack_mask)
                draft_values = pl.reg.gather(draft_rows, pack_offset, pack_mask)
                conf_values = pl.reg.gather(confidence_rows, pack_offset, pack_mask)
                _draft = pl.reg.store(pack_i, publish_row * 8, draft_values, pack_mask)
                _conf = pl.reg.store(pack_c, publish_row * 8, conf_values, pack_mask)
        ids = pl.store(pl.tile.set_validshape(pl.reshape(pack_i, [8, 8]), rows, 5), [first_row, 0], ids)
        probs = pl.store(pl.tile.set_validshape(pl.reshape(pack_c, [8, 8]), rows, 5), [first_row, 0], probs)
    pl.system.barrier(pipe=pl.PipeType.ALL)
    pl.system.syncall(core_type=pl.KernelType.MIX)
    return probs


@pl.jit.incore(level=pl.Level.CORE_GROUP)
def _recurrence_group(
    logits: pl.Tensor,
    embed: pl.Out[pl.Tensor],
    w2: pl.Tensor,
    partials: pl.Out[pl.Tensor],
    hidden: pl.Tensor,
    w1: pl.Tensor,
    confidence_weight: pl.Tensor,
    draft_state: pl.Out[pl.Tensor],
    confidence_state: pl.Out[pl.Tensor],
    ids: pl.Out[pl.Tensor],
    probs: pl.Out[pl.Tensor],
    anchors: pl.Tensor,
):
    ids = _recurrence_cube(
        logits, embed, w2, partials,
        hidden, w1, confidence_weight,
        draft_state, confidence_state, ids, probs, anchors,
    )
    probs = _recurrence_vector(
        logits, embed, w2, partials,
        hidden, w1, confidence_weight,
        draft_state, confidence_state, ids, probs, anchors,
    )
    return ids, probs


@pl.jit.inline(auto_scope=False)
def markov_sample(
    base_logits: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, VOCAB], pl.FP32],
    head_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16],
    anchor_ids: pl.Tensor[[B_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB, MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB, MARKOV_RANK], pl.BF16],
    confidence_weight: pl.Tensor[[1, D + MARKOV_RANK], pl.FP32],
    draft_ids: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.INT32]],
    confidences: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.FP32]],
):
    """Generate five drafts with the previous sampled ID at every step."""
    base_logits.bind_dynamic(0, B_DYN)
    head_hidden.bind_dynamic(0, B_DYN)
    anchor_ids.bind_dynamic(0, B_DYN)
    draft_ids.bind_dynamic(0, B_DYN)
    confidences.bind_dynamic(0, B_DYN)
    batch = pl.tensor.dim(anchor_ids, 0)
    padded = (batch + 15) // 16 * 16
    embedding = pl.create_tensor([padded, 256], dtype=pl.BF16)
    clusters = pl.system.available_cluster_count()
    partial_width = clusters * PARTIAL_STRIDE
    partials = pl.create_tensor([batch, partial_width], dtype=pl.INT32)
    draft_state = pl.create_tensor([batch, 80], dtype=pl.INT32)
    confidence_state = pl.create_tensor([batch, 80], dtype=pl.FP32)
    logits = pl.reshape(base_logits, [batch, DSPARK_QUERY_WIDTH * VOCAB])
    hidden = pl.reshape(head_hidden, [batch * DSPARK_QUERY_WIDTH, D])
    # Every cluster must reach the mixed and AIV hardware barriers.
    with pl.spmd(pl.system.available_cluster_count(), name_hint="markov_sample", sync_start=True):
        ids_out, conf_out = _recurrence_group(
            logits, embedding, markov_w2, partials, hidden, markov_w1, confidence_weight,
            draft_state, confidence_state, draft_ids, confidences, anchor_ids,
        )
    return ids_out, conf_out


@pl.jit(auto_scope=False)
def markov_sample_test(
    base_logits: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, VOCAB], pl.FP32],
    head_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16],
    anchor_ids: pl.Tensor[[B_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB, MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB, MARKOV_RANK], pl.BF16],
    confidence_weight: pl.Tensor[[1, D + MARKOV_RANK], pl.FP32],
    draft_ids: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.INT32]],
    confidences: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.FP32]],
):
    """Validate Markov sampling from logits and resolved anchor IDs."""
    return markov_sample(
        base_logits, head_hidden, anchor_ids, markov_w1, markov_w2,
        confidence_weight, draft_ids, confidences,
    )


def _markov_output(
    draft_x_hc: pl.Tensor[[MOE_TOKENS, HC_MULT, D], pl.FP32],
    draft_pre_mix: pl.Tensor[[MOE_TOKENS, HC_MULT], pl.FP32],
    head_hidden: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16]],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[D, VOCAB_PER_TP], pl.BF16, pl.NZ],
    num_sampled: pl.Tensor[[B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[B_DYN], pl.INT64],
    anchor_ids: pl.Tensor[[B_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB, MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB, MARKOV_RANK], pl.BF16],
    confidence_weight: pl.Tensor[[1, D + MARKOV_RANK], pl.FP32],
    draft_ids: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.INT32]],
    confidences: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.FP32]],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[LOGIT_ROWS, VOCAB], pl.BF16],
    logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    logits_epoch: pl.Scalar[pl.INT32],
):
    """Project draft hidden states to full-vocabulary logits, then sample five IDs."""
    head_hidden.bind_dynamic(0, B_DYN)
    num_sampled.bind_dynamic(0, B_DYN)
    last_sampled.bind_dynamic(0, B_DYN)
    next_prefill_tokens.bind_dynamic(0, B_DYN)
    draft_ids.bind_dynamic(0, B_DYN)
    confidences.bind_dynamic(0, B_DYN)
    batch = pl.tensor.dim(num_sampled, 0)
    rows = batch * DSPARK_QUERY_WIDTH
    flat_hidden = pl.reshape(head_hidden, [rows, D])
    hc_head(draft_x_hc[:rows, :, :], draft_pre_mix[:rows, :], flat_hidden, worker_limit=64)
    normalized = pl.create_tensor([rows, D], dtype=pl.BF16)
    rms_norm(flat_hidden, final_norm_weight, normalized, rows, width=D, zero_tail=False)
    row_indices = pl.create_tensor([LOGIT_ROWS], dtype=pl.INT32)
    for _core in pl.spmd(1, name_hint="dspark_logit_rows"):
        for row in pl.range(LOGIT_ROWS):
            source = pl.cast(-1, pl.INT32)
            if row < rows:
                source = pl.cast(row, pl.INT32)
            pl.write(row_indices, [row], source)
    logits = pl.create_tensor([LOGIT_ROWS, VOCAB], dtype=pl.FP32)
    lm_head(
        normalized, lm_head_weight, row_indices, logits,
        hidden_window, hidden_done, logits_window, logits_done,
        rank, LOGIT_ROWS,
    )
    clear_lm_head_signals(logits, hidden_done, logits_done)
    base_logits = pl.reshape(logits[:rows, :], [batch, DSPARK_QUERY_WIDTH, VOCAB])
    markov_sample(
        base_logits, head_hidden, anchor_ids, markov_w1, markov_w2,
        confidence_weight, draft_ids, confidences,
    )
    return draft_ids, confidences


markov_output = pl.jit.inline(auto_scope=False)(_markov_output)
l2_markov_output = pl.jit(auto_scope=False)(_markov_output)


@pl.jit.host
def l3_markov_output(
    draft_x_hc: pl.Tensor[[EP_SIZE, MOE_TOKENS, HC_MULT, D], pl.FP32],
    draft_pre_mix: pl.Tensor[[EP_SIZE, MOE_TOKENS, HC_MULT], pl.FP32],
    head_hidden: pl.Out[pl.Tensor[[EP_SIZE, B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16]],
    final_norm_weight: pl.Tensor[[EP_SIZE, D], pl.BF16],
    lm_head_weight: pl.Tensor[[EP_SIZE, D, VOCAB_PER_TP], pl.BF16],
    num_sampled: pl.Tensor[[EP_SIZE, B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[EP_SIZE, B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[EP_SIZE, B_DYN], pl.INT64],
    anchor_ids: pl.Tensor[[EP_SIZE, B_DYN], pl.INT64],
    markov_w1: pl.Tensor[[EP_SIZE, VOCAB, MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[EP_SIZE, VOCAB, MARKOV_RANK], pl.BF16],
    confidence_weight: pl.Tensor[[EP_SIZE, 1, D + MARKOV_RANK], pl.FP32],
    draft_ids: pl.Out[pl.Tensor[[EP_SIZE, B_DYN, DSPARK_QUERY_WIDTH], pl.INT32]],
    confidences: pl.Out[pl.Tensor[[EP_SIZE, B_DYN, DSPARK_QUERY_WIDTH], pl.FP32]],
    logits_epoch: pl.Scalar[pl.INT32],
):
    head_hidden.bind_dynamic(1, B_DYN)
    num_sampled.bind_dynamic(1, B_DYN)
    last_sampled.bind_dynamic(1, B_DYN)
    next_prefill_tokens.bind_dynamic(1, B_DYN)
    anchor_ids.bind_dynamic(1, B_DYN)
    draft_ids.bind_dynamic(1, B_DYN)
    confidences.bind_dynamic(1, B_DYN)
    hidden_window_buf = pld.alloc_window_buffer([GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    hidden_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    logits_window_buf = pld.alloc_window_buffer([LOGIT_ROWS, VOCAB], dtype=pl.BF16)
    logits_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        hidden_window = pld.window(hidden_window_buf, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        hidden_done = pld.window(hidden_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        logits_window = pld.window(logits_window_buf, [LOGIT_ROWS, VOCAB], dtype=pl.BF16)
        logits_done = pld.window(logits_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        l2_markov_output(
            draft_x_hc[rank], draft_pre_mix[rank], head_hidden[rank],
            final_norm_weight[rank], lm_head_weight[rank],
            num_sampled[rank], last_sampled[rank], next_prefill_tokens[rank], anchor_ids[rank],
            markov_w1[rank], markov_w2[rank], confidence_weight[rank],
            draft_ids[rank], confidences[rank],
            hidden_window, hidden_done, logits_window, logits_done,
            rank, logits_epoch, device=rank,
        )


def golden_markov_sample(
    base_logits: torch.Tensor,
    anchor_ids: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    head_hidden: torch.Tensor,
    confidence_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy full-vocabulary sampling with a previous-token Markov bias."""
    if base_logits.ndim != 3 or anchor_ids.shape != base_logits.shape[:1]:
        raise ValueError("base logits must be [batch, steps, vocab] with one anchor per request")
    batch, steps, vocab = base_logits.shape
    if markov_w1.shape[0] != vocab or markov_w2.shape != markov_w1.shape:
        raise ValueError("Markov factors must be [vocab, rank]")
    confidence_shape = (1, head_hidden.shape[-1] + markov_w1.shape[-1])
    if head_hidden.shape[:2] != (batch, steps) or confidence_weight.shape != confidence_shape:
        raise ValueError("confidence weight must cover hidden and Markov dimensions")
    drafts = torch.empty(batch, steps, dtype=anchor_ids.dtype, device=anchor_ids.device)
    confidences = []
    previous = anchor_ids.long()
    for step in range(steps):
        embedding = markov_w1[previous]
        bias = embedding.float() @ markov_w2.float().T
        logits = base_logits[:, step].float()
        biased_logits = logits + bias
        previous = biased_logits.argmax(dim=-1)
        drafts[:, step] = previous
        hidden_features = head_hidden[:, step].float()
        markov_features = embedding.float()
        features = torch.cat((hidden_features, markov_features), dim=-1)
        confidence_logits = features @ confidence_weight.float().T
        confidence = torch.sigmoid(confidence_logits).squeeze(-1)
        confidences.append(confidence)
    return drafts, torch.stack(confidences, dim=-1)


def golden_markov_sample_test(tensors):
    """Fill draft IDs and confidences for the standalone sampling entry."""
    ids, confidence = golden_markov_sample(
        tensors["base_logits"], tensors["anchor_ids"], tensors["markov_w1"],
        tensors["markov_w2"], tensors["head_hidden"], tensors["confidence_weight"],
    )
    tensors["draft_ids"].copy_(ids.int())
    tensors["confidences"].copy_(confidence)


def build_tensor_specs(batch=2):
    """Build five-step Markov chains, tied biases, and nonzero confidence features."""
    from golden import TensorSpec

    anchors = torch.arange(batch, dtype=torch.int64) * (DSPARK_QUERY_WIDTH + 1)
    base = torch.zeros(batch, DSPARK_QUERY_WIDTH, VOCAB, dtype=torch.float32)
    hidden = torch.zeros(batch, DSPARK_QUERY_WIDTH, D, dtype=torch.bfloat16)
    hidden[..., 0] = 1.0
    hidden[..., 1] = torch.arange(DSPARK_QUERY_WIDTH).to(torch.bfloat16)
    w1 = torch.zeros(VOCAB, MARKOV_RANK, dtype=torch.bfloat16)
    w2 = torch.full_like(w1, -16.0)
    for request in range(batch):
        for step in range(DSPARK_QUERY_WIDTH):
            token = request * (DSPARK_QUERY_WIDTH + 1) + step
            w1[token, token] = 16.0
            w2[0, token] = 16.0
            w2[token + 1, token] = 16.0
            base[request, step, token + 1] = 1.0
    confidence_weight = torch.zeros(1, D + MARKOV_RANK)
    confidence_weight[0, 0] = 0.25
    confidence_weight[0, 1] = 0.125
    confidence_weight[0, D:] = 1.0 / 32
    return [
        TensorSpec("base_logits", list(base.shape), torch.float32, init_value=base),
        TensorSpec("head_hidden", list(hidden.shape), torch.bfloat16, init_value=hidden),
        TensorSpec("anchor_ids", [batch], torch.int64, init_value=anchors),
        TensorSpec("markov_w1", list(w1.shape), torch.bfloat16, init_value=w1),
        TensorSpec("markov_w2", list(w2.shape), torch.bfloat16, init_value=w2),
        TensorSpec("confidence_weight", list(confidence_weight.shape), torch.float32, init_value=confidence_weight),
        TensorSpec("draft_ids", [batch, DSPARK_QUERY_WIDTH], torch.int32),
        TensorSpec("confidences", [batch, DSPARK_QUERY_WIDTH], torch.float32),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser(description="V4.1 DSpark Markov sampling validation")
    parser.add_argument("--lm-head-tp", type=int, default=LM_HEAD_TP_SIZE, choices=SUPPORTED_LM_HEAD_TP_SIZES)
    parser.add_argument("--dp", type=int, default=DP_SIZE)
    parser.add_argument("--ep", type=int, default=EP_SIZE, choices=SUPPORTED_EP_SIZES)
    parser.add_argument("-p", "--platform", choices=("a5",), default="a5")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--runtime-dir")
    args = parser.parse_args()
    if (args.lm_head_tp, args.dp, args.ep) != (LM_HEAD_TP_SIZE, DP_SIZE, EP_SIZE):
        parser.error("--lm-head-tp/--dp/--ep must match the import-time configuration")
    result = run(
        fn=markov_sample_test,
        specs=build_tensor_specs(),
        golden_fn=golden_markov_sample_test,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        config=dict(backend_type=args.platform, execution_mode=False, device_id=args.device),
        rtol=1e-3,
        atol=1e-3,
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
