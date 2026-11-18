# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.language as pl

import pypto.language.distributed as pld

from .config import FLASH as M

from .lm_head import DONE_VALUE, GROUP_LOGIT_ROWS, MAX_LOGIT_ROWS, TP_SIZE, VOCAB_PER_TP, lm_head

from .markov_head import markov_head

B_DYN = pl.dynamic("NATIVE_M_DSPARK_MARKOV_B_DYN")

DSPARK_MARKOV_RANK = 256

DSPARK_QUERY_WIDTH = 5

DSPARK_QUERY_PAD = 8

DSPARK_SUPPORTED_BATCHES = (4,)

DSPARK_MAX_BATCH = max(DSPARK_SUPPORTED_BATCHES)

DSPARK_MOE_TOKENS = DSPARK_MAX_BATCH * DSPARK_QUERY_PAD

assert DSPARK_QUERY_WIDTH < DSPARK_QUERY_PAD

D = M.hidden_size

VOCAB = M.vocab_size

EPS = M.rms_norm_eps

HIDDEN_TILE = 512

RMS_M_TILE = 8

LM_M_TILE = 16

LM_N_TILE = 128

LM_K_TILE = 256

MARKOV_M_TILE = 16

MARKOV_ID_PAD = 16

CONFIDENCE_PAD = 8

GREEDY_VOCAB_CHUNK = 256

GREEDY_NUM_CHUNKS = VOCAB // GREEDY_VOCAB_CHUNK

GREEDY_PARTITION_TILE = 16

GREEDY_PARTITIONS = (GREEDY_NUM_CHUNKS + GREEDY_PARTITION_TILE - 1) // GREEDY_PARTITION_TILE

GREEDY_CHUNK_PAD = 512

GREEDY_ARGMAX_ROWS = 8

NEG_INF = -3.402823e38

assert D % HIDDEN_TILE == 0

assert D % LM_K_TILE == 0

assert VOCAB % LM_N_TILE == 0

assert VOCAB % GREEDY_VOCAB_CHUNK == 0

assert GREEDY_NUM_CHUNKS <= GREEDY_CHUNK_PAD

assert CONFIDENCE_PAD * 4 == 32

for _batch in DSPARK_SUPPORTED_BATCHES:
    assert (_batch * DSPARK_QUERY_PAD) % LM_M_TILE == 0


@pl.jit.inline
def normalize_head_hidden(
    head_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    normalized: pl.Tensor[[DSPARK_MOE_TOKENS, D], pl.BF16],
    head_ready_tid: pl.Scalar[pl.TASK_ID],
):
    batch = pl.tensor.dim(head_hidden, 0)
    active_tokens = batch * DSPARK_QUERY_WIDTH
    padded_tokens = batch * DSPARK_QUERY_PAD
    hidden_flat = pl.reshape(head_hidden, [active_tokens, D])
    padded_hidden = pl.create_tensor([DSPARK_MOE_TOKENS, D], dtype=pl.BF16)
    hidden_blocks = D // HIDDEN_TILE
    with pl.spmd(
        (DSPARK_MOE_TOKENS // RMS_M_TILE) * hidden_blocks,
        name_hint="dspark_hidden_zero",
        deps=[head_ready_tid],
    ) as hidden_zero_tid:
        zero_task = pl.tile.get_block_idx()
        zero_row = (zero_task // hidden_blocks) * RMS_M_TILE
        zero_col = (zero_task % hidden_blocks) * HIDDEN_TILE
        padded_hidden[
            zero_row : zero_row + RMS_M_TILE,
            zero_col : zero_col + HIDDEN_TILE,
        ] = pl.full([RMS_M_TILE, HIDDEN_TILE], dtype=pl.BF16, value=0.0)
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dspark_hidden_pad",
        deps=[hidden_zero_tid],
    ) as hidden_pad_tid:
        for token in pl.range(active_tokens):
            padded_hidden[token : token + 1, :] = hidden_flat[token : token + 1, :]

    # Keep this normalization local. Passing a dynamically sized temporary to
    # the generic rms_norm inline kernel loses its inferred tensor metadata
    # during JIT specialization.
    with pl.spmd(
        padded_tokens // RMS_M_TILE,
        name_hint="dspark_final_norm",
        deps=[hidden_pad_tid],
        allow_early_resolve=True,
    ) as final_norm_tid:
        row_block = pl.tile.get_block_idx()
        row_offset = row_block * RMS_M_TILE
        square_sum = pl.full([1, RMS_M_TILE], dtype=pl.FP32, value=0.0)
        for hidden_block in pl.pipeline(D // HIDDEN_TILE, stage=2):
            hidden_offset = hidden_block * HIDDEN_TILE
            rms_hidden_tile = pl.cast(
                padded_hidden[
                    row_offset : row_offset + RMS_M_TILE,
                    hidden_offset : hidden_offset + HIDDEN_TILE,
                ],
                target_type=pl.FP32,
            )
            square_sum = pl.add(
                square_sum,
                pl.reshape(
                    pl.row_sum(pl.mul(rms_hidden_tile, rms_hidden_tile)),
                    [1, RMS_M_TILE],
                ),
            )
        inv_rms = pl.reshape(
            pl.rsqrt(
                pl.add(pl.mul(square_sum, 1.0 / D), EPS),
                high_precision=True,
            ),
            [RMS_M_TILE, 1],
        )
        for hidden_block in pl.pipeline(D // HIDDEN_TILE, stage=2):
            hidden_offset = hidden_block * HIDDEN_TILE
            rms_hidden_tile = pl.cast(
                padded_hidden[
                    row_offset : row_offset + RMS_M_TILE,
                    hidden_offset : hidden_offset + HIDDEN_TILE,
                ],
                target_type=pl.FP32,
            )
            norm_tile = pl.cast(
                pl.reshape(
                    final_norm_weight[hidden_offset : hidden_offset + HIDDEN_TILE],
                    [1, HIDDEN_TILE],
                ),
                target_type=pl.FP32,
            )
            normalized[
                row_offset : row_offset + RMS_M_TILE,
                hidden_offset : hidden_offset + HIDDEN_TILE,
            ] = pl.cast(
                pl.col_expand_mul(
                    pl.row_expand_mul(rms_hidden_tile, inv_rms),
                    norm_tile,
                ),
                target_type=pl.BF16,
                mode="rint",
            )
    return final_norm_tid


@pl.jit.inline
def greedy_markov_step(
    head_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16],
    base_logits: pl.Tensor,
    num_sampled: pl.Tensor[[B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[B_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    confidence_head_weight: pl.Tensor[[1, D + DSPARK_MARKOV_RANK], pl.FP32],
    draft_token_scratch: pl.Tensor[[MARKOV_M_TILE, MARKOV_ID_PAD], pl.INT32],
    confidence_scratch: pl.Tensor[[DSPARK_QUERY_WIDTH, MARKOV_M_TILE], pl.FP32],
    base_logits_ready_tid: pl.Scalar[pl.TASK_ID],
    start_tid: pl.Scalar[pl.TASK_ID],
    step: pl.Scalar[pl.INT32],
):
    batch = pl.tensor.dim(num_sampled, 0)
    previous_token_ids = pl.create_tensor([batch], dtype=pl.INT64)
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dspark_markov_previous_tokens",
        deps=[start_tid],
    ) as previous_tokens_tid:
        for request in pl.range(batch):
            previous_token = pl.cast(0, pl.INT64)
            if step == 0:
                previous_token = pl.read(next_prefill_tokens, [request])
                if pl.read(num_sampled, [request]) > 0:
                    previous_token = pl.read(last_sampled, [request])
            if step > 0:
                previous_token = pl.cast(
                    pl.read(draft_token_scratch, [request, step - 1]),
                    pl.INT64,
                )
            pl.write(previous_token_ids, [request], previous_token)

    markov_bias = pl.create_tensor([batch, VOCAB], dtype=pl.FP32)
    markov_embedding = pl.create_tensor([batch, DSPARK_MARKOV_RANK], dtype=pl.BF16)
    (
        markov_bias,
        markov_embedding,
        markov_embedding_tid,
        markov_logits_tid,
    ) = markov_head(
        previous_token_ids,
        markov_w1,
        markov_w2,
        markov_bias,
        markov_embedding,
        previous_tokens_tid,
    )

    confidence_blocks = (batch + CONFIDENCE_PAD - 1) // CONFIDENCE_PAD
    with pl.spmd(
        confidence_blocks,
        name_hint="dspark_confidence_head",
        deps=[markov_embedding_tid],
    ) as confidence_tid:
        # Weight reads bypass L2.
        pl.set_cache_policy(confidence_head_weight, pl.CachePolicy.BYPASS)
        confidence_block = pl.tile.get_block_idx()
        request_base = confidence_block * CONFIDENCE_PAD
        valid_rows = pl.min(CONFIDENCE_PAD, batch - request_base)
        confidence_logit = pl.full(
            [1, CONFIDENCE_PAD],
            dtype=pl.FP32,
            value=0.0,
        )
        for hidden_block in pl.range(D // HIDDEN_TILE):
            hidden_offset = hidden_block * HIDDEN_TILE
            hidden_bf16 = pl.reshape(
                pl.slice(
                    head_hidden,
                    [CONFIDENCE_PAD, 1, HIDDEN_TILE],
                    [request_base, step, hidden_offset],
                    valid_shape=[valid_rows, 1, HIDDEN_TILE],
                ),
                [CONFIDENCE_PAD, HIDDEN_TILE],
            )
            hidden_fp32 = pl.cast(hidden_bf16, target_type=pl.FP32)
            hidden_weight = confidence_head_weight[
                0:1,
                hidden_offset : hidden_offset + HIDDEN_TILE,
            ]
            confidence_logit = pl.add(
                confidence_logit,
                pl.reshape(
                    pl.row_sum(pl.col_expand_mul(hidden_fp32, hidden_weight)),
                    [1, CONFIDENCE_PAD],
                ),
            )
        markov_bf16 = pl.slice(
            markov_embedding,
            [CONFIDENCE_PAD, DSPARK_MARKOV_RANK],
            [request_base, 0],
            valid_shape=[valid_rows, DSPARK_MARKOV_RANK],
        )
        markov_fp32 = pl.cast(markov_bf16, target_type=pl.FP32)
        markov_weight = confidence_head_weight[
            0:1,
            D : D + DSPARK_MARKOV_RANK,
        ]
        confidence_logit = pl.add(
            confidence_logit,
            pl.reshape(
                pl.row_sum(pl.col_expand_mul(markov_fp32, markov_weight)),
                [1, CONFIDENCE_PAD],
            ),
        )
        confidence_prob = pl.recip(pl.add(pl.exp(pl.neg(confidence_logit)), 1.0))
        confidence_scratch[
            step : step + 1,
            request_base : request_base + CONFIDENCE_PAD,
        ] = confidence_prob

    partial_scores = pl.create_tensor([batch, GREEDY_PARTITIONS, MARKOV_ID_PAD], dtype=pl.FP32)
    partial_ids = pl.create_tensor([batch, GREEDY_PARTITIONS, MARKOV_ID_PAD], dtype=pl.INT32)
    with pl.spmd(
        batch * GREEDY_PARTITIONS,
        name_hint="greedy_vocab_partitions",
        deps=[base_logits_ready_tid, confidence_tid, markov_logits_tid],
    ) as partial_tid:
        block = pl.tile.get_block_idx()
        request = block // GREEDY_PARTITIONS
        partition = block % GREEDY_PARTITIONS
        source_row = request * DSPARK_QUERY_WIDTH + step
        chunk_maxima = pl.full([GREEDY_ARGMAX_ROWS, GREEDY_CHUNK_PAD], dtype=pl.FP32, value=NEG_INF)
        chunk_token_ids = pl.full([1, GREEDY_CHUNK_PAD], dtype=pl.INT32, value=0)
        for local_chunk in pl.range(GREEDY_PARTITION_TILE):
            chunk = partition * GREEDY_PARTITION_TILE + local_chunk
            if chunk < GREEDY_NUM_CHUNKS:
                vocab_offset = chunk * GREEDY_VOCAB_CHUNK
                scores = pl.full([GREEDY_ARGMAX_ROWS, GREEDY_VOCAB_CHUNK], dtype=pl.FP32, value=NEG_INF)
                base = pl.slice(base_logits, [1, GREEDY_VOCAB_CHUNK], [source_row, vocab_offset])
                bias = markov_bias[request : request + 1, vocab_offset : vocab_offset + GREEDY_VOCAB_CHUNK]
                scores[0:1, 0:GREEDY_VOCAB_CHUNK] = pl.add(base, bias)
                local_winner = pl.row_argmax(scores)
                local_token = pl.read(local_winner, [0, 0])
                local_value = pl.read(scores, [0, pl.cast(local_token, pl.INDEX)])
                pl.write(chunk_maxima, [0, local_chunk], local_value)
                pl.write(chunk_token_ids, [0, local_chunk], pl.cast(vocab_offset, pl.INT32) + local_token)
        winning_chunk = pl.read(pl.row_argmax(chunk_maxima), [0, 0])
        winner_col = pl.cast(winning_chunk, pl.INDEX)
        winning_score = pl.read(chunk_maxima, [0, winner_col])
        winning_token = pl.read(chunk_token_ids, [0, winner_col])
        pl.write(partial_scores, [request, partition, 0], winning_score)
        pl.write(partial_ids, [request, partition, 0], winning_token)
    with pl.spmd(MARKOV_M_TILE, name_hint="greedy_partition_reduce", deps=[partial_tid]) as greedy_tid:
        request = pl.tile.get_block_idx()
        if request < batch:
            maxima = pl.full([GREEDY_ARGMAX_ROWS, GREEDY_CHUNK_PAD], dtype=pl.FP32, value=NEG_INF)
            ids = pl.full([1, GREEDY_CHUNK_PAD], dtype=pl.INT32, value=0)
            for partition in pl.range(GREEDY_PARTITIONS):
                pl.write(maxima, [0, partition], pl.read(partial_scores, [request, partition, 0]))
                pl.write(ids, [0, partition], pl.read(partial_ids, [request, partition, 0]))
            winning_partition = pl.read(pl.row_argmax(maxima), [0, 0])
            winning_token = pl.read(ids, [0, pl.cast(winning_partition, pl.INDEX)])
            token_row = draft_token_scratch[request : request + 1, 0:MARKOV_ID_PAD]
            if step == 0:
                pl.write(token_row, [0, 0], winning_token)
            if step == 1:
                pl.write(token_row, [0, 1], winning_token)
            if step == 2:
                pl.write(token_row, [0, 2], winning_token)
            if step == 3:
                pl.write(token_row, [0, 3], winning_token)
            if step == 4:
                pl.write(token_row, [0, 4], winning_token)
            if step == 5:
                pl.write(token_row, [0, 5], winning_token)
            if step == 6:
                pl.write(token_row, [0, 6], winning_token)
            draft_token_scratch[request : request + 1, 0:MARKOV_ID_PAD] = token_row

    return greedy_tid


@pl.jit.inline
def sample_from_base_logits(
    head_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16],
    base_logits: pl.Tensor,
    num_sampled: pl.Tensor[[B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[B_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    confidence_head_weight: pl.Tensor[[1, D + DSPARK_MARKOV_RANK], pl.FP32],
    draft_token_ids: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.INT32],
    confidence_probs: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.FP32],
    base_logits_ready_tid: pl.Scalar[pl.TASK_ID],
    output_owner_tp_rank: pl.Scalar[pl.INT32],
):
    batch = pl.tensor.dim(num_sampled, 0)
    draft_token_scratch = pl.create_tensor(
        [MARKOV_M_TILE, MARKOV_ID_PAD],
        dtype=pl.INT32,
    )
    confidence_scratch = pl.create_tensor(
        [DSPARK_QUERY_WIDTH, MARKOV_M_TILE],
        dtype=pl.FP32,
    )
    with pl.spmd(
        batch,
        name_hint="dspark_markov_token_scratch_zero",
    ) as token_scratch_zero_tid:
        request = pl.tile.get_block_idx()
        draft_token_scratch[
            request : request + 1,
            0:MARKOV_ID_PAD,
        ] = pl.full([1, MARKOV_ID_PAD], dtype=pl.INT32, value=0)
    step_0_tid = greedy_markov_step(
        head_hidden,
        base_logits,
        num_sampled,
        last_sampled,
        next_prefill_tokens,
        markov_w1,
        markov_w2,
        confidence_head_weight,
        draft_token_scratch,
        confidence_scratch,
        base_logits_ready_tid,
        token_scratch_zero_tid,
        pl.cast(0, pl.INT32),
    )
    step_1_tid = greedy_markov_step(
        head_hidden,
        base_logits,
        num_sampled,
        last_sampled,
        next_prefill_tokens,
        markov_w1,
        markov_w2,
        confidence_head_weight,
        draft_token_scratch,
        confidence_scratch,
        base_logits_ready_tid,
        step_0_tid,
        pl.cast(1, pl.INT32),
    )
    step_2_tid = greedy_markov_step(
        head_hidden,
        base_logits,
        num_sampled,
        last_sampled,
        next_prefill_tokens,
        markov_w1,
        markov_w2,
        confidence_head_weight,
        draft_token_scratch,
        confidence_scratch,
        base_logits_ready_tid,
        step_1_tid,
        pl.cast(2, pl.INT32),
    )
    step_3_tid = greedy_markov_step(
        head_hidden,
        base_logits,
        num_sampled,
        last_sampled,
        next_prefill_tokens,
        markov_w1,
        markov_w2,
        confidence_head_weight,
        draft_token_scratch,
        confidence_scratch,
        base_logits_ready_tid,
        step_2_tid,
        pl.cast(3, pl.INT32),
    )
    step_4_tid = greedy_markov_step(
        head_hidden,
        base_logits,
        num_sampled,
        last_sampled,
        next_prefill_tokens,
        markov_w1,
        markov_w2,
        confidence_head_weight,
        draft_token_scratch,
        confidence_scratch,
        base_logits_ready_tid,
        step_3_tid,
        pl.cast(4, pl.INT32),
    )
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dspark_markov_token_output_copy",
        deps=[step_4_tid],
    ) as token_output_ready_tid:
        if output_owner_tp_rank == 0:
            for request in pl.range(batch):
                for output_step in pl.range(DSPARK_QUERY_WIDTH):
                    pl.write(
                        draft_token_ids,
                        [request, output_step],
                        pl.read(draft_token_scratch, [request, output_step]),
                    )
                    pl.write(
                        confidence_probs,
                        [request, output_step],
                        pl.read(confidence_scratch, [output_step, request]),
                    )
        else:
            for request in pl.range(batch):
                for output_step in pl.range(DSPARK_QUERY_WIDTH):
                    pl.write(draft_token_ids, [request, output_step], pl.cast(0, pl.INT32))
                    pl.write(confidence_probs, [request, output_step], pl.cast(0.0, pl.FP32))
    return draft_token_ids, confidence_probs, token_output_ready_tid


@pl.jit.inline(auto_scope=False)
def markov_inline(
    head_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16, pl.NZ],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    num_sampled: pl.Tensor[[B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[B_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    confidence_head_weight: pl.Tensor[[1, D + DSPARK_MARKOV_RANK], pl.FP32],
    draft_token_ids: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.INT32],
    confidence_probs: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.FP32],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    head_ready_tid: pl.Scalar[pl.TASK_ID],
):
    head_hidden.bind_dynamic(0, B_DYN)
    num_sampled.bind_dynamic(0, B_DYN)
    last_sampled.bind_dynamic(0, B_DYN)
    next_prefill_tokens.bind_dynamic(0, B_DYN)
    draft_token_ids.bind_dynamic(0, B_DYN)
    confidence_probs.bind_dynamic(0, B_DYN)
    normalized = pl.create_tensor([DSPARK_MOE_TOKENS, D], dtype=pl.BF16)
    final_norm_tid = normalize_head_hidden(
        head_hidden,
        final_norm_weight,
        normalized,
        head_ready_tid,
    )
    base_logits = pl.create_tensor([MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
    base_logits, base_logits_tid = lm_head(
        normalized,
        lm_head_weight,
        logit_row_indices,
        base_logits,
        hidden_window,
        hidden_done,
        logits_window,
        logits_done,
        group_base,
        tp_rank,
        DONE_VALUE,
        final_norm_tid,
    )
    draft_token_ids, confidence_probs, token_output_ready_tid = sample_from_base_logits(
        head_hidden,
        base_logits,
        num_sampled,
        last_sampled,
        next_prefill_tokens,
        markov_w1,
        markov_w2,
        confidence_head_weight,
        draft_token_ids,
        confidence_probs,
        base_logits_tid,
        tp_rank,
    )
    return draft_token_ids, confidence_probs, token_output_ready_tid
