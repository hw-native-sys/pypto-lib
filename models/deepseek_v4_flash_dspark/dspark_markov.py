# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compose the target LM head with sequential Markov sampling and confidence."""

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig

from config import FLASH as M
from lm_head import (
    DONE_VALUE,
    GROUP_LOGIT_ROWS,
    MAX_LOGIT_ROWS,
    TP_SIZE,
    VOCAB_PER_TP,
    WORLD_SIZE,
    lm_head,
)
from markov_head import markov_head


B_DYN = pl.dynamic("DSPARK_MARKOV_B_DYN")

# DSpark Markov program contract.
DSPARK_MARKOV_RANK = 256
DSPARK_QUERY_WIDTH = 7
DSPARK_QUERY_PAD = 8
DSPARK_SUPPORTED_BATCHES = (4, 8, 12, 16)
DSPARK_MAX_BATCH = max(DSPARK_SUPPORTED_BATCHES)
DSPARK_MOE_TOKENS = DSPARK_MAX_BATCH * DSPARK_QUERY_PAD

assert DSPARK_QUERY_WIDTH < DSPARK_QUERY_PAD

# model config
D = M.hidden_size
VOCAB = M.vocab_size
EPS = M.rms_norm_eps

# tiling
HIDDEN_TILE = 512
RMS_M_TILE = 8
LM_M_TILE = 16
LM_N_TILE = 128
LM_K_TILE = 256
MARKOV_M_TILE = DSPARK_MAX_BATCH
MARKOV_ID_PAD = 8
CONFIDENCE_PAD = 8
GREEDY_VOCAB_CHUNK = 256
GREEDY_NUM_CHUNKS = VOCAB // GREEDY_VOCAB_CHUNK
GREEDY_CHUNK_PAD = 512
GREEDY_ARGMAX_ROWS = 8
NEG_INF = -3.402823e38

assert D % HIDDEN_TILE == 0
assert D % LM_K_TILE == 0
assert VOCAB % LM_N_TILE == 0
assert VOCAB % GREEDY_VOCAB_CHUNK == 0
assert GREEDY_NUM_CHUNKS <= GREEDY_CHUNK_PAD
assert CONFIDENCE_PAD * 4 == 32
# Base-logit row tiling must exactly cover every supported padded batch.
for _batch in DSPARK_SUPPORTED_BATCHES:
    assert (_batch * DSPARK_QUERY_PAD) % LM_M_TILE == 0


@pl.jit.inline
def normalize_head_hidden(
    head_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    normalized: pl.Tensor[[DSPARK_MOE_TOKENS, D], pl.BF16],
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
                    final_norm_weight[
                        hidden_offset : hidden_offset + HIDDEN_TILE
                    ],
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
def compute_base_logits(
    head_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB, D], pl.BF16],
    base_logits: pl.Tensor[[DSPARK_MOE_TOKENS, VOCAB], pl.FP32],
):
    batch = pl.tensor.dim(head_hidden, 0)
    padded_tokens = batch * DSPARK_QUERY_PAD
    normalized = pl.create_tensor([DSPARK_MOE_TOKENS, D], dtype=pl.BF16)
    final_norm_tid = normalize_head_hidden(
        head_hidden,
        final_norm_weight,
        normalized,
    )
    row_blocks = padded_tokens // LM_M_TILE
    vocab_blocks = VOCAB // LM_N_TILE
    with pl.spmd(
        row_blocks * vocab_blocks,
        name_hint="dspark_base_logits",
        deps=[final_norm_tid],
    ) as base_logits_tid:
        task = pl.tile.get_block_idx()
        row_block = task // vocab_blocks
        vocab_block = task - row_block * vocab_blocks
        row_offset = row_block * LM_M_TILE
        vocab_offset = vocab_block * LM_N_TILE
        hidden_tile = normalized[
            row_offset : row_offset + LM_M_TILE,
            0:LM_K_TILE,
        ]
        weight_tile = lm_head_weight[
            vocab_offset : vocab_offset + LM_N_TILE,
            0:LM_K_TILE,
        ]
        logits_acc = pl.matmul(
            hidden_tile,
            weight_tile,
            b_trans=True,
            out_dtype=pl.FP32,
        )
        for hidden_offset in pl.pipeline(
            LM_K_TILE,
            D,
            LM_K_TILE,
            stage=2,
        ):
            hidden_tile = normalized[
                row_offset : row_offset + LM_M_TILE,
                hidden_offset : hidden_offset + LM_K_TILE,
            ]
            weight_tile = lm_head_weight[
                vocab_offset : vocab_offset + LM_N_TILE,
                hidden_offset : hidden_offset + LM_K_TILE,
            ]
            logits_acc = pl.matmul_acc(
                logits_acc,
                hidden_tile,
                weight_tile,
                b_trans=True,
            )
        base_logits[
            row_offset : row_offset + LM_M_TILE,
            vocab_offset : vocab_offset + LM_N_TILE,
        ] = logits_acc
    return base_logits_tid


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
    )

    confidence_blocks = (batch + CONFIDENCE_PAD - 1) // CONFIDENCE_PAD
    with pl.spmd(
        confidence_blocks,
        name_hint="dspark_confidence_head",
        deps=[markov_embedding_tid],
    ) as confidence_tid:
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
                    pl.row_sum(
                        pl.col_expand_mul(hidden_fp32, hidden_weight)
                    ),
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
        confidence_prob = pl.recip(
            pl.add(pl.exp(pl.neg(confidence_logit)), 1.0)
        )
        confidence_scratch[
            step : step + 1,
            request_base : request_base + CONFIDENCE_PAD,
        ] = confidence_prob

    with pl.spmd(
        MARKOV_M_TILE,
        name_hint="dspark_markov_greedy",
        deps=[base_logits_ready_tid, confidence_tid, markov_logits_tid],
    ) as greedy_tid:
        request = pl.tile.get_block_idx()
        if request < batch:
            source_row = request * DSPARK_QUERY_WIDTH + step
            chunk_maxima = pl.full(
                [GREEDY_ARGMAX_ROWS, GREEDY_CHUNK_PAD],
                dtype=pl.FP32,
                value=NEG_INF,
            )
            chunk_token_ids = pl.full(
                [1, GREEDY_CHUNK_PAD],
                dtype=pl.INT32,
                value=0,
            )
            for chunk in pl.range(GREEDY_NUM_CHUNKS):
                vocab_offset = chunk * GREEDY_VOCAB_CHUNK
                scores = pl.full(
                    [GREEDY_ARGMAX_ROWS, GREEDY_VOCAB_CHUNK],
                    dtype=pl.FP32,
                    value=NEG_INF,
                )
                scores[0:1, 0:GREEDY_VOCAB_CHUNK] = pl.add(
                    pl.slice(
                        base_logits,
                        [1, GREEDY_VOCAB_CHUNK],
                        [pl.cast(source_row, pl.INDEX), vocab_offset],
                    ),
                    markov_bias[
                        request : request + 1,
                        vocab_offset : vocab_offset + GREEDY_VOCAB_CHUNK,
                    ],
                )
                local_winner = pl.row_argmax(scores)
                local_token = pl.read(local_winner, [0, 0])
                pl.write(
                    chunk_maxima,
                    [0, chunk],
                    pl.read(scores, [0, pl.cast(local_token, pl.INDEX)]),
                )
                pl.write(
                    chunk_token_ids,
                    [0, chunk],
                    pl.cast(vocab_offset, pl.INT32) + local_token,
                )

            winning_chunk = pl.read(pl.row_argmax(chunk_maxima), [0, 0])
            winning_token = pl.read(
                chunk_token_ids,
                [0, pl.cast(winning_chunk, pl.INDEX)],
            )
            token_row = draft_token_scratch[
                request : request + 1,
                0:MARKOV_ID_PAD,
            ]
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
            draft_token_scratch[
                request : request + 1,
                0:MARKOV_ID_PAD,
            ] = token_row

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
        head_hidden, base_logits, num_sampled, last_sampled, next_prefill_tokens,
        markov_w1, markov_w2, confidence_head_weight,
        draft_token_scratch, confidence_scratch,
        base_logits_ready_tid, token_scratch_zero_tid, pl.cast(0, pl.INT32)
    )
    step_1_tid = greedy_markov_step(
        head_hidden, base_logits, num_sampled, last_sampled, next_prefill_tokens,
        markov_w1, markov_w2, confidence_head_weight,
        draft_token_scratch, confidence_scratch,
        base_logits_ready_tid, step_0_tid, pl.cast(1, pl.INT32)
    )
    step_2_tid = greedy_markov_step(
        head_hidden, base_logits, num_sampled, last_sampled, next_prefill_tokens,
        markov_w1, markov_w2, confidence_head_weight,
        draft_token_scratch, confidence_scratch,
        base_logits_ready_tid, step_1_tid, pl.cast(2, pl.INT32)
    )
    step_3_tid = greedy_markov_step(
        head_hidden, base_logits, num_sampled, last_sampled, next_prefill_tokens,
        markov_w1, markov_w2, confidence_head_weight,
        draft_token_scratch, confidence_scratch,
        base_logits_ready_tid, step_2_tid, pl.cast(3, pl.INT32)
    )
    step_4_tid = greedy_markov_step(
        head_hidden, base_logits, num_sampled, last_sampled, next_prefill_tokens,
        markov_w1, markov_w2, confidence_head_weight,
        draft_token_scratch, confidence_scratch,
        base_logits_ready_tid, step_3_tid, pl.cast(4, pl.INT32)
    )
    step_5_tid = greedy_markov_step(
        head_hidden, base_logits, num_sampled, last_sampled, next_prefill_tokens,
        markov_w1, markov_w2, confidence_head_weight,
        draft_token_scratch, confidence_scratch,
        base_logits_ready_tid, step_4_tid, pl.cast(5, pl.INT32)
    )
    step_6_tid = greedy_markov_step(
        head_hidden, base_logits, num_sampled, last_sampled, next_prefill_tokens,
        markov_w1, markov_w2, confidence_head_weight,
        draft_token_scratch, confidence_scratch,
        base_logits_ready_tid, step_5_tid, pl.cast(6, pl.INT32)
    )
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dspark_markov_token_output_copy",
        deps=[step_6_tid],
    ):
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
    return draft_token_ids, confidence_probs


@pl.jit
def markov_sample(
    head_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB, D], pl.BF16],
    num_sampled: pl.Tensor[[B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[B_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    confidence_head_weight: pl.Tensor[[1, D + DSPARK_MARKOV_RANK], pl.FP32],
    draft_token_ids: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.INT32]],
    confidence_probs: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.FP32]],
):
    head_hidden.bind_dynamic(0, B_DYN)
    num_sampled.bind_dynamic(0, B_DYN)
    last_sampled.bind_dynamic(0, B_DYN)
    next_prefill_tokens.bind_dynamic(0, B_DYN)
    draft_token_ids.bind_dynamic(0, B_DYN)
    confidence_probs.bind_dynamic(0, B_DYN)
    base_logits = pl.create_tensor(
        [DSPARK_MOE_TOKENS, VOCAB],
        dtype=pl.FP32,
    )
    base_logits_tid = compute_base_logits(
        head_hidden,
        final_norm_weight,
        lm_head_weight,
        base_logits,
    )
    return sample_from_base_logits(
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
    )


@pl.jit
def l2_distributed_markov_sample(
    head_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    num_sampled: pl.Tensor[[B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[B_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    confidence_head_weight: pl.Tensor[[1, D + DSPARK_MARKOV_RANK], pl.FP32],
    draft_token_ids: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.INT32]],
    confidence_probs: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.FP32]],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS * VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
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
    return sample_from_base_logits(
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
    )


@pl.jit.host
def l3_distributed_markov_sample(
    head_hidden: pl.Tensor[[WORLD_SIZE, B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16],
    final_norm_weight: pl.Tensor[[WORLD_SIZE, D], pl.BF16],
    lm_head_weight: pl.Tensor[[WORLD_SIZE, VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS], pl.INT32],
    num_sampled: pl.Tensor[[WORLD_SIZE, B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[WORLD_SIZE, B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[WORLD_SIZE, B_DYN], pl.INT64],
    markov_w1: pl.Tensor[[WORLD_SIZE, VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[WORLD_SIZE, VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    confidence_head_weight: pl.Tensor[
        [WORLD_SIZE, 1, D + DSPARK_MARKOV_RANK], pl.FP32
    ],
    draft_token_ids: pl.Out[
        pl.Tensor[[WORLD_SIZE, B_DYN, DSPARK_QUERY_WIDTH], pl.INT32]
    ],
    confidence_probs: pl.Out[
        pl.Tensor[[WORLD_SIZE, B_DYN, DSPARK_QUERY_WIDTH], pl.FP32]
    ],
):
    head_hidden.bind_dynamic(1, B_DYN)
    num_sampled.bind_dynamic(1, B_DYN)
    last_sampled.bind_dynamic(1, B_DYN)
    next_prefill_tokens.bind_dynamic(1, B_DYN)
    draft_token_ids.bind_dynamic(1, B_DYN)
    confidence_probs.bind_dynamic(1, B_DYN)
    hidden_window_buf = pld.alloc_window_buffer(GROUP_LOGIT_ROWS * D * 2)
    logits_window_buf = pld.alloc_window_buffer(MAX_LOGIT_ROWS * VOCAB * 4)
    hidden_done_buf = pld.alloc_window_buffer(TP_SIZE * 4)
    logits_done_buf = pld.alloc_window_buffer(TP_SIZE * 4)

    for rank in pl.range(pld.world_size()):
        hidden_window = pld.window(
            hidden_window_buf,
            [GROUP_LOGIT_ROWS, D],
            dtype=pl.BF16,
        )
        logits_window = pld.window(
            logits_window_buf,
            [MAX_LOGIT_ROWS * VOCAB],
            dtype=pl.FP32,
        )
        hidden_done = pld.window(hidden_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        logits_done = pld.window(logits_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        l2_distributed_markov_sample(
            head_hidden[rank],
            final_norm_weight[rank],
            lm_head_weight[rank],
            logit_row_indices[rank],
            num_sampled[rank],
            last_sampled[rank],
            next_prefill_tokens[rank],
            markov_w1[rank],
            markov_w2[rank],
            confidence_head_weight[rank],
            draft_token_ids[rank],
            confidence_probs[rank],
            hidden_window,
            hidden_done,
            logits_window,
            logits_done,
            rank // TP_SIZE * TP_SIZE,
            rank % TP_SIZE,
            device=rank,
        )


def build_tensor_specs(batch: int, *, distributed: bool = False):
    """Build a deterministic nonzero Markov validation case."""
    import torch
    from golden import TensorSpec

    if batch not in DSPARK_SUPPORTED_BATCHES:
        raise ValueError(f"unsupported DSpark batch {batch}; expected one of {DSPARK_SUPPORTED_BATCHES}")

    def init_head_hidden():
        hidden = torch.ones(batch, DSPARK_QUERY_WIDTH, D, dtype=torch.bfloat16)
        request_delta = torch.arange(batch, dtype=torch.float32).view(batch, 1) / 64.0
        step_delta = torch.arange(DSPARK_QUERY_WIDTH, dtype=torch.float32).view(1, -1) / 32.0
        hidden[:, :, 0] = (0.25 + request_delta + step_delta).to(torch.bfloat16)
        if distributed:
            # Every TP rank owns a distinct SP request slice. A rank-specific
            # one-hot feature makes the fused SP-gather/TP-vocab path observable.
            rank_hidden = []
            for rank in range(WORLD_SIZE):
                group_hidden = hidden.clone()
                group_hidden[:, :, 1] = 1.0 + (rank // TP_SIZE) / 8.0
                group_hidden[:, :, 2 : 2 + TP_SIZE] = 0.0
                group_hidden[:, :, 2 + rank % TP_SIZE] = 1.0
                rank_hidden.append(group_hidden)
            return torch.stack(rank_hidden)
        return hidden

    def init_lm_head_weight():
        if distributed:
            weight = torch.zeros(
                WORLD_SIZE,
                VOCAB_PER_TP,
                D,
                dtype=torch.bfloat16,
            )
            for rank in range(WORLD_SIZE):
                if rank % TP_SIZE == 0:
                    # Rows 4+owner_tp have no Markov bias. Distinct winners for
                    # the SP owners prove their hidden rows are not duplicated.
                    for owner_tp in range(TP_SIZE):
                        weight[rank, 4 + owner_tp, 2 + owner_tp] = 8.0
            return weight
        weight = torch.zeros(VOCAB, D, dtype=torch.bfloat16)
        weight[0, :LM_K_TILE] = 1.0 / LM_K_TILE
        return weight

    def init_last_sampled():
        return torch.arange(batch, dtype=torch.int64) % 2 + 2

    def init_next_prefill_tokens():
        return 3 - torch.arange(batch, dtype=torch.int64) % 2

    def init_markov_w1():
        weight = torch.zeros(VOCAB, DSPARK_MARKOV_RANK, dtype=torch.bfloat16)
        weight[0, 0] = 1.0
        weight[1, 0] = -1.0
        weight[2, 0] = 1.0
        weight[3, 0] = -1.0
        return weight

    def init_markov_w2():
        weight = torch.zeros(VOCAB, DSPARK_MARKOV_RANK, dtype=torch.bfloat16)
        weight[0, 0] = -4.0
        weight[1, 0] = 4.0
        return weight

    def init_confidence_head_weight():
        weight = torch.zeros(1, D + DSPARK_MARKOV_RANK, dtype=torch.float32)
        weight[0, 0] = 0.75
        weight[0, 1] = -0.25
        weight[0, D] = 0.5
        return weight

    def replicate(init_fn):
        value = init_fn()
        if distributed:
            return torch.stack([value] * WORLD_SIZE)
        return value

    def with_world(*shape):
        if distributed:
            return [WORLD_SIZE, *shape]
        return list(shape)

    specs = [
        TensorSpec(
            "head_hidden",
            with_world(batch, DSPARK_QUERY_WIDTH, D),
            torch.bfloat16,
            init_value=init_head_hidden,
        ),
        TensorSpec(
            "final_norm_weight",
            with_world(D),
            torch.bfloat16,
            init_value=lambda: replicate(
                lambda: torch.ones(D, dtype=torch.bfloat16)
            ),
        ),
        TensorSpec(
            "lm_head_weight",
            [WORLD_SIZE, VOCAB_PER_TP, D] if distributed else [VOCAB, D],
            torch.bfloat16,
            init_value=init_lm_head_weight,
            **({"resident": "stacked"} if distributed else {}),
        ),
    ]
    if distributed:
        active_tokens = batch * DSPARK_QUERY_WIDTH

        def init_logit_row_indices():
            indices = torch.full(
                (WORLD_SIZE, MAX_LOGIT_ROWS),
                -1,
                dtype=torch.int32,
            )
            indices[:, :active_tokens] = torch.arange(
                active_tokens,
                dtype=torch.int32,
            )
            return indices

        specs.append(
            TensorSpec(
                "logit_row_indices",
                [WORLD_SIZE, MAX_LOGIT_ROWS],
                torch.int32,
                init_value=init_logit_row_indices,
            )
        )

    specs.extend([
        TensorSpec(
            "num_sampled",
            with_world(batch),
            torch.int32,
            init_value=lambda: replicate(
                lambda: torch.arange(batch, dtype=torch.int32) % 2
            ),
        ),
        TensorSpec(
            "last_sampled",
            with_world(batch),
            torch.int64,
            init_value=lambda: replicate(init_last_sampled),
        ),
        TensorSpec(
            "next_prefill_tokens",
            with_world(batch),
            torch.int64,
            init_value=lambda: replicate(init_next_prefill_tokens),
        ),
        TensorSpec(
            "markov_w1",
            with_world(VOCAB, DSPARK_MARKOV_RANK),
            torch.bfloat16,
            init_value=lambda: replicate(init_markov_w1),
            **({"resident": "stacked"} if distributed else {}),
        ),
        TensorSpec(
            "markov_w2",
            with_world(VOCAB, DSPARK_MARKOV_RANK),
            torch.bfloat16,
            init_value=lambda: replicate(init_markov_w2),
            **({"resident": "stacked"} if distributed else {}),
        ),
        TensorSpec(
            "confidence_head_weight",
            with_world(1, D + DSPARK_MARKOV_RANK),
            torch.float32,
            init_value=lambda: replicate(init_confidence_head_weight),
            **({"resident": "stacked"} if distributed else {}),
        ),
        TensorSpec(
            "draft_token_ids",
            with_world(batch, DSPARK_QUERY_WIDTH),
            torch.int32,
        ),
        TensorSpec(
            "confidence_probs",
            with_world(batch, DSPARK_QUERY_WIDTH),
            torch.float32,
        ),
    ])
    return specs


def golden_nonzero_markov(tensors):
    """Validate the complete nonzero support and sequential Markov chain."""
    import torch

    hidden_fp32 = tensors["head_hidden"].float()
    inv_rms = torch.rsqrt(hidden_fp32.square().mean(dim=-1, keepdim=True) + EPS)
    normalized = (
        hidden_fp32 * inv_rms * tensors["final_norm_weight"].float()
    ).to(torch.bfloat16)
    # The deterministic fixture has nonzero LM-head and Markov rows only in
    # this leading support; all remaining vocabulary scores are exactly zero.
    validation_support = 8
    base_logits = normalized.float().matmul(
        tensors["lm_head_weight"][:validation_support].float().t()
    )

    previous = torch.where(
        tensors["num_sampled"] > 0,
        tensors["last_sampled"],
        tensors["next_prefill_tokens"],
    ).long()
    for step in range(DSPARK_QUERY_WIDTH):
        embedding = tensors["markov_w1"].float().index_select(0, previous)
        confidence_input = torch.cat(
            [hidden_fp32[:, step], embedding],
            dim=-1,
        )
        confidence_logits = confidence_input.matmul(
            tensors["confidence_head_weight"].float().t()
        ).squeeze(-1)
        tensors["confidence_probs"][:, step] = torch.sigmoid(confidence_logits)
        markov_bias = embedding.matmul(
            tensors["markov_w2"][:validation_support].float().t()
        )
        scores = base_logits[:, step] + markov_bias
        assert torch.all(scores.max(dim=-1).values > 0)
        previous = torch.argmax(scores, dim=-1)
        tensors["draft_token_ids"][:, step] = previous.to(torch.int32)


def golden_distributed_markov(tensors):
    """Apply the single-rank golden to every DP-owned TP-group result."""
    import torch

    for rank in range(WORLD_SIZE):
        group_base = rank // TP_SIZE * TP_SIZE
        full_lm_head_weight = torch.cat(
            [
                tensors["lm_head_weight"][group_base + tp_rank]
                for tp_rank in range(TP_SIZE)
            ],
            dim=0,
        )
        rank_tensors = {
            "head_hidden": tensors["head_hidden"][rank],
            "final_norm_weight": tensors["final_norm_weight"][rank],
            "lm_head_weight": full_lm_head_weight,
            "num_sampled": tensors["num_sampled"][rank],
            "last_sampled": tensors["last_sampled"][rank],
            "next_prefill_tokens": tensors["next_prefill_tokens"][rank],
            "markov_w1": tensors["markov_w1"][rank],
            "markov_w2": tensors["markov_w2"][rank],
            "confidence_head_weight": tensors["confidence_head_weight"][rank],
            "draft_token_ids": tensors["draft_token_ids"][rank],
            "confidence_probs": tensors["confidence_probs"][rank],
        }
        golden_nonzero_markov(rank_tensors)


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser(description="Validate the DeepSeek V4 DSpark Markov sampler.")
    parser.add_argument("--batch", type=int, choices=DSPARK_SUPPORTED_BATCHES, default=4)
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=[2, 4, 8, 16])
    parser.add_argument("--dp", type=int, default=1, choices=[1, 2, 4, 8, 16])
    parser.add_argument("-d", "--device", default="0")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--dump-passes", action="store_true")
    args = parser.parse_args()

    assert args.tp == TP_SIZE
    assert args.dp * args.tp == WORLD_SIZE
    config = dict(dump_passes=args.dump_passes, platform=args.platform)
    fn = markov_sample
    golden_fn = golden_nonzero_markov
    if args.distributed:
        device_ids = [int(device) for device in args.device.split(",")]
        assert len(device_ids) >= WORLD_SIZE
        fn = l3_distributed_markov_sample
        golden_fn = golden_distributed_markov
        config["distributed_config"] = DistributedConfig(
            device_ids=device_ids[:WORLD_SIZE],
            num_sub_workers=0,
        )
    else:
        config["device_id"] = int(args.device)

    result = run(
        fn=fn,
        specs=build_tensor_specs(args.batch, distributed=args.distributed),
        golden_fn=golden_fn,
        config=config,
        rtol=2e-3,
        atol=2e-3,
        compile_only=args.compile_only,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
