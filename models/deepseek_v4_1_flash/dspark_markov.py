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

import config as C
from config import B_DYN, DSPARK_MARKOV_RANK, DSPARK_QUERY_WIDTH, FLASH
from lm_head import GROUP_LOGIT_ROWS, MAX_LOGIT_ROWS, TP_SIZE as LM_HEAD_TP_SIZE, VOCAB_PER_TP, lm_head
from rmsnorm import rms_norm_tiled


D = FLASH.hidden_size
VOCAB = FLASH.vocab_size


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
    if head_hidden.shape[:2] != (batch, steps) or confidence_weight.shape != (
        1,
        head_hidden.shape[-1] + markov_w1.shape[-1],
    ):
        raise ValueError("confidence weight must cover hidden and Markov dimensions")
    drafts = torch.empty(batch, steps, dtype=anchor_ids.dtype, device=anchor_ids.device)
    confidences = []
    previous = anchor_ids.long()
    for step in range(steps):
        embedding = markov_w1[previous]
        bias = embedding.float() @ markov_w2.float().T
        previous = (base_logits[:, step].float() + bias).argmax(dim=-1)
        drafts[:, step] = previous
        features = torch.cat((head_hidden[:, step].float(), embedding.float()), dim=-1)
        confidences.append(torch.sigmoid(features @ confidence_weight.float().T).squeeze(-1))
    return drafts, torch.stack(confidences, dim=-1)


def golden_markov_from_state(
    base_logits: torch.Tensor,
    num_sampled: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    head_hidden: torch.Tensor,
    confidence_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        num_sampled.ndim != 1
        or last_sampled.shape != num_sampled.shape
        or next_prefill_tokens.shape != num_sampled.shape
    ):
        raise ValueError("draft state must have one entry per request")
    anchor_ids = torch.where(num_sampled > 0, last_sampled, next_prefill_tokens)
    return golden_markov_sample(base_logits, anchor_ids, markov_w1, markov_w2, head_hidden, confidence_weight)


MARKOV_RANK = DSPARK_MARKOV_RANK
MARKOV_BATCH_TILE = 16
VOCAB_TILE = 128
HIDDEN_TILE = 256
MARKOV_CORES = 48
ARGMAX_ROWS = 8
RESULT_STRIDE = 16

assert VOCAB % VOCAB_TILE == 0
assert D % HIDDEN_TILE == 0
if LM_HEAD_TP_SIZE != C.TP_SIZE:
    raise ValueError(f"DSpark LM head TP={LM_HEAD_TP_SIZE} differs from drafter TP={C.TP_SIZE}")


@pl.jit.inline
def markov_head(
    previous_ids: pl.Tensor[[B_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB, MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB, MARKOV_RANK], pl.BF16],
    embedding: pl.Tensor[[B_DYN, MARKOV_RANK], pl.BF16],
    bias: pl.Tensor[[B_DYN, VOCAB], pl.FP32],
):
    """Evaluate W2[W1[previous_id]] over the complete draft vocabulary."""
    batch = pl.tensor.dim(previous_ids, 0)
    for request in pl.spmd(batch, name_hint="dspark_markov_embedding"):
        pl.set_cache_policy(markov_w1, pl.CachePolicy.BYPASS)
        token = pl.cast(pl.read(previous_ids, [request]), pl.INDEX)
        embedding[request : request + 1, :] = markov_w1[token : token + 1, :]

    padded_batch = (batch + MARKOV_BATCH_TILE - 1) // MARKOV_BATCH_TILE * MARKOV_BATCH_TILE
    work_items = (padded_batch // MARKOV_BATCH_TILE) * (VOCAB // VOCAB_TILE)
    for worker in pl.spmd(MARKOV_CORES, name_hint="dspark_markov_bias"):
        pl.set_cache_policy(markov_w2, pl.CachePolicy.BYPASS)
        for task in pl.range(worker, work_items, MARKOV_CORES):
            request = task // (VOCAB // VOCAB_TILE) * MARKOV_BATCH_TILE
            vocab_start = (task % (VOCAB // VOCAB_TILE)) * VOCAB_TILE
            valid_rows = pl.min(MARKOV_BATCH_TILE, batch - request)
            previous = pl.slice(
                embedding,
                [MARKOV_BATCH_TILE, MARKOV_RANK],
                [request, 0],
                valid_shape=[valid_rows, MARKOV_RANK],
            )
            factors = markov_w2[vocab_start : vocab_start + VOCAB_TILE, :]
            projected = pl.matmul(previous, factors, b_trans=True, out_dtype=pl.FP32)
            bias[request : request + MARKOV_BATCH_TILE, vocab_start : vocab_start + VOCAB_TILE] = (
                pl.set_validshape(projected, valid_rows, VOCAB_TILE)
            )
    return embedding, bias


@pl.jit.inline
def markov_greedy_step(
    base_logits: pl.Tensor[[B_DYN, VOCAB], pl.FP32],
    head_hidden: pl.Tensor[[B_DYN, D], pl.BF16],
    embedding: pl.Tensor[[B_DYN, MARKOV_RANK], pl.BF16],
    bias: pl.Tensor[[B_DYN, VOCAB], pl.FP32],
    confidence_weight: pl.Tensor[[1, D + MARKOV_RANK], pl.FP32],
    next_ids: pl.Tensor[[B_DYN, RESULT_STRIDE], pl.INT32],
    confidence: pl.Tensor[[B_DYN, RESULT_STRIDE], pl.FP32],
):
    """Greedy argmax with stable lowest-id ties and the confidence head."""
    batch = pl.tensor.dim(base_logits, 0)
    for worker in pl.spmd(batch, name_hint="dspark_markov_greedy"):
        for request in pl.range(worker, worker + 1):
            best_score = pl.cast(-3.402823e38, pl.FP32)
            best_id = pl.cast(0, pl.INT32)
            for chunk in pl.range(VOCAB // VOCAB_TILE):
                start = chunk * VOCAB_TILE
                scores = pl.full([ARGMAX_ROWS, VOCAB_TILE], dtype=pl.FP32, value=-3.402823e38)
                scores[0:1, :] = pl.add(
                    base_logits[request : request + 1, start : start + VOCAB_TILE],
                    bias[request : request + 1, start : start + VOCAB_TILE],
                )
                index = pl.read(pl.row_argmax(scores), [0, 0])
                score = pl.read(scores, [0, pl.cast(index, pl.INDEX)])
                if score > best_score:
                    best_score = score
                    best_id = pl.cast(start, pl.INT32) + index
            pl.write(next_ids, [request, 0], best_id)

            logit = pl.full([1, ARGMAX_ROWS], dtype=pl.FP32, value=0.0)
            for offset in pl.range(0, D, HIDDEN_TILE):
                hidden = pl.cast(head_hidden[request : request + 1, offset : offset + HIDDEN_TILE], pl.FP32)
                weight = confidence_weight[0:1, offset : offset + HIDDEN_TILE]
                padded_hidden = pl.full([ARGMAX_ROWS, HIDDEN_TILE], dtype=pl.FP32, value=0.0)
                padded_hidden[0:1, :] = hidden
                logit = pl.add(
                    logit,
                    pl.reshape(pl.row_sum(pl.col_expand_mul(padded_hidden, weight)), [1, ARGMAX_ROWS]),
                )
            markov = pl.cast(embedding[request : request + 1, :], pl.FP32)
            weight = confidence_weight[0:1, D : D + MARKOV_RANK]
            padded_markov = pl.full([ARGMAX_ROWS, MARKOV_RANK], dtype=pl.FP32, value=0.0)
            padded_markov[0:1, :] = markov
            logit = pl.add(
                logit,
                pl.reshape(pl.row_sum(pl.col_expand_mul(padded_markov, weight)), [1, ARGMAX_ROWS]),
            )
            probability = pl.recip(pl.add(pl.exp(pl.neg(logit)), 1.0))
            pl.write(confidence, [request, 0], pl.read(probability, [0, 0]))
    return next_ids, confidence


@pl.jit.inline
def _sample_step(
    base_logits: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, VOCAB], pl.FP32],
    head_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16],
    previous_ids: pl.Tensor[[B_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB, MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB, MARKOV_RANK], pl.BF16],
    confidence_weight: pl.Tensor[[1, D + MARKOV_RANK], pl.FP32],
    draft_ids: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.INT32],
    confidences: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.FP32],
    step: pl.Scalar[pl.INT32],
):
    batch = pl.tensor.dim(previous_ids, 0)
    embedding = pl.create_tensor([batch, MARKOV_RANK], dtype=pl.BF16)
    bias = pl.create_tensor([batch, VOCAB], dtype=pl.FP32)
    next_ids = pl.create_tensor([batch, RESULT_STRIDE], dtype=pl.INT32)
    confidence = pl.create_tensor([batch, RESULT_STRIDE], dtype=pl.FP32)
    markov_head(previous_ids, markov_w1, markov_w2, embedding, bias)
    logits_step = pl.create_tensor([batch, VOCAB], dtype=pl.FP32)
    hidden_step = pl.create_tensor([batch, D], dtype=pl.BF16)
    for worker in pl.spmd(MARKOV_CORES, name_hint="dspark_markov_logits_step"):
        for task in pl.range(worker, batch * (VOCAB // VOCAB_TILE), MARKOV_CORES):
            request = task // (VOCAB // VOCAB_TILE)
            start = (task % (VOCAB // VOCAB_TILE)) * VOCAB_TILE
            logits_tile = pl.slice(base_logits, [1, 1, VOCAB_TILE], [request, step, start])
            logits_step[request : request + 1, start : start + VOCAB_TILE] = pl.reshape(
                logits_tile, [1, VOCAB_TILE]
            )
    for worker in pl.spmd(MARKOV_CORES, name_hint="dspark_markov_hidden_step"):
        for task in pl.range(worker, batch * (D // HIDDEN_TILE), MARKOV_CORES):
            request = task // (D // HIDDEN_TILE)
            start = (task % (D // HIDDEN_TILE)) * HIDDEN_TILE
            hidden_tile = pl.slice(head_hidden, [1, 1, HIDDEN_TILE], [request, step, start])
            hidden_step[request : request + 1, start : start + HIDDEN_TILE] = pl.reshape(
                hidden_tile, [1, HIDDEN_TILE]
            )
    markov_greedy_step(logits_step, hidden_step, embedding, bias, confidence_weight, next_ids, confidence)
    for worker in pl.spmd(1, name_hint="dspark_markov_advance"):
        for request in pl.range(worker, batch):
            token = pl.read(next_ids, [request, 0])
            pl.write(previous_ids, [request], pl.cast(token, pl.INT64))
            pl.write(draft_ids, [request, step], token)
            pl.write(confidences, [request, step], pl.read(confidence, [request, 0]))
    return previous_ids, draft_ids, confidences


def _markov_sample(
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
    previous_ids = pl.create_tensor([batch], dtype=pl.INT64)
    for worker in pl.spmd(1, name_hint="dspark_markov_anchor"):
        for request in pl.range(worker, batch):
            pl.write(previous_ids, [request], pl.read(anchor_ids, [request]))
    _sample_step(
        base_logits,
        head_hidden,
        previous_ids,
        markov_w1,
        markov_w2,
        confidence_weight,
        draft_ids,
        confidences,
        pl.const(0, pl.INT32),
    )
    _sample_step(
        base_logits,
        head_hidden,
        previous_ids,
        markov_w1,
        markov_w2,
        confidence_weight,
        draft_ids,
        confidences,
        pl.const(1, pl.INT32),
    )
    _sample_step(
        base_logits,
        head_hidden,
        previous_ids,
        markov_w1,
        markov_w2,
        confidence_weight,
        draft_ids,
        confidences,
        pl.const(2, pl.INT32),
    )
    _sample_step(
        base_logits,
        head_hidden,
        previous_ids,
        markov_w1,
        markov_w2,
        confidence_weight,
        draft_ids,
        confidences,
        pl.const(3, pl.INT32),
    )
    _sample_step(
        base_logits,
        head_hidden,
        previous_ids,
        markov_w1,
        markov_w2,
        confidence_weight,
        draft_ids,
        confidences,
        pl.const(4, pl.INT32),
    )
    return draft_ids, confidences


markov_sample_inline = pl.jit.inline(auto_scope=False)(_markov_sample)
markov_sample = pl.jit(auto_scope=False)(_markov_sample)


def _markov_sample_from_hidden(
    head_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    num_sampled: pl.Tensor[[B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[B_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB, MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB, MARKOV_RANK], pl.BF16],
    confidence_weight: pl.Tensor[[1, D + MARKOV_RANK], pl.FP32],
    draft_ids: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.INT32]],
    confidences: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.FP32]],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
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
    normalized = pl.create_tensor([rows, D], dtype=pl.BF16)
    for block in pl.spmd((rows + 7) // 8, name_hint="dspark_final_norm"):
        rms_norm_tiled(flat_hidden, final_norm_weight, normalized, block, rows, output_fp32=False)
    base_logits = pl.create_tensor([batch, DSPARK_QUERY_WIDTH, VOCAB], dtype=pl.FP32)
    flat_logits = pl.reshape(base_logits, [rows, VOCAB])
    group_base = rank // C.TP_SIZE * C.TP_SIZE
    tp_rank = rank % C.TP_SIZE
    chunks = (rows + MAX_LOGIT_ROWS - 1) // MAX_LOGIT_ROWS
    completed_epochs = logits_epoch + chunks - 1
    for chunk in pl.range(chunks):
        row_indices = pl.create_tensor([MAX_LOGIT_ROWS], dtype=pl.INT32)
        for row in pl.spmd(MAX_LOGIT_ROWS, name_hint="dspark_logit_rows"):
            source = pl.cast(-1, pl.INT32)
            global_row = chunk * MAX_LOGIT_ROWS + row
            if global_row < rows:
                source = pl.cast(global_row, pl.INT32)
            pl.write(row_indices, [row], source)
        logits = pl.create_tensor([MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
        lm_head(
            normalized,
            lm_head_weight,
            row_indices,
            logits,
            hidden_window,
            hidden_done,
            logits_window,
            logits_done,
            group_base,
            tp_rank,
            logits_epoch + chunk,
            completed_epochs,
        )
        for row in pl.spmd(MAX_LOGIT_ROWS, name_hint="dspark_pack_logits"):
            global_row = chunk * MAX_LOGIT_ROWS + row
            if global_row < rows:
                for offset in pl.range(0, VOCAB, VOCAB_TILE):
                    flat_logits[global_row : global_row + 1, offset : offset + VOCAB_TILE] = logits[
                        row : row + 1, offset : offset + VOCAB_TILE
                    ]
    anchor_ids = pl.create_tensor([batch], dtype=pl.INT64)
    for request in pl.spmd(batch, name_hint="dspark_markov_anchor"):
        anchor = pl.read(next_prefill_tokens, [request])
        if pl.read(num_sampled, [request]) > 0:
            anchor = pl.read(last_sampled, [request])
        pl.write(anchor_ids, [request], anchor)
    markov_sample_inline(
        base_logits,
        head_hidden,
        anchor_ids,
        markov_w1,
        markov_w2,
        confidence_weight,
        draft_ids,
        confidences,
    )
    return draft_ids, confidences


markov_sample_from_hidden = pl.jit.inline(auto_scope=False)(_markov_sample_from_hidden)
l2_markov_sample_from_hidden = pl.jit(auto_scope=False)(_markov_sample_from_hidden)


@pl.jit.host
def l3_markov_sample_from_hidden(
    head_hidden: pl.Tensor[[C.EP_SIZE, B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16],
    final_norm_weight: pl.Tensor[[C.EP_SIZE, D], pl.BF16],
    lm_head_weight: pl.Tensor[[C.EP_SIZE, VOCAB_PER_TP, D], pl.BF16],
    num_sampled: pl.Tensor[[C.EP_SIZE, B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[C.EP_SIZE, B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[C.EP_SIZE, B_DYN], pl.INT64],
    markov_w1: pl.Tensor[[C.EP_SIZE, VOCAB, MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[C.EP_SIZE, VOCAB, MARKOV_RANK], pl.BF16],
    confidence_weight: pl.Tensor[[C.EP_SIZE, 1, D + MARKOV_RANK], pl.FP32],
    draft_ids: pl.Out[pl.Tensor[[C.EP_SIZE, B_DYN, DSPARK_QUERY_WIDTH], pl.INT32]],
    confidences: pl.Out[pl.Tensor[[C.EP_SIZE, B_DYN, DSPARK_QUERY_WIDTH], pl.FP32]],
    logits_epoch: pl.Scalar[pl.INT32],
):
    head_hidden.bind_dynamic(1, B_DYN)
    num_sampled.bind_dynamic(1, B_DYN)
    last_sampled.bind_dynamic(1, B_DYN)
    next_prefill_tokens.bind_dynamic(1, B_DYN)
    draft_ids.bind_dynamic(1, B_DYN)
    confidences.bind_dynamic(1, B_DYN)
    hidden_window_buf = pld.alloc_window_buffer([GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    hidden_done_buf = pld.alloc_window_buffer([C.TP_SIZE, 1], dtype=pl.INT32)
    logits_window_buf = pld.alloc_window_buffer([MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
    logits_done_buf = pld.alloc_window_buffer([C.TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        hidden_window = pld.window(hidden_window_buf, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        hidden_done = pld.window(hidden_done_buf, [C.TP_SIZE, 1], dtype=pl.INT32)
        logits_window = pld.window(logits_window_buf, [MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
        logits_done = pld.window(logits_done_buf, [C.TP_SIZE, 1], dtype=pl.INT32)
        l2_markov_sample_from_hidden(
            head_hidden[rank],
            final_norm_weight[rank],
            lm_head_weight[rank],
            num_sampled[rank],
            last_sampled[rank],
            next_prefill_tokens[rank],
            markov_w1[rank],
            markov_w2[rank],
            confidence_weight[rank],
            draft_ids[rank],
            confidences[rank],
            hidden_window,
            hidden_done,
            logits_window,
            logits_done,
            rank,
            logits_epoch,
            device=rank,
        )


def run_markov_case(
    *,
    platform: str = "a5sim",
    device_id: int = 0,
    compile_only: bool = False,
):
    """Validate the complete five-token Markov recurrence."""
    from golden import TensorSpec, run

    previous = torch.tensor([7], dtype=torch.int64)
    base = torch.zeros(1, DSPARK_QUERY_WIDTH, VOCAB, dtype=torch.float32)
    base[:, :, 13] = 1.0
    hidden = torch.zeros(1, DSPARK_QUERY_WIDTH, D, dtype=torch.bfloat16)
    hidden[:, :, 0] = 1.0
    w1 = torch.zeros(VOCAB, MARKOV_RANK, dtype=torch.bfloat16)
    w1[7, 0] = 2.0
    w2 = torch.zeros_like(w1)
    w2[11, 0] = 3.0
    w2[12, 0] = -3.0
    confidence_weight = torch.zeros(1, D + MARKOV_RANK)
    confidence_weight[0, 0] = 1.0
    confidence_weight[0, D] = 0.5

    def golden(values):
        ids, probabilities = golden_markov_sample(
            values["base_logits"],
            values["anchor_ids"],
            values["markov_w1"],
            values["markov_w2"],
            values["head_hidden"],
            values["confidence_weight"],
        )
        values["draft_ids"].copy_(ids.int())
        values["confidences"].copy_(probabilities)

    def exact_ids(actual, expected, **_kwargs):
        return torch.equal(actual, expected), f"draft IDs: {actual.tolist()} vs {expected.tolist()}"

    specs = [
        TensorSpec("base_logits", list(base.shape), torch.float32, init_value=base),
        TensorSpec("head_hidden", list(hidden.shape), torch.bfloat16, init_value=hidden),
        TensorSpec("anchor_ids", [1], torch.int64, init_value=previous),
        TensorSpec("markov_w1", [VOCAB, MARKOV_RANK], torch.bfloat16, init_value=w1),
        TensorSpec("markov_w2", [VOCAB, MARKOV_RANK], torch.bfloat16, init_value=w2),
        TensorSpec("confidence_weight", [1, D + MARKOV_RANK], torch.float32, init_value=confidence_weight),
        TensorSpec("draft_ids", [1, DSPARK_QUERY_WIDTH], torch.int32),
        TensorSpec("confidences", [1, DSPARK_QUERY_WIDTH], torch.float32),
    ]
    return run(
        fn=markov_sample,
        specs=specs,
        golden_fn=golden,
        compare_fn={"draft_ids": exact_ids},
        config={"platform": platform, "device_id": device_id},
        compile_only=compile_only,
        rtol=1e-3,
        atol=1e-3,
    )


if __name__ == "__" + "main__":
    import argparse

    parser = argparse.ArgumentParser(description="V4.1 DSpark Markov validation")
    parser.add_argument("-p", "--platform", choices=("a5", "a5sim"), default="a5sim")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    result = run_markov_case(
        platform=args.platform,
        device_id=args.device,
        compile_only=args.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(1)
