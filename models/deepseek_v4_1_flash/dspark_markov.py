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

from pathlib import Path

import pypto.language as pl
import pypto.language.distributed as pld
import torch
from pypto.runtime import pto_isa_include_dir

from config import (
    B_DYN,
    DP_SIZE,
    DSPARK_MARKOV_RANK,
    DSPARK_QUERY_WIDTH,
    EP_SIZE,
    FLASH,
    DECODE_BATCH,
    SUPPORTED_EP_SIZES,
    SUPPORTED_LM_HEAD_TP_SIZES,
    LM_HEAD_TP_SIZE,
)
from lm_head import (
    GROUP_LOGIT_ROWS, LM_HEAD_M_TILE, MAX_LOGIT_ROWS,
    VOCAB_PER_TP, lm_head, lm_head_project,
)
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
MARKOV_CORES = 48
RESULT_STRIDE = 16
MAX_LOGIT_CHUNKS = (DECODE_BATCH * DSPARK_QUERY_WIDTH + MAX_LOGIT_ROWS - 1) // MAX_LOGIT_ROWS

if (D, MARKOV_RANK, DSPARK_QUERY_WIDTH) != (5120, 256, 5):
    raise ValueError("native Markov recurrence requires hidden_size=5120, rank=256, and five steps")
if VOCAB % 256 != 0:
    raise ValueError("native Markov recurrence requires complete 256-column vocabulary panels")
if EP_SIZE % LM_HEAD_TP_SIZE:
    raise ValueError("EP must be divisible by LM head TP")


@pl.jit.inline(auto_scope=False)
def _markov_embedding(
    previous_ids: pl.Tensor,
    markov_w1: pl.Tensor,
    embedding: pl.Tensor,
    batch: pl.Scalar[pl.INDEX],
    id_stride: pl.Scalar[pl.INDEX],
    token_ready_tid: pl.Scalar[pl.TASK_ID],
):
    with pl.spmd(batch, name_hint="dspark_markov_embedding", deps=[token_ready_tid]) as embedding_tid:
        request = pl.tile.get_block_idx()
        if request < batch:
            token = pl.cast(pl.read(previous_ids, [request * id_stride]), pl.INDEX)
            embedding[request : request + 1, :] = markov_w1[token : token + 1, :]

    return embedding_tid


@pl.jit.inline
def markov_head(
    previous_ids: pl.Tensor[[B_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB, MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB, MARKOV_RANK], pl.BF16],
    embedding: pl.Out[pl.Tensor[[B_DYN, MARKOV_RANK], pl.BF16]],
    bias: pl.Out[pl.Tensor[[B_DYN, VOCAB], pl.FP32]],
    token_ready_tid: pl.Scalar[pl.TASK_ID] = None,
):
    """Evaluate W2[W1[previous_id]] over the complete draft vocabulary."""
    batch = pl.tensor.dim(previous_ids, 0)
    embedding_tid = _markov_embedding(previous_ids, markov_w1, embedding, batch, pl.const(1, pl.INDEX), token_ready_tid)

    padded_batch = (batch + MARKOV_BATCH_TILE - 1) // MARKOV_BATCH_TILE * MARKOV_BATCH_TILE
    work_items = (padded_batch // MARKOV_BATCH_TILE) * (VOCAB // VOCAB_TILE)
    with pl.spmd(MARKOV_CORES, name_hint="dspark_markov_bias", deps=[embedding_tid]) as bias_tid:
        worker = pl.tile.get_block_idx()
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
    return embedding, bias, embedding_tid, bias_tid


@pl.jit.inline
def markov_head_from_state(
    num_sampled: pl.Tensor[[B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[B_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB, MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB, MARKOV_RANK], pl.BF16],
    embedding: pl.Out[pl.Tensor[[B_DYN, MARKOV_RANK], pl.BF16]],
    bias: pl.Out[pl.Tensor[[B_DYN, VOCAB], pl.FP32]],
):
    """Evaluate the first Markov step directly from caller-owned state."""
    batch = pl.tensor.dim(num_sampled, 0)
    # Keep a deterministic fallback in case a runtime scope has no active
    # worker for a padded request.
    # Keep an explicit DSL tensor so @pl.jit can infer shape and dtype.
    # The per-request loop below initializes every element before the first
    # Markov projection, using the accepted token or prefill fallback.
    previous_ids = pl.create_tensor([batch], dtype=pl.INT64)
    with pl.spmd(1, name_hint="dspark_markov_state_anchor") as anchor_tid:
        # These buffers are produced by the preceding target-accept program;
        # bypass stale per-device cache lines at the cross-program handoff.
        pl.set_cache_policy(num_sampled, pl.CachePolicy.BYPASS)
        pl.set_cache_policy(last_sampled, pl.CachePolicy.BYPASS)
        pl.set_cache_policy(next_prefill_tokens, pl.CachePolicy.BYPASS)
        request_base = pl.tile.get_block_idx()
        for request in pl.range(request_base, batch):
            token = pl.read(next_prefill_tokens, [request])
            if pl.read(num_sampled, [request]) > 0:
                token = pl.read(last_sampled, [request])
            pl.write(previous_ids, [request], pl.cast(token, pl.INT64))
    return markov_head(previous_ids, markov_w1, markov_w2, embedding, bias, anchor_tid)


@pl.jit.extern(core_type="mixed", dual_aiv_dispatch=True,
               aic_source=Path(__file__).parent / "kernels" / "markov_recurrence.cpp",
               aiv_source=Path(__file__).parent / "kernels" / "markov_recurrence.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _markov_recurrence_native(
    base_logits: pl.Tensor,
    embedding: pl.Out[pl.Tensor],
    markov_w2: pl.Tensor,
    partial_scores: pl.Out[pl.Tensor],
    partial_ids: pl.Out[pl.Tensor],
    head_hidden: pl.Tensor,
    markov_w1: pl.Tensor,
    confidence_weight: pl.Tensor,
    draft_state: pl.Out[pl.Tensor],
    confidence_state: pl.Out[pl.Tensor],
    draft_ids: pl.Out[pl.Tensor],
    confidences: pl.Out[pl.Tensor],
    anchor_ids: pl.Tensor,
) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor]:
    return embedding, partial_scores, partial_ids, draft_state, confidence_state, draft_ids, confidences


def _markov_sample(
    base_logits: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, VOCAB], pl.FP32],
    head_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16],
    anchor_ids: pl.Tensor[[B_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB, MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB, MARKOV_RANK], pl.BF16],
    confidence_weight: pl.Tensor[[1, D + MARKOV_RANK], pl.FP32],
    draft_ids: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.INT32]],
    confidences: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.FP32]],
    base_logits_ready_tid: pl.Scalar[pl.TASK_ID] = None,
):
    """Generate five drafts with the previous sampled ID at every step."""
    base_logits.bind_dynamic(0, B_DYN)
    head_hidden.bind_dynamic(0, B_DYN)
    anchor_ids.bind_dynamic(0, B_DYN)
    draft_ids.bind_dynamic(0, B_DYN)
    confidences.bind_dynamic(0, B_DYN)
    batch = pl.tensor.dim(anchor_ids, 0)
    embedding = pl.create_tensor([batch, MARKOV_RANK], dtype=pl.BF16)
    logits_view = pl.slice(base_logits, [batch, DSPARK_QUERY_WIDTH, VOCAB], [0, 0, 0])
    hidden_view = pl.slice(head_hidden, [batch, DSPARK_QUERY_WIDTH, D], [0, 0, 0])
    partial_scores = pl.create_tensor([batch, 32, RESULT_STRIDE], dtype=pl.FP32)
    partial_ids = pl.create_tensor([batch, 32, RESULT_STRIDE], dtype=pl.INT32)
    draft_state = pl.create_tensor([batch, DSPARK_QUERY_WIDTH, RESULT_STRIDE], dtype=pl.INT64)
    confidence_state = pl.create_tensor([batch, DSPARK_QUERY_WIDTH, RESULT_STRIDE], dtype=pl.FP32)
    ids_view = pl.slice(draft_ids, [batch, DSPARK_QUERY_WIDTH], [0, 0])
    conf_view = pl.slice(confidences, [batch, DSPARK_QUERY_WIDTH], [0, 0])
    anchors_view = pl.slice(anchor_ids, [batch], [0])
    workers = pl.min(32, pl.system.available_cluster_count())
    with pl.spmd(workers, name_hint="dspark_markov_recurrence", sync_start=True, deps=[base_logits_ready_tid]):
        embedding, partial_scores, partial_ids, draft_state, confidence_state, published_ids, published_confidences = _markov_recurrence_native(
            logits_view, embedding, markov_w2, partial_scores, partial_ids, hidden_view,
            markov_w1, confidence_weight, draft_state, confidence_state, ids_view, conf_view, anchors_view,
        )
    return published_ids, published_confidences


def _markov_sample_from_state(
    base_logits: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, VOCAB], pl.FP32],
    head_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16],
    num_sampled: pl.Tensor[[B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[B_DYN], pl.INT64],
    anchor_ids: pl.Tensor[[B_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB, MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB, MARKOV_RANK], pl.BF16],
    confidence_weight: pl.Tensor[[1, D + MARKOV_RANK], pl.FP32],
    draft_ids: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.INT32]],
    confidences: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.FP32]],
):
    """Generate drafts with the first anchor read directly from request state."""
    base_logits.bind_dynamic(0, B_DYN)
    head_hidden.bind_dynamic(0, B_DYN)
    num_sampled.bind_dynamic(0, B_DYN)
    last_sampled.bind_dynamic(0, B_DYN)
    next_prefill_tokens.bind_dynamic(0, B_DYN)
    anchor_ids.bind_dynamic(0, B_DYN)
    draft_ids.bind_dynamic(0, B_DYN)
    confidences.bind_dynamic(0, B_DYN)
    batch = pl.tensor.dim(num_sampled, 0)
    previous_ids = pl.create_tensor([batch], dtype=pl.INT64)
    with pl.spmd(1, name_hint="dspark_markov_anchor_ready") as anchor_tid:
        worker = pl.tile.get_block_idx()
        # Resolve the first Markov input from the same state contract as the
        # golden path: decode consumes last_sampled, while prefill has no
        # sampled token and consumes next_prefill_tokens.
        pl.set_cache_policy(num_sampled, pl.CachePolicy.BYPASS)
        pl.set_cache_policy(last_sampled, pl.CachePolicy.BYPASS)
        pl.set_cache_policy(next_prefill_tokens, pl.CachePolicy.BYPASS)
        for request in pl.range(worker, batch):
            token = pl.read(next_prefill_tokens, [request])
            if pl.read(num_sampled, [request]) > 0:
                token = pl.read(last_sampled, [request])
            pl.write(previous_ids, [request], pl.cast(token, pl.INT64))
    markov_sample_inline(
        base_logits, head_hidden, previous_ids, markov_w1, markov_w2, confidence_weight,
        draft_ids, confidences, anchor_tid,
    )
    return draft_ids, confidences


markov_sample_inline = pl.jit.inline(auto_scope=False)(_markov_sample)
markov_sample = pl.jit(auto_scope=False)(_markov_sample)
markov_sample_from_state_inline = pl.jit.inline(auto_scope=False)(_markov_sample_from_state)


@pl.jit(auto_scope=False)
def markov_sample_validation(
    base_logits: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, VOCAB], pl.FP32],
    head_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16],
    anchor_ids: pl.Tensor[[B_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB, MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB, MARKOV_RANK], pl.BF16],
    confidence_weight: pl.Tensor[[1, D + MARKOV_RANK], pl.FP32],
    draft_ids: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.INT32]],
    confidences: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.FP32]],
):
    """Validation entry without the optional cross-program task dependency."""
    ready_tid = pl.system.task_dummy(deps=[])
    return markov_sample_inline(
        base_logits, head_hidden, anchor_ids, markov_w1, markov_w2,
        confidence_weight, draft_ids, confidences, ready_tid,
    )


def _markov_sample_from_hidden(
    head_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16],
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
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
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
    normalized = pl.create_tensor([rows, D], dtype=pl.BF16)
    with pl.spmd((rows + 7) // 8, name_hint="dspark_final_norm") as logits_ready_tid:
        block = pl.tile.get_block_idx()
        rms_norm_tiled(flat_hidden, final_norm_weight, normalized, block, rows, output_fp32=False)
    base_logits = pl.create_tensor([batch, DSPARK_QUERY_WIDTH, VOCAB], dtype=pl.FP32)
    flat_logits = pl.reshape(base_logits, [rows, VOCAB])
    if LM_HEAD_TP_SIZE == 1:
        logits_ready_tid = lm_head_project(normalized, lm_head_weight, flat_logits, LM_HEAD_M_TILE)
        # The caller supplies the already resolved anchor: decode passes the
        # accepted target token, while prefill passes next_prefill_tokens.
        markov_sample_inline(
            base_logits,
            head_hidden,
            anchor_ids,
            markov_w1,
            markov_w2,
            confidence_weight,
            draft_ids,
            confidences,
            logits_ready_tid,
        )
    else:
        group_base = rank // LM_HEAD_TP_SIZE * LM_HEAD_TP_SIZE
        tp_rank = rank % LM_HEAD_TP_SIZE
        # Keep the producer event outside a dynamic pl.range. PTOAS requires
        # values returned from an inline call to remain in one SSA scope; a fixed
        # maximum also lets the final LM-head event gate Markov sampling safely.
        completed_epochs = logits_epoch + MAX_LOGIT_CHUNKS - 1
        for chunk in pl.unroll(MAX_LOGIT_CHUNKS):
            row_indices = pl.create_tensor([MAX_LOGIT_ROWS], dtype=pl.INT32)
            with pl.spmd(1, name_hint="dspark_logit_rows", deps=[logits_ready_tid]):
                core = pl.tile.get_block_idx()
                for row in pl.range(core * MAX_LOGIT_ROWS, (core + 1) * MAX_LOGIT_ROWS):
                    source = pl.cast(-1, pl.INT32)
                    global_row = chunk * MAX_LOGIT_ROWS + row
                    if global_row < rows:
                        source = pl.cast(global_row, pl.INT32)
                    pl.write(row_indices, [row], source)
            logits = pl.create_tensor([MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
            logits, logits_ready_tid = lm_head(
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
            with pl.spmd(MAX_LOGIT_ROWS, name_hint="dspark_pack_logits", deps=[logits_ready_tid]) as logits_ready_tid:
                row = pl.tile.get_block_idx()
                global_row = chunk * MAX_LOGIT_ROWS + row
                if global_row < rows:
                    for offset in pl.range(0, VOCAB, VOCAB_TILE):
                        flat_logits[global_row : global_row + 1, offset : offset + VOCAB_TILE] = logits[
                            row : row + 1, offset : offset + VOCAB_TILE
                        ]
        # The caller supplies the already resolved anchor: decode passes the
        # accepted target token, while prefill passes next_prefill_tokens.
        markov_sample_inline(
            base_logits,
            head_hidden,
            anchor_ids,
            markov_w1,
            markov_w2,
            confidence_weight,
            draft_ids,
            confidences,
            logits_ready_tid,
        )
    return draft_ids, confidences


markov_sample_from_hidden = pl.jit.inline(auto_scope=False)(_markov_sample_from_hidden)
l2_markov_sample_from_hidden = pl.jit(auto_scope=False)(_markov_sample_from_hidden)


@pl.jit.host
def l3_markov_sample_from_hidden(
    head_hidden: pl.Tensor[[EP_SIZE, B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16],
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
    logits_window_buf = pld.alloc_window_buffer([MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
    logits_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        hidden_window = pld.window(hidden_window_buf, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        hidden_done = pld.window(hidden_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        logits_window = pld.window(logits_window_buf, [MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
        logits_done = pld.window(logits_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        l2_markov_sample_from_hidden(
            head_hidden[rank],
            final_norm_weight[rank],
            lm_head_weight[rank],
            num_sampled[rank],
            last_sampled[rank],
            next_prefill_tokens[rank],
            anchor_ids[rank],
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


def run_markov_case(*, platform: str = "a5", device_id: int = 0,
                    compile_only: bool = False):
    """Validate the complete five-token Markov recurrence."""
    if platform != "a5":
        raise ValueError("V4.1 Markov native kernels require A5 device execution; simulator support is unavailable")
    from golden import TensorSpec, run

    batch = 12
    previous = torch.arange(batch, dtype=torch.int64) * (DSPARK_QUERY_WIDTH + 1)
    base = torch.zeros(batch, DSPARK_QUERY_WIDTH, VOCAB, dtype=torch.float32)
    hidden = torch.zeros(batch, DSPARK_QUERY_WIDTH, D, dtype=torch.bfloat16)
    hidden[:, :, 0] = torch.arange(batch, dtype=torch.float32)[:, None] / 4.0 - 1.0
    hidden[:, :, 0] += torch.arange(DSPARK_QUERY_WIDTH, dtype=torch.float32) / 16.0
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
    confidence_weight[0, 0] = 1.0
    confidence_weight[0, D:] = 1.0 / 32.0

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
        TensorSpec("anchor_ids", [batch], torch.int64, init_value=previous),
        TensorSpec("markov_w1", [VOCAB, MARKOV_RANK], torch.bfloat16, init_value=w1),
        TensorSpec("markov_w2", [VOCAB, MARKOV_RANK], torch.bfloat16, init_value=w2),
        TensorSpec("confidence_weight", [1, D + MARKOV_RANK], torch.float32, init_value=confidence_weight),
        TensorSpec("draft_ids", [batch, DSPARK_QUERY_WIDTH], torch.int32),
        TensorSpec("confidences", [batch, DSPARK_QUERY_WIDTH], torch.float32),
    ]
    return run(
        fn=markov_sample_validation,
        specs=specs,
        golden_fn=golden,
        compare_fn={"draft_ids": exact_ids},
        config={"backend_type": "a5", "device_id": device_id,
                "execution_mode": platform == "a5sim"},
        compile_only=compile_only,
        rtol=1e-3,
        atol=1e-3,
    )


if __name__ == "__" + "main__":
    import argparse

    parser = argparse.ArgumentParser(description="V4.1 DSpark Markov validation")
    parser.add_argument("--lm-head-tp", type=int, default=LM_HEAD_TP_SIZE, choices=SUPPORTED_LM_HEAD_TP_SIZES)
    parser.add_argument("--dp", type=int, default=DP_SIZE)
    parser.add_argument("--ep", type=int, default=EP_SIZE, choices=SUPPORTED_EP_SIZES)
    parser.add_argument("-p", "--platform", choices=("a5", "a5sim"), default="a5")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if (args.lm_head_tp, args.dp, args.ep) != (LM_HEAD_TP_SIZE, DP_SIZE, EP_SIZE):
        parser.error("--lm-head-tp/--dp/--ep must match the import-time configuration")
    result = run_markov_case(
        platform=args.platform,
        device_id=args.device,
        compile_only=args.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(1)
