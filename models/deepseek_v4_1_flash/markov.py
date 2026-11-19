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
    B_DYN, DP_SIZE, MOE_TOKENS,
    DSPARK_MARKOV_RANK, DSPARK_QUERY_WIDTH,
    EP_SIZE, FLASH, DECODE_BATCH,
    SUPPORTED_EP_SIZES, SUPPORTED_LM_HEAD_TP_SIZES, LM_HEAD_TP_SIZE,
)
from lm_head import VOCAB_PER_TP, lm_head
from rmsnorm import rms_norm
from hc import hc_head


# model config
D = FLASH.hidden_size
HC_MULT = FLASH.hc_mult
VOCAB = FLASH.vocab_size
MARKOV_RANK = DSPARK_MARKOV_RANK

# Scratch layout and shared logit capacity.
RESULT_STRIDE = 16
LOGIT_ROWS = (DECODE_BATCH * DSPARK_QUERY_WIDTH + 15) // 16 * 16
GROUP_LOGIT_ROWS = LM_HEAD_TP_SIZE * LOGIT_ROWS

if (D, MARKOV_RANK, DSPARK_QUERY_WIDTH) != (5120, 256, 5):
    raise ValueError("native Markov recurrence requires hidden_size=5120, rank=256, and five steps")
if VOCAB % 256 != 0:
    raise ValueError("native Markov recurrence requires complete 256-column vocabulary panels")
if EP_SIZE % LM_HEAD_TP_SIZE:
    raise ValueError("EP must be divisible by LM head TP")


@pl.jit.extern(
    core_type="mixed", dual_aiv_dispatch=True,
    aic_source=Path(__file__).parent / "kernels" / "markov_recurrence.cpp",
    aiv_source=Path(__file__).parent / "kernels" / "markov_recurrence.cpp",
    include_dirs=(pto_isa_include_dir(),),
)
def _markov_sample(
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
    embedding = pl.create_tensor([batch, MARKOV_RANK], dtype=pl.BF16)
    logits_view = base_logits[:batch, :DSPARK_QUERY_WIDTH, :VOCAB]
    hidden_view = head_hidden[:batch, :DSPARK_QUERY_WIDTH, :D]
    partial_scores = pl.create_tensor([batch, 32, RESULT_STRIDE], dtype=pl.FP32)
    partial_ids = pl.create_tensor([batch, 32, RESULT_STRIDE], dtype=pl.INT32)
    draft_state = pl.create_tensor([batch, DSPARK_QUERY_WIDTH, RESULT_STRIDE], dtype=pl.INT64)
    confidence_state = pl.create_tensor([batch, DSPARK_QUERY_WIDTH, RESULT_STRIDE], dtype=pl.FP32)
    ids_view = draft_ids[:batch, :DSPARK_QUERY_WIDTH]
    conf_view = confidences[:batch, :DSPARK_QUERY_WIDTH]
    anchors_view = anchor_ids[:batch]
    workers = pl.min(32, pl.system.available_cluster_count())
    with pl.spmd(workers, name_hint="dspark_markov_recurrence", sync_start=True):
        embedding, partial_scores, partial_ids, draft_state, confidence_state, ids_out, conf_out = _markov_sample(
            logits_view, embedding, markov_w2, partial_scores, partial_ids, hidden_view,
            markov_w1, confidence_weight, draft_state, confidence_state, ids_view, conf_view, anchors_view,
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
