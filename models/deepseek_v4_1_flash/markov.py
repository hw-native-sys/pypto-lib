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
    B_DYN, DP_SIZE, MOE_TOKENS, DECODE_SEQ,
    DSPARK_MARKOV_RANK, DSPARK_QUERY_WIDTH,
    EP_SIZE, FLASH, DECODE_BATCH,
    SUPPORTED_EP_SIZES, SUPPORTED_LM_HEAD_TP_SIZES, LM_HEAD_TP_SIZE,
)
from lm_head import VOCAB_PER_TP, clear_lm_head_signals, lm_head
from rmsnorm import rms_norm
from hc import hc_head
from sample import ARGMAX_WIDTH_TILE, PROPOSAL_STREAM, golden_categorical, golden_dspark_probs, sample_markov_proposal_row


PROPOSAL_PARAMS_DYN = pl.dynamic("MARKOV_PROPOSAL_PARAMS")

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

if (D, MARKOV_RANK, DSPARK_QUERY_WIDTH) != (5120, 256, 5):
    raise ValueError("Markov recurrence requires hidden_size=5120, rank=256, and five steps")
if VOCAB % VOCAB_TILE != 0:
    raise ValueError("Markov recurrence requires complete 256-column vocabulary panels")
if EP_SIZE % LM_HEAD_TP_SIZE:
    raise ValueError("EP must be divisible by LM head TP")


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


@pl.jit.inline(auto_scope=False)
def markov_propose(
    base_logits: pl.Tensor,
    head_hidden: pl.Tensor,
    anchor_ids: pl.Tensor,
    markov_w1: pl.Tensor,
    markov_w2: pl.Tensor,
    confidence_weight: pl.Tensor,
    temperatures: pl.Tensor,
    top_ks: pl.Tensor,
    top_ps: pl.Tensor,
    seeds: pl.Tensor,
    first_positions: pl.Tensor,
    draft_ids: pl.Tensor,
    confidences: pl.Tensor,
    proposal_probs: pl.Tensor,
):
    """Sample each Markov step from its filtered vocabulary distribution."""
    batch = pl.tensor.dim(anchor_ids, 0)
    padded = (batch + 15) // 16 * 16
    hidden = pl.reshape(head_hidden, [batch * DSPARK_QUERY_WIDTH, D])
    base = pl.reshape(base_logits, [batch * DSPARK_QUERY_WIDTH, VOCAB])
    aiv_count = pl.system.available_aiv_count()
    embedding_workers = pl.min(padded, aiv_count)
    request_workers = pl.min(batch, aiv_count)
    previous_selected = pl.create_tensor([batch, 8], dtype=pl.INT32)
    for step in pl.unroll(DSPARK_QUERY_WIDTH):
        embedding = pl.create_tensor([padded, MARKOV_RANK], dtype=pl.BF16)
        for worker in pl.spmd(embedding_workers, name_hint="markov_proposal_embedding"):
            for request in pl.range(worker, padded, embedding_workers):
                previous = pl.cast(0, pl.INT64)
                if request < batch:
                    if step == 0:
                        previous = pl.read(anchor_ids, [request])
                    else:
                        previous = pl.cast(pl.read(previous_selected, [request, 0]), pl.INT64)
                embedding[request : request + 1, :] = markov_w1[previous : previous + 1, :]

        bias = pl.create_tensor([padded, VOCAB], dtype=pl.FP32)
        workers = pl.system.available_cluster_count()
        for worker in pl.spmd(workers, name_hint="markov_proposal_bias"):
            for panel in pl.range(worker, VOCAB // VOCAB_TILE, workers):
                col = panel * VOCAB_TILE
                for row in pl.range(0, padded, 16):
                    left = pl.load(embedding, [row, 0], [16, MARKOV_RANK], target_memory=pl.Mem.Mat)
                    right = pl.load(markov_w2, [col, 0], [VOCAB_TILE, MARKOV_RANK], target_memory=pl.Mem.Mat)
                    right_transposed = pl.tile.transpose_view(right)
                    product = pl.matmul(left, right_transposed, out_dtype=pl.FP32)
                    pl.store(product, [row, col], bias)

        scores = pl.create_tensor([batch, VOCAB], dtype=pl.FP32)
        for worker in pl.spmd(request_workers, name_hint="markov_proposal_logits"):
            for request in pl.range(worker, batch, request_workers):
                row = request * DSPARK_QUERY_WIDTH + step
                for col in pl.range(0, VOCAB, 1280):
                    logits = pl.load(base, [row, col], [1, 1280])
                    correction = pl.load(bias, [request, col], [1, 1280])
                    pl.store(pl.add(logits, correction), [request, col], scores)

        scores_grid = pl.reshape(scores, [batch * (VOCAB // ARGMAX_WIDTH_TILE), ARGMAX_WIDTH_TILE])
        filtered = pl.create_tensor([batch, VOCAB], dtype=pl.FP32)
        weights = pl.create_tensor([batch, VOCAB], dtype=pl.FP32)
        selected = pl.create_tensor([batch, 8], dtype=pl.INT32)
        confidence_state = pl.create_tensor([batch, 80], dtype=pl.FP32)
        for worker in pl.spmd(request_workers, name_hint="markov_proposal_sample"):
            for request in pl.range(worker, batch, request_workers):
                token = sample_markov_proposal_row(
                    scores, scores_grid, filtered, weights, proposal_probs, temperatures, top_ks,
                    top_ps, seeds, first_positions, request, step, DECODE_SEQ,
                )
                sampled = pl.full([1, 8], dtype=pl.INT32, value=0)
                pl.write(sampled, [0, 0], token)
                selected[request : request + 1, :] = sampled
                _confidence = _store_markov_confidence(
                    hidden, embedding, confidence_weight, confidence_state, request, step,
                )

        for _worker in pl.spmd(1, name_hint="markov_proposal_publish"):
            for request in pl.range(batch):
                pl.write(draft_ids, [request, step], pl.read(selected, [request, 0]))
                pl.write(confidences, [request, step], pl.read(confidence_state, [request, step * 16]))
        previous_selected = selected
    return draft_ids, confidences, proposal_probs


@pl.jit(auto_scope=False)
def markov_proposal_test(
    base_logits: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, VOCAB], pl.FP32],
    head_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16],
    anchor_ids: pl.Tensor[[B_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB, MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB, MARKOV_RANK], pl.BF16],
    confidence_weight: pl.Tensor[[1, D + MARKOV_RANK], pl.FP32],
    temperatures: pl.Tensor[[PROPOSAL_PARAMS_DYN], pl.FP32],
    top_ks: pl.Tensor[[PROPOSAL_PARAMS_DYN], pl.INT32],
    top_ps: pl.Tensor[[PROPOSAL_PARAMS_DYN], pl.FP32],
    seeds: pl.Tensor[[PROPOSAL_PARAMS_DYN], pl.INT32],
    first_positions: pl.Tensor[[B_DYN], pl.INT32],
    draft_ids: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.INT32]],
    confidences: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.FP32]],
    proposal_probs: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH * VOCAB], pl.FP32]],
):
    anchor_ids.bind_dynamic(0, B_DYN)
    return markov_propose(
        base_logits, head_hidden, anchor_ids, markov_w1, markov_w2, confidence_weight,
        temperatures, top_ks, top_ps, seeds, first_positions,
        draft_ids, confidences, proposal_probs,
    )


@pl.jit(auto_scope=False)
def l2_markov_output(
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
    temperatures: pl.Tensor[[DECODE_BATCH * DECODE_SEQ], pl.FP32],
    top_ks: pl.Tensor[[DECODE_BATCH * DECODE_SEQ], pl.INT32],
    top_ps: pl.Tensor[[DECODE_BATCH * DECODE_SEQ], pl.FP32],
    seeds: pl.Tensor[[DECODE_BATCH * DECODE_SEQ], pl.INT32],
    first_positions: pl.Tensor[[B_DYN], pl.INT32],
    proposal_probs: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH * VOCAB], pl.FP32]],
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
    first_positions.bind_dynamic(0, B_DYN)
    proposal_probs.bind_dynamic(0, B_DYN)
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
    return markov_propose(
        base_logits, head_hidden, anchor_ids, markov_w1, markov_w2, confidence_weight,
        temperatures, top_ks, top_ps, seeds, first_positions,
        draft_ids, confidences, proposal_probs,
    )


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
    temperatures: pl.Tensor[[EP_SIZE, DECODE_BATCH * DECODE_SEQ], pl.FP32],
    top_ks: pl.Tensor[[EP_SIZE, DECODE_BATCH * DECODE_SEQ], pl.INT32],
    top_ps: pl.Tensor[[EP_SIZE, DECODE_BATCH * DECODE_SEQ], pl.FP32],
    seeds: pl.Tensor[[EP_SIZE, DECODE_BATCH * DECODE_SEQ], pl.INT32],
    first_positions: pl.Tensor[[EP_SIZE, B_DYN], pl.INT32],
    proposal_probs: pl.Out[pl.Tensor[[EP_SIZE, B_DYN, DSPARK_QUERY_WIDTH * VOCAB], pl.FP32]],
    draft_ids: pl.Out[pl.Tensor[[EP_SIZE, B_DYN, DSPARK_QUERY_WIDTH], pl.INT32]],
    confidences: pl.Out[pl.Tensor[[EP_SIZE, B_DYN, DSPARK_QUERY_WIDTH], pl.FP32]],
    logits_epoch: pl.Scalar[pl.INT32],
):
    first_positions.bind_dynamic(1, B_DYN)
    proposal_probs.bind_dynamic(1, B_DYN)
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
            temperatures[rank], top_ks[rank], top_ps[rank], seeds[rank],
            first_positions[rank], proposal_probs[rank], draft_ids[rank], confidences[rank],
            hidden_window, hidden_done, logits_window, logits_done,
            rank, logits_epoch, device=rank,
        )


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


def build_proposal_specs():
    from golden import TensorSpec

    specs = build_tensor_specs()
    by_name = {spec.name: spec for spec in specs}
    logits = by_name["base_logits"].init_value
    logits[1].fill_(-4.0)
    logits[1, :, 5:9] = torch.tensor([0.1, 0.3, 0.2, 0.4])
    by_name["markov_w2"].init_value.div_(128)
    params = [
        TensorSpec("temperatures", [2 * DECODE_SEQ], torch.float32, torch.tensor([0.0, 0.7]).repeat_interleave(DECODE_SEQ)),
        TensorSpec("top_ks", [2 * DECODE_SEQ], torch.int32, 3),
        TensorSpec("top_ps", [2 * DECODE_SEQ], torch.float32, 0.75),
        TensorSpec("seeds", [2 * DECODE_SEQ], torch.int32, 71),
        TensorSpec("first_positions", [2], torch.int32, torch.tensor([51, 73])),
    ]
    return specs[:-2] + params + specs[-2:] + [
        TensorSpec("proposal_probs", [2, DSPARK_QUERY_WIDTH * VOCAB], torch.float32),
    ]


def golden_proposal_test(tensors):
    proposal_probs = tensors["proposal_probs"].view(2, DSPARK_QUERY_WIDTH, VOCAB)
    for request in range(2):
        params_row = request * DECODE_SEQ
        temperature = float(tensors["temperatures"][params_row])
        top_k = int(tensors["top_ks"][params_row])
        top_p = float(tensors["top_ps"][params_row])
        seed = int(tensors["seeds"][params_row])
        first = int(tensors["first_positions"][request])
        previous = int(tensors["anchor_ids"][request])
        for step in range(DSPARK_QUERY_WIDTH):
            embedding = tensors["markov_w1"][previous].float()
            bias = tensors["markov_w2"].float() @ embedding
            scores = tensors["base_logits"][request, step] + bias
            if temperature == 0:
                previous = int(scores.argmax())
                proposal_probs[request, step, previous] = 1.0
            else:
                probs = golden_dspark_probs(scores, temperature, top_k, top_p)
                proposal_probs[request, step] = probs
                previous = golden_categorical(probs, seed, first + step, PROPOSAL_STREAM)
            tensors["draft_ids"][request, step] = previous
            hidden = tensors["head_hidden"][request, step].float()
            features = torch.cat((hidden, embedding))
            score = features @ tensors["confidence_weight"][0]
            tensors["confidences"][request, step] = score.sigmoid()


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
        a5_default_runtime="host_build_graph",
        fn=markov_proposal_test,
        specs=build_proposal_specs(),
        golden_fn=golden_proposal_test,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        config=dict(backend_type=args.platform, execution_mode=False, device_id=args.device),
        rtol=1e-3,
        atol=1e-3,
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
