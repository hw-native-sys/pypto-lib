# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compose the shared target LM head and rank-256 sequential Markov sampler."""

import pypto.language as pl

from config import FLASH as M
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
# Base-logit row tiling must exactly cover every supported padded batch.
for _batch in DSPARK_SUPPORTED_BATCHES:
    assert (_batch * DSPARK_QUERY_PAD) % LM_M_TILE == 0


@pl.jit.inline
def compute_base_logits(
    head_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB, D], pl.BF16],
    base_logits: pl.Tensor[[DSPARK_MOE_TOKENS, VOCAB], pl.FP32],
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
    normalized = pl.create_tensor([DSPARK_MOE_TOKENS, D], dtype=pl.BF16)
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
    base_logits: pl.Tensor[[DSPARK_MOE_TOKENS, VOCAB], pl.FP32],
    num_sampled: pl.Tensor[[B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[B_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    draft_token_scratch: pl.Tensor[[MARKOV_M_TILE, MARKOV_ID_PAD], pl.INT32],
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
    markov_bias, markov_embedding = markov_head(
        previous_token_ids,
        markov_w1,
        markov_w2,
        markov_bias,
        markov_embedding,
    )

    with pl.spmd(
        MARKOV_M_TILE,
        name_hint="dspark_markov_greedy",
        deps=[base_logits_ready_tid, previous_tokens_tid],
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
    draft_token_ids: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.INT32]],
):
    head_hidden.bind_dynamic(0, B_DYN)
    num_sampled.bind_dynamic(0, B_DYN)
    last_sampled.bind_dynamic(0, B_DYN)
    next_prefill_tokens.bind_dynamic(0, B_DYN)
    draft_token_ids.bind_dynamic(0, B_DYN)
    batch = pl.tensor.dim(num_sampled, 0)
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
    draft_token_scratch = pl.create_tensor(
        [MARKOV_M_TILE, MARKOV_ID_PAD],
        dtype=pl.INT32,
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
        base_logits, num_sampled, last_sampled, next_prefill_tokens,
        markov_w1, markov_w2,
        draft_token_scratch,
        base_logits_tid, token_scratch_zero_tid, pl.cast(0, pl.INT32)
    )
    step_1_tid = greedy_markov_step(
        base_logits, num_sampled, last_sampled, next_prefill_tokens,
        markov_w1, markov_w2,
        draft_token_scratch,
        base_logits_tid, step_0_tid, pl.cast(1, pl.INT32)
    )
    step_2_tid = greedy_markov_step(
        base_logits, num_sampled, last_sampled, next_prefill_tokens,
        markov_w1, markov_w2,
        draft_token_scratch,
        base_logits_tid, step_1_tid, pl.cast(2, pl.INT32)
    )
    step_3_tid = greedy_markov_step(
        base_logits, num_sampled, last_sampled, next_prefill_tokens,
        markov_w1, markov_w2,
        draft_token_scratch,
        base_logits_tid, step_2_tid, pl.cast(3, pl.INT32)
    )
    step_4_tid = greedy_markov_step(
        base_logits, num_sampled, last_sampled, next_prefill_tokens,
        markov_w1, markov_w2,
        draft_token_scratch,
        base_logits_tid, step_3_tid, pl.cast(4, pl.INT32)
    )
    step_5_tid = greedy_markov_step(
        base_logits, num_sampled, last_sampled, next_prefill_tokens,
        markov_w1, markov_w2,
        draft_token_scratch,
        base_logits_tid, step_4_tid, pl.cast(5, pl.INT32)
    )
    step_6_tid = greedy_markov_step(
        base_logits, num_sampled, last_sampled, next_prefill_tokens,
        markov_w1, markov_w2,
        draft_token_scratch,
        base_logits_tid, step_5_tid, pl.cast(6, pl.INT32)
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
    return draft_token_ids


def build_tensor_specs(batch: int):
    """Build a deterministic nonzero Markov validation case."""
    import torch
    from golden import TensorSpec

    if batch not in DSPARK_SUPPORTED_BATCHES:
        raise ValueError(f"unsupported DSpark batch {batch}; expected one of {DSPARK_SUPPORTED_BATCHES}")

    def init_lm_head_weight():
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

    return [
        TensorSpec("head_hidden", [batch, DSPARK_QUERY_WIDTH, D], torch.bfloat16, init_value=1),
        TensorSpec("final_norm_weight", [D], torch.bfloat16, init_value=1),
        TensorSpec("lm_head_weight", [VOCAB, D], torch.bfloat16, init_value=init_lm_head_weight),
        TensorSpec(
            "num_sampled",
            [batch],
            torch.int32,
            init_value=lambda: torch.arange(batch, dtype=torch.int32) % 2,
        ),
        TensorSpec("last_sampled", [batch], torch.int64, init_value=init_last_sampled),
        TensorSpec(
            "next_prefill_tokens",
            [batch],
            torch.int64,
            init_value=init_next_prefill_tokens,
        ),
        TensorSpec("markov_w1", [VOCAB, DSPARK_MARKOV_RANK], torch.bfloat16, init_value=init_markov_w1),
        TensorSpec("markov_w2", [VOCAB, DSPARK_MARKOV_RANK], torch.bfloat16, init_value=init_markov_w2),
        TensorSpec(
            "draft_token_ids",
            [batch, DSPARK_QUERY_WIDTH],
            torch.int32,
            is_output=True,
        ),
    ]


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
        markov_bias = embedding.matmul(
            tensors["markov_w2"][:validation_support].float().t()
        )
        scores = base_logits[:, step] + markov_bias
        assert torch.all(scores.max(dim=-1).values > 0)
        previous = torch.argmax(scores, dim=-1)
        tensors["draft_token_ids"][:, step] = previous.to(torch.int32)


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser(description="Validate the DeepSeek V4 DSpark Markov sampler.")
    parser.add_argument("--batch", type=int, choices=DSPARK_SUPPORTED_BATCHES, default=4)
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--dump-passes", action="store_true")
    args = parser.parse_args()

    result = run_jit(
        fn=markov_sample,
        specs=build_tensor_specs(args.batch),
        golden_fn=golden_nonzero_markov,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=2e-3,
        atol=2e-3,
        compile_only=args.compile_only,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
