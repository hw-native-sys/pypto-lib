# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Top-k candidate selection for Qwen3-14B logits."""

from __future__ import annotations

import pypto.language as pl

from config import QWEN3_14B as M
from config import QWEN3_14B_TILING as T


BATCH = M.batch
VOCAB = M.vocab
REAL_VOCAB = M.real_vocab
VOCAB_CHUNK = T.vocab_chunk
NUM_VOCAB_CHUNKS = VOCAB // VOCAB_CHUNK
REAL_NUM_FULL_VOCAB_CHUNKS = REAL_VOCAB // VOCAB_CHUNK
REAL_VOCAB_TAIL = REAL_VOCAB % VOCAB_CHUNK
REAL_NUM_VOCAB_CHUNKS = REAL_NUM_FULL_VOCAB_CHUNKS + (1 if REAL_VOCAB_TAIL != 0 else 0)
TOPK = 32
TOPK_GROUP_WIDTH = 2048
TOPK_NUM_FULL_GROUPS = REAL_VOCAB // TOPK_GROUP_WIDTH
TOPK_GROUP_TAIL = REAL_VOCAB % TOPK_GROUP_WIDTH
TOPK_NUM_GROUPS = TOPK_NUM_FULL_GROUPS + (1 if TOPK_GROUP_TAIL != 0 else 0)
TOPK_CANDIDATE_PAD = 4096
TOPK_FINAL_HALF = TOPK_CANDIDATE_PAD // 2
GREEDY_CHUNK_PAD = VOCAB_CHUNK
GREEDY_SORT_TOPK = 16
FP32_NEG_INF = -3.402823e38

assert VOCAB % VOCAB_CHUNK == 0
assert REAL_VOCAB <= VOCAB
assert TOPK <= VOCAB_CHUNK
assert VOCAB_CHUNK == 512
assert TOPK_GROUP_WIDTH % VOCAB_CHUNK == 0
assert TOPK_GROUP_TAIL == REAL_VOCAB_TAIL
assert TOPK_NUM_GROUPS * TOPK <= TOPK_CANDIDATE_PAD
assert NUM_VOCAB_CHUNKS <= GREEDY_CHUNK_PAD


@pl.jit.inline
def _topk_group_pairs(
    logits: pl.Tensor[[BATCH, VOCAB], pl.FP32],
    batch_idx: pl.Scalar[pl.INDEX],
    group_idx: pl.Scalar[pl.INDEX],
):
    group_start = group_idx * TOPK_GROUP_WIDTH
    scores = logits[batch_idx : batch_idx + 1, group_start : group_start + TOPK_GROUP_WIDTH]
    indices = pl.arange(
        pl.cast(group_start, target_type=pl.UINT32),
        [1, TOPK_GROUP_WIDTH],
        dtype=pl.UINT32,
    )
    pairs = pl.sort32(scores, indices)
    pairs = pl.mrgsort(pairs, block_len=64)
    pairs = pl.mrgsort(pairs, block_len=256)
    pairs = pl.mrgsort(pairs, block_len=1024)
    return pairs[:, 0 : 2 * TOPK]


@pl.jit
def topk_select_fwd(
    logits: pl.Tensor[[BATCH, VOCAB], pl.FP32],
    sampling_control: pl.Tensor[[2], pl.INT32],
    topk_values: pl.Out[pl.Tensor[[BATCH, TOPK], pl.FP32]],
    topk_indices: pl.Out[pl.Tensor[[BATCH, TOPK], pl.INT32]],
):
    """Return greedy top-1 or the largest TOPK real-vocab candidates per row."""
    for b in pl.parallel(BATCH):
        if b < pl.read(sampling_control, [0]):
            if pl.read(sampling_control, [1]) == 1:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="greedy_select"):
                    idx_init = pl.arange(0, [1, VOCAB_CHUNK], dtype=pl.UINT32)
                    chunk_vals = pl.create_tensor([1, GREEDY_CHUNK_PAD], dtype=pl.FP32)
                    chunk_vals[:, :] = pl.full([1, GREEDY_CHUNK_PAD], dtype=pl.FP32, value=FP32_NEG_INF)
                    for c in pl.range(REAL_NUM_VOCAB_CHUNKS):
                        c0 = c * VOCAB_CHUNK
                        local_scores = logits[b : b + 1, c0 : c0 + VOCAB_CHUNK]
                        if REAL_VOCAB_TAIL != 0:
                            if c == REAL_NUM_FULL_VOCAB_CHUNKS:
                                local_scores_valid = pl.set_validshape(local_scores, 1, REAL_VOCAB_TAIL)
                                local_scores_padded = pl.fillpad(
                                    local_scores_valid, pad_value=pl.PadValue.min
                                )
                                sorted_pairs = pl.sort32(local_scores_padded, idx_init)
                            else:
                                sorted_pairs = pl.sort32(local_scores, idx_init)
                        else:
                            sorted_pairs = pl.sort32(local_scores, idx_init)
                        sorted_pairs = pl.mrgsort(sorted_pairs, block_len=64)
                        sorted_pairs = pl.mrgsort(sorted_pairs, block_len=256)
                        top_pairs = sorted_pairs[:, 0 : 2 * GREEDY_SORT_TOPK]
                        top_vals = pl.gather(top_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
                        pl.write(chunk_vals, [0, c], pl.read(top_vals, [0, 0]))

                    chunk_sorted = pl.sort32(chunk_vals, idx_init)
                    chunk_sorted = pl.mrgsort(chunk_sorted, block_len=64)
                    chunk_sorted = pl.mrgsort(chunk_sorted, block_len=256)
                    chunk_top_pairs = chunk_sorted[:, 0 : 2 * GREEDY_SORT_TOPK]
                    chunk_top_vals = pl.gather(chunk_top_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
                    best_val = pl.read(chunk_top_vals, [0, 0])
                    chunk_i32 = pl.cast(0, pl.INT32)
                    for c in pl.range(REAL_NUM_VOCAB_CHUNKS):
                        scan_c = (REAL_NUM_VOCAB_CHUNKS - 1) - c
                        if pl.read(chunk_vals, [0, scan_c]) == best_val:
                            chunk_i32 = pl.cast(scan_c, pl.INT32)

                    local_token = pl.cast(0, pl.INT32)
                    chunk_base = chunk_i32 * pl.cast(VOCAB_CHUNK, target_type=pl.INT32)
                    winning_logits = pl.slice(
                        logits,
                        [1, VOCAB_CHUNK],
                        [pl.cast(b, pl.INDEX), pl.cast(chunk_base, target_type=pl.INDEX)],
                    )
                    if REAL_VOCAB_TAIL != 0:
                        if chunk_i32 == pl.cast(REAL_NUM_FULL_VOCAB_CHUNKS, target_type=pl.INT32):
                            winning_logits_valid = pl.set_validshape(winning_logits, 1, REAL_VOCAB_TAIL)
                            winning_logits_padded = pl.fillpad(
                                winning_logits_valid, pad_value=pl.PadValue.min
                            )
                            for t in pl.range(VOCAB_CHUNK):
                                scan_t = (VOCAB_CHUNK - 1) - t
                                if (
                                    pl.read(
                                        winning_logits_padded,
                                        [0, pl.cast(scan_t, pl.INDEX)],
                                    )
                                    == best_val
                                ):
                                    local_token = pl.cast(scan_t, pl.INT32)
                        else:
                            for t in pl.range(VOCAB_CHUNK):
                                scan_t = (VOCAB_CHUNK - 1) - t
                                if pl.read(winning_logits, [0, pl.cast(scan_t, pl.INDEX)]) == best_val:
                                    local_token = pl.cast(scan_t, pl.INT32)
                    else:
                        for t in pl.range(VOCAB_CHUNK):
                            scan_t = (VOCAB_CHUNK - 1) - t
                            if pl.read(winning_logits, [0, pl.cast(scan_t, pl.INDEX)]) == best_val:
                                local_token = pl.cast(scan_t, pl.INT32)

                    token_id = chunk_base + local_token
                    if token_id >= pl.cast(REAL_VOCAB, target_type=pl.INT32):
                        token_id = pl.cast(0, pl.INT32)
                    topk_values[b : b + 1, :] = pl.full([1, TOPK], dtype=pl.FP32, value=FP32_NEG_INF)
                    topk_indices[b : b + 1, :] = pl.full([1, TOPK], dtype=pl.INT32, value=0)
                    pl.write(topk_values, [b, 0], best_val)
                    pl.write(topk_indices, [b, 0], token_id)
            else:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="topk_select"):
                    candidate_vals = pl.create_tensor([1, TOPK_CANDIDATE_PAD], dtype=pl.FP32)
                    candidate_vals[:, :] = pl.full(
                        [1, TOPK_CANDIDATE_PAD], dtype=pl.FP32, value=FP32_NEG_INF
                    )
                    candidate_ids = pl.create_tensor([1, TOPK_CANDIDATE_PAD], dtype=pl.INT32)
                    candidate_ids[:, :] = pl.full([1, TOPK_CANDIDATE_PAD], dtype=pl.INT32, value=0)

                    for g in pl.range(TOPK_NUM_FULL_GROUPS):
                        group_pairs = _topk_group_pairs(logits, b, g)
                        group_vals = pl.gather(group_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
                        group_ids = pl.gather(
                            group_pairs,
                            mask_pattern=pl.tile.MaskPattern.P1010,
                            output_dtype=pl.INT32,
                        )
                        candidate_offset = g * TOPK
                        for k in pl.range(TOPK):
                            pl.write(
                                candidate_vals,
                                [0, candidate_offset + k],
                                pl.read(group_vals, [0, k]),
                            )
                            pl.write(
                                candidate_ids,
                                [0, candidate_offset + k],
                                pl.read(group_ids, [0, k]),
                            )

                    if TOPK_GROUP_TAIL != 0:
                        tail_start = TOPK_NUM_FULL_GROUPS * TOPK_GROUP_WIDTH
                        tail_scores_raw = logits[
                            b : b + 1, tail_start : tail_start + VOCAB_CHUNK
                        ]
                        tail_scores = pl.fillpad(
                            pl.set_validshape(tail_scores_raw, 1, TOPK_GROUP_TAIL),
                            pad_value=pl.PadValue.min,
                        )
                        tail_indices = pl.arange(
                            pl.cast(tail_start, target_type=pl.UINT32),
                            [1, VOCAB_CHUNK],
                            dtype=pl.UINT32,
                        )
                        tail_pairs = pl.sort32(tail_scores, tail_indices)
                        tail_pairs = pl.mrgsort(tail_pairs, block_len=64)
                        tail_pairs = pl.mrgsort(tail_pairs, block_len=256)
                        tail_pairs = tail_pairs[:, 0 : 2 * TOPK]
                        tail_vals = pl.gather(tail_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
                        tail_ids = pl.gather(
                            tail_pairs,
                            mask_pattern=pl.tile.MaskPattern.P1010,
                            output_dtype=pl.INT32,
                        )
                        tail_offset = TOPK_NUM_FULL_GROUPS * TOPK
                        for k in pl.range(TOPK):
                            pl.write(
                                candidate_vals,
                                [0, tail_offset + k],
                                pl.read(tail_vals, [0, k]),
                            )
                            pl.write(
                                candidate_ids,
                                [0, tail_offset + k],
                                pl.read(tail_ids, [0, k]),
                            )

                    candidate_positions = pl.arange(0, [1, TOPK_CANDIDATE_PAD], dtype=pl.UINT32)
                    candidate_sorted = pl.sort32(candidate_vals, candidate_positions)
                    candidate_sorted = pl.mrgsort(candidate_sorted, block_len=64)
                    candidate_sorted = pl.mrgsort(candidate_sorted, block_len=256)
                    candidate_sorted = pl.mrgsort(candidate_sorted, block_len=1024)
                    half0_pairs = candidate_sorted[:, 0 : 2 * TOPK]
                    half1_pairs = candidate_sorted[
                        :, 2 * TOPK_FINAL_HALF : 2 * TOPK_FINAL_HALF + 2 * TOPK
                    ]
                    candidate_pairs = pl.mrgsort(half0_pairs, half1_pairs)[:, 0 : 2 * TOPK]
                    topk_values[b : b + 1, :] = pl.gather(
                        candidate_pairs,
                        mask_pattern=pl.tile.MaskPattern.P0101,
                    )
                    selected_positions = pl.gather(
                        candidate_pairs,
                        mask_pattern=pl.tile.MaskPattern.P1010,
                        output_dtype=pl.INT32,
                    )
                    for k in pl.range(TOPK):
                        candidate_pos = pl.read(selected_positions, [0, k])
                        token_id = pl.read(
                            candidate_ids,
                            [0, pl.cast(candidate_pos, target_type=pl.INDEX)],
                        )
                        pl.write(topk_indices, [b, k], token_id)
        else:
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="topk_select_inactive"):
                topk_values[b : b + 1, :] = pl.full([1, TOPK], dtype=pl.FP32, value=FP32_NEG_INF)
                topk_indices[b : b + 1, :] = pl.full([1, TOPK], dtype=pl.INT32, value=0)

    return topk_values, topk_indices


def build_tensor_specs(selection_k=TOPK):
    import torch

    from golden import TensorSpec

    def init_logits():
        if selection_k == 1:
            logits = torch.full((BATCH, VOCAB), -1000.0, dtype=torch.float32)
            logits[:, 7] = 5.0
            logits[:, 42] = 5.0
            if REAL_VOCAB < VOCAB:
                logits[0, REAL_VOCAB] = 6.0
            return logits

        logits = torch.randn(BATCH, VOCAB, dtype=torch.float32)
        logits[:, REAL_VOCAB:] = -1000.0
        logits[0, 0:TOPK] = torch.arange(TOPK, 0, -1, dtype=torch.float32) + 1000.0
        if BATCH > 1:
            spread_ids = torch.arange(TOPK, dtype=torch.long) * (REAL_VOCAB // TOPK) + 7
            logits[1, spread_ids] = torch.arange(TOPK, 0, -1, dtype=torch.float32) + 1000.0
        if REAL_VOCAB < VOCAB:
            logits[0, REAL_VOCAB] = 2000.0
        return logits

    return [
        TensorSpec("logits", [BATCH, VOCAB], torch.float32, init_value=init_logits),
        TensorSpec(
            "sampling_control",
            [2],
            torch.int32,
            init_value=lambda: torch.tensor([2 if selection_k == TOPK else 1, selection_k], dtype=torch.int32),
        ),
        TensorSpec("topk_values", [BATCH, TOPK], torch.float32, is_output=True),
        TensorSpec("topk_indices", [BATCH, TOPK], torch.int32, is_output=True),
    ]


def golden_topk_select(tensors):
    import torch

    active_batch = int(tensors["sampling_control"][0].item())
    selection_k = int(tensors["sampling_control"][1].item())
    tensors["topk_values"][:] = FP32_NEG_INF
    tensors["topk_indices"][:] = 0
    logits = tensors["logits"][:active_batch, :REAL_VOCAB].float()
    if selection_k == 1:
        token_ids = torch.argmax(logits, dim=-1)
        values = torch.gather(logits, dim=-1, index=token_ids[:, None])
        tensors["topk_values"][:active_batch, :1] = values
        tensors["topk_indices"][:active_batch, :1] = token_ids[:, None].to(torch.int32)
        return

    vals, idx = torch.topk(logits, TOPK, dim=-1, largest=True, sorted=True)
    tensors["topk_values"][:active_batch] = vals
    tensors["topk_indices"][:active_batch] = idx.to(torch.int32)


if __name__ == "__main__":
    import argparse

    from golden import run_jit, topk_pair_compare

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"]
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--selection-k", type=int, choices=[1, TOPK], default=TOPK)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=topk_select_fwd,
        specs=build_tensor_specs(args.selection_k),
        golden_fn=golden_topk_select,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-5,
        atol=1e-5,
        compare_fn={"topk_indices": topk_pair_compare("topk_values")},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
