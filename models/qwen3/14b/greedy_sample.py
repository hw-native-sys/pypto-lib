# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Greedy token sampling for Qwen3-14B logits."""

from __future__ import annotations

import pypto.language as pl

from config import BATCH, REAL_VOCAB, VOCAB


VOCAB_CHUNK = 512
CHUNK_PAD = 512
NUM_VOCAB_CHUNKS = VOCAB // VOCAB_CHUNK
REAL_NUM_FULL_VOCAB_CHUNKS = REAL_VOCAB // VOCAB_CHUNK
REAL_VOCAB_TAIL = REAL_VOCAB % VOCAB_CHUNK
REAL_NUM_VOCAB_CHUNKS = REAL_NUM_FULL_VOCAB_CHUNKS + (1 if REAL_VOCAB_TAIL != 0 else 0)
SAMPLED_IDS_PAD = 8
TOPK = 16

assert VOCAB % VOCAB_CHUNK == 0
assert NUM_VOCAB_CHUNKS <= CHUNK_PAD
assert REAL_VOCAB <= VOCAB


@pl.jit
def greedy_sample_fwd(
    logits: pl.Tensor[[BATCH, VOCAB], pl.FP32],
    sampled_ids: pl.Out[pl.Tensor[[BATCH, SAMPLED_IDS_PAD], pl.INT32]],
):
    """Select the argmax token id per batch row."""
    for b in pl.parallel(BATCH):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="greedy_sample"):
            idx_init = pl.arange(0, [1, VOCAB_CHUNK], dtype=pl.UINT32)
            chunk_vals = pl.create_tensor([1, CHUNK_PAD], dtype=pl.FP32)
            chunk_vals[:, :] = pl.full([1, CHUNK_PAD], dtype=pl.FP32, value=-3.402823e38)
            for c in pl.range(REAL_NUM_VOCAB_CHUNKS):
                c0 = c * VOCAB_CHUNK
                local_scores = logits[b : b + 1, c0 : c0 + VOCAB_CHUNK]
                if REAL_VOCAB_TAIL != 0:
                    if c == REAL_NUM_FULL_VOCAB_CHUNKS:
                        local_scores_valid = pl.set_validshape(local_scores, 1, REAL_VOCAB_TAIL)
                        local_scores_padded = pl.fillpad(local_scores_valid, pad_value=pl.PadValue.min)
                        sorted_pairs = pl.sort32(local_scores_padded, idx_init)
                    else:
                        sorted_pairs = pl.sort32(local_scores, idx_init)
                else:
                    sorted_pairs = pl.sort32(local_scores, idx_init)
                sorted_pairs = pl.mrgsort(sorted_pairs, block_len=64)
                sorted_pairs = pl.mrgsort(sorted_pairs, block_len=256)
                top_pairs = sorted_pairs[:, 0 : 2 * TOPK]
                top_vals = pl.gather(top_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
                best_val = pl.read(top_vals, [0, 0])
                pl.write(chunk_vals, [0, c], best_val)

            chunk_sorted = pl.sort32(chunk_vals, idx_init)
            chunk_sorted = pl.mrgsort(chunk_sorted, block_len=64)
            chunk_sorted = pl.mrgsort(chunk_sorted, block_len=256)
            chunk_top_pairs = chunk_sorted[:, 0 : 2 * TOPK]
            chunk_top_vals = pl.gather(chunk_top_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
            best_val = pl.read(chunk_top_vals, [0, 0])
            chunk_i32 = pl.cast(0, pl.INT32)
            for c in pl.range(REAL_NUM_VOCAB_CHUNKS):
                scan_c = (REAL_NUM_VOCAB_CHUNKS - 1) - c
                val = pl.read(chunk_vals, [0, scan_c])
                if val == best_val:
                    chunk_i32 = pl.cast(scan_c, pl.INT32)

            local_token = pl.cast(0, pl.INT32)
            chunk_base = chunk_i32 * pl.cast(VOCAB_CHUNK, target_type=pl.INT32)
            chunk_base_idx = pl.cast(chunk_base, target_type=pl.INDEX)
            winning_logits = pl.slice(logits, [1, VOCAB_CHUNK], [pl.cast(b, pl.INDEX), chunk_base_idx])
            if REAL_VOCAB_TAIL != 0:
                if chunk_i32 == pl.cast(REAL_NUM_FULL_VOCAB_CHUNKS, target_type=pl.INT32):
                    winning_logits_valid = pl.set_validshape(winning_logits, 1, REAL_VOCAB_TAIL)
                    winning_logits_padded = pl.fillpad(winning_logits_valid, pad_value=pl.PadValue.min)
                    for t in pl.range(VOCAB_CHUNK):
                        scan_t = (VOCAB_CHUNK - 1) - t
                        val = pl.read(winning_logits_padded, [0, pl.cast(scan_t, pl.INDEX)])
                        if val == best_val:
                            local_token = pl.cast(scan_t, pl.INT32)
                else:
                    for t in pl.range(VOCAB_CHUNK):
                        scan_t = (VOCAB_CHUNK - 1) - t
                        val = pl.read(winning_logits, [0, pl.cast(scan_t, pl.INDEX)])
                        if val == best_val:
                            local_token = pl.cast(scan_t, pl.INT32)
            else:
                for t in pl.range(VOCAB_CHUNK):
                    scan_t = (VOCAB_CHUNK - 1) - t
                    val = pl.read(winning_logits, [0, pl.cast(scan_t, pl.INDEX)])
                    if val == best_val:
                        local_token = pl.cast(scan_t, pl.INT32)
            token_id = chunk_base + local_token
            if token_id >= pl.cast(REAL_VOCAB, target_type=pl.INT32):
                token_id = pl.cast(0, pl.INT32)
            token_out = pl.create_tensor([1, SAMPLED_IDS_PAD], dtype=pl.INT32)
            token_out[:, :] = pl.full([1, SAMPLED_IDS_PAD], dtype=pl.INT32, value=0)
            pl.write(token_out, [0, 0], token_id)
            sampled_ids[b : b + 1, :] = token_out

    return sampled_ids


def build_tensor_specs():
    import torch

    from golden import TensorSpec

    def init_logits():
        logits = torch.full((BATCH, VOCAB), -1000.0)
        logits[0::2, 0] = 0.0
        logits[0::2, 7] = 5.0
        logits[0::2, 42] = 5.0
        logits[1::2, 0] = 5.0
        logits[1::2, 7] = 0.0
        logits[:, REAL_VOCAB:] = logits[:, :1]
        if REAL_VOCAB < VOCAB:
            logits[0, REAL_VOCAB] = 6.0
        return logits

    return [
        TensorSpec("logits", [BATCH, VOCAB], torch.float32, init_value=init_logits),
        TensorSpec("sampled_ids", [BATCH, SAMPLED_IDS_PAD], torch.int32, is_output=True),
    ]


def golden_greedy_sample(tensors):
    import torch

    logits = tensors["logits"].float()
    sampled_ids = torch.argmax(logits[:, :REAL_VOCAB], dim=-1).to(torch.int32).view(BATCH, 1)
    tensors["sampled_ids"][:] = 0
    tensors["sampled_ids"][:, :1] = sampled_ids


if __name__ == "__main__":
    import argparse

    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=greedy_sample_fwd,
        specs=build_tensor_specs(),
        golden_fn=golden_greedy_sample,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=0,
        atol=0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
