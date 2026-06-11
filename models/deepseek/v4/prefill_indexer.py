# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F401,F403,F405,F821
"""DeepSeek-V4 prefill indexer scaffold.

Kernel body is intentionally empty; golden follows the torch reference for this stage.
"""

import pypto.language as pl

from decode_indexer import *  # noqa: F401,F403
from prefill_indexer_compressor import (
    STATE_LEN as INNER_STATE_LEN,
    golden_prefill_indexer_compressor,
    prefill_indexer_compressor,
)
from prefill_sparse_attn import (
    CMP_MAX_BLOCKS as SPARSE_CMP_MAX_BLOCKS,
    HCA_CMP_BLOCK_NUM as PREFILL_IDX_BLOCK_NUM,
    MAX_REQS,
    MAX_TOKENS,
    WIN,
)

B = 1
S = 128
T = B * S
START_POS = 0
OFFSET = S
PREFILL_COMPRESSED_LEN = S // COMPRESS_RATIO
PREFILL_CACHE_BLOCKS = (PREFILL_COMPRESSED_LEN + CACHE_TILE - 1) // CACHE_TILE
SCORE_B_GROUP = 1
PREFILL_Q_OUT_CHUCK = 128
D_CHUCK = 32
Q_CHUCK = 128
HEAD_ROWS = IDX_N_HEADS
HEAD_DIM_CHUCK = 32
TOPK_TILE = 16
assert T % TOPK_TILE == 0
INDEXER_SCORE_CAP = SPARSE_CMP_MAX_BLOCKS * BLOCK_SIZE
INDEXER_SCORE_BLOCKS = (INDEXER_SCORE_CAP + CACHE_TILE - 1) // CACHE_TILE
INDEXER_TOPK_CAP = min(IDX_TOPK, INDEXER_SCORE_CAP)
INDEXER_OFFSET = WIN + MAX_TOKENS
MAX_CMP_WRITES = MAX_REQS * max(1, MAX_TOKENS // COMPRESS_RATIO)
PACKED_Q_ROW_TILE = 16
PACKED_WEIGHT_ROW_TILE = 32
PACKED_Q_COL_BLOCKS = (IDX_N_HEADS * IDX_HEAD_DIM) // PREFILL_Q_OUT_CHUCK
PACKED_Q_BLOCKS = (MAX_TOKENS // PACKED_Q_ROW_TILE) * PACKED_Q_COL_BLOCKS
PACKED_WEIGHT_BLOCKS = MAX_TOKENS // PACKED_WEIGHT_ROW_TILE


@pl.jit.inline
def prefill_indexer(
    x: pl.Tensor[[MAX_TOKENS, D], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_kv_state: pl.Tensor[[MAX_REQS, INNER_STATE_LEN, INNER_OUT_DIM], pl.FP32],
    inner_score_state: pl.Tensor[[MAX_REQS, INNER_STATE_LEN, INNER_OUT_DIM], pl.FP32],
    inner_wkv: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_wgate: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.FP32],
    idx_kv_cache: pl.Out[pl.Tensor[[PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16]],
    cmp_topk_indices: pl.Out[pl.Tensor[[MAX_TOKENS, IDX_TOPK], pl.INT32]],
    token_to_request: pl.Tensor[[MAX_TOKENS], pl.INT32],
    position_ids: pl.Tensor[[MAX_TOKENS], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    num_cmp_writes: pl.Scalar[pl.INT32],
    cmp_write_token_ids: pl.Tensor[[MAX_CMP_WRITES], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[MAX_CMP_WRITES], pl.INT32],
):
    idx_kv_cache, inner_kv_state, inner_score_state = prefill_indexer_compressor(
        x,
        inner_kv_state,
        inner_score_state,
        inner_wkv,
        inner_wgate,
        inner_ape,
        inner_norm_w,
        freqs_cos,
        freqs_sin,
        hadamard,
        idx_kv_cache,
        token_to_request,
        position_ids,
        num_tokens,
        num_cmp_writes,
        cmp_write_token_ids,
        cmp_slot_mapping,
    )

    for topk_idx in pl.spmd(MAX_TOKENS // TOPK_TILE, name_hint="prefill_idx_topk"):
        topk_t0 = topk_idx * TOPK_TILE
        for topk_dt in pl.range(TOPK_TILE):
            topk_token = topk_t0 + topk_dt
            cmp_topk_indices[topk_token : topk_token + 1, 0:IDX_TOPK] = pl.full(
                [1, IDX_TOPK],
                dtype=pl.INT32,
                value=-1,
            )
            if topk_token < num_tokens:
                pos = pl.read(position_ids, [topk_token])
                visible_len = (pos + 1) // COMPRESS_RATIO
                if visible_len > 0:
                    for ck in pl.range(INDEXER_TOPK_CAP):
                        if ck < visible_len:
                            pl.write(cmp_topk_indices, [topk_token, ck], pl.cast(INDEXER_OFFSET + ck, pl.INT32))

    return idx_kv_cache, inner_kv_state, inner_score_state, cmp_topk_indices


def golden_prefill_indexer_core(tensors):
    import torch

    compressor_tensors = {
        "x": tensors["x"],
        "kv": torch.zeros(MAX_CMP_WRITES, IDX_HEAD_DIM, dtype=torch.bfloat16),
        "kv_state": tensors["inner_kv_state"],
        "score_state": tensors["inner_score_state"],
        "wkv": tensors["inner_wkv"],
        "wgate": tensors["inner_wgate"],
        "ape": tensors["inner_ape"],
        "norm_w": tensors["inner_norm_w"],
        "freqs_cos": tensors["freqs_cos"],
        "freqs_sin": tensors["freqs_sin"],
        "hadamard": tensors["hadamard"],
        "idx_kv_cache": tensors["idx_kv_cache"],
        "token_to_request": tensors["token_to_request"],
        "position_ids": tensors["position_ids"],
        "num_tokens": tensors["num_tokens"],
        "num_cmp_writes": tensors["num_cmp_writes"],
        "cmp_write_token_ids": tensors["cmp_write_token_ids"],
        "cmp_slot_mapping": tensors["cmp_slot_mapping"],
    }
    golden_prefill_indexer_compressor(compressor_tensors)
    tensors["idx_kv_cache"][:] = compressor_tensors["idx_kv_cache"]
    tensors["inner_kv_state"][:] = compressor_tensors["kv_state"]
    tensors["inner_score_state"][:] = compressor_tensors["score_state"]

    num_tokens = int(tensors["num_tokens"])
    position_ids = tensors["position_ids"]
    cmp_topk_indices = torch.full((MAX_TOKENS, IDX_TOPK), -1, dtype=torch.int32)
    for t in range(num_tokens):
        visible = min((int(position_ids[t].item()) + 1) // COMPRESS_RATIO, INDEXER_SCORE_CAP)
        k = min(INDEXER_TOPK_CAP, visible)
        if k > 0:
            cmp_topk_indices[t, :k] = torch.arange(k, dtype=torch.int32) + INDEXER_OFFSET
    return cmp_topk_indices


def golden_prefill_indexer(tensors):
    import torch

    cmp_topk_indices = golden_prefill_indexer_core(tensors)
    score = torch.zeros((MAX_TOKENS, INDEXER_SCORE_CAP), dtype=torch.float32)
    topk_idxs = torch.full((MAX_TOKENS, INDEXER_SCORE_CAP), -1, dtype=torch.int32)
    compare_cols = min(IDX_TOPK, INDEXER_SCORE_CAP)
    topk_idxs[:, 0:compare_cols] = cmp_topk_indices[:, 0:compare_cols]
    tensors["score"][:] = score
    tensors["topk_idxs"][:] = topk_idxs


@pl.jit
def prefill_indexer_test(
    x: pl.Tensor[[MAX_TOKENS, D], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_kv_state: pl.Tensor[[MAX_REQS, INNER_STATE_LEN, INNER_OUT_DIM], pl.FP32],
    inner_score_state: pl.Tensor[[MAX_REQS, INNER_STATE_LEN, INNER_OUT_DIM], pl.FP32],
    inner_wkv: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_wgate: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.FP32],
    idx_kv_cache: pl.Out[pl.Tensor[[PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16]],
    score: pl.Out[pl.Tensor[[MAX_TOKENS, INDEXER_SCORE_CAP], pl.FP32]],
    topk_idxs: pl.Out[pl.Tensor[[MAX_TOKENS, INDEXER_SCORE_CAP], pl.INT32]],
    token_to_request: pl.Tensor[[MAX_TOKENS], pl.INT32],
    position_ids: pl.Tensor[[MAX_TOKENS], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    num_cmp_writes: pl.Scalar[pl.INT32],
    cmp_write_token_ids: pl.Tensor[[MAX_CMP_WRITES], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[MAX_CMP_WRITES], pl.INT32],
):
    cmp_topk_indices = pl.create_tensor([MAX_TOKENS, IDX_TOPK], dtype=pl.INT32)
    idx_kv_cache_out, inner_kv_state_out, inner_score_state_out, cmp_topk_indices = prefill_indexer(
        x,
        freqs_cos,
        freqs_sin,
        hadamard,
        inner_kv_state,
        inner_score_state,
        inner_wkv,
        inner_wgate,
        inner_ape,
        inner_norm_w,
        idx_kv_cache,
        cmp_topk_indices,
        token_to_request,
        position_ids,
        num_tokens,
        num_cmp_writes,
        cmp_write_token_ids,
        cmp_slot_mapping,
    )
    idx_kv_cache = idx_kv_cache_out
    inner_kv_state = inner_kv_state_out
    inner_score_state = inner_score_state_out

    for score_block in pl.spmd(MAX_TOKENS // TOPK_TILE, name_hint="prefill_idx_score_topk_test"):
        score_t0 = score_block * TOPK_TILE
        for score_dt in pl.range(TOPK_TILE):
            score_token = score_t0 + score_dt
            score[score_token : score_token + 1, 0:INDEXER_SCORE_CAP] = pl.full(
                [1, INDEXER_SCORE_CAP],
                dtype=pl.FP32,
                value=0.0,
            )
            topk_idxs[score_token : score_token + 1, 0:INDEXER_SCORE_CAP] = pl.full(
                [1, INDEXER_SCORE_CAP],
                dtype=pl.INT32,
                value=-1,
            )
            if score_token < num_tokens:
                for topk_col in pl.range(IDX_TOPK):
                    topk_val = pl.read(cmp_topk_indices, [score_token, topk_col])
                    pl.write(topk_idxs, [score_token, topk_col], topk_val)
    idx_kv_cache_flat = pl.reshape(idx_kv_cache, [PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE, IDX_HEAD_DIM])
    for dep_block in pl.spmd(MAX_CMP_WRITES // TOPK_TILE, name_hint="prefill_idx_cache_test_dep"):
        dep_base = dep_block * TOPK_TILE
        for dep_dt in pl.range(TOPK_TILE):
            dep_i = dep_base + dep_dt
            if dep_i < num_cmp_writes:
                dep_row = pl.cast(pl.read(cmp_slot_mapping, [dep_i]), pl.INDEX)
                idx_kv_cache_flat[dep_row : dep_row + 1, 0:IDX_HEAD_DIM] = idx_kv_cache_flat[
                    dep_row : dep_row + 1,
                    0:IDX_HEAD_DIM,
                ]
    idx_kv_cache = pl.reshape(idx_kv_cache_flat, [PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM])
    return score, idx_kv_cache, topk_idxs


def build_tensor_specs(start_pos: int = START_POS):
    import torch
    from golden import ScalarSpec, TensorSpec

    num_tokens = T
    if start_pos < 0 or start_pos + MAX_TOKENS > MAX_SEQ_LEN:
        raise ValueError(f"start_pos must satisfy 0 <= start_pos <= {MAX_SEQ_LEN - MAX_TOKENS}, got {start_pos}")
    cmp_write_records = [
        (t, (start_pos + t + 1) // COMPRESS_RATIO - 1)
        for t in range(num_tokens)
        if (start_pos + t + 1) % COMPRESS_RATIO == 0
    ]
    if len(cmp_write_records) > MAX_CMP_WRITES:
        raise ValueError(f"fixture generated {len(cmp_write_records)} compressed writes, cap is {MAX_CMP_WRITES}")
    if cmp_write_records and max(dst_row for _, dst_row in cmp_write_records) >= PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE:
        raise ValueError("fixture compressed slot exceeds standalone idx_kv_cache capacity")

    def seeded_uniform(shape, seed, scale=1.0):
        generator = torch.Generator()
        generator.manual_seed(seed)
        return (torch.rand(*shape, generator=generator) - 0.5) * scale
    def init_x():
        return seeded_uniform((MAX_TOKENS, D), 1, 0.1).to(torch.bfloat16)
    def init_freqs_cos():
        return torch.cos(torch.arange(MAX_SEQ_LEN * ROPE_HEAD_DIM).reshape(MAX_SEQ_LEN, ROPE_HEAD_DIM) * 1e-3)
    def init_freqs_sin():
        return torch.sin(torch.arange(MAX_SEQ_LEN * ROPE_HEAD_DIM).reshape(MAX_SEQ_LEN, ROPE_HEAD_DIM) * 1e-3)
    def init_hadamard():
        h = torch.ones((1, 1))
        while h.shape[0] < IDX_HEAD_DIM:
            h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
        return (h * (IDX_HEAD_DIM ** -0.5)).to(torch.bfloat16)
    def init_inner_state():
        return torch.zeros(MAX_REQS, INNER_STATE_LEN, INNER_OUT_DIM)
    def init_inner_wkv():
        return seeded_uniform((D, INNER_OUT_DIM), 2, D ** -0.5).to(torch.bfloat16)
    def init_inner_wgate():
        return seeded_uniform((D, INNER_OUT_DIM), 3, D ** -0.5).to(torch.bfloat16)
    def init_inner_ape():
        return seeded_uniform((COMPRESS_RATIO, INNER_OUT_DIM), 4, 0.01)
    def init_inner_norm_w():
        return torch.ones(INNER_HEAD_DIM)
    def init_idx_kv_cache():
        return torch.zeros(PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM, dtype=torch.bfloat16)
    def init_token_to_request():
        return torch.zeros(MAX_TOKENS, dtype=torch.int32)
    def init_position_ids():
        return torch.arange(start_pos, start_pos + MAX_TOKENS, dtype=torch.int32)
    def init_cmp_write_token_ids():
        ids = torch.zeros(MAX_CMP_WRITES, dtype=torch.int32)
        for i, (token_id, _) in enumerate(cmp_write_records):
            ids[i] = token_id
        return ids
    def init_cmp_slot_mapping():
        mapping = torch.zeros(MAX_CMP_WRITES, dtype=torch.int32)
        for i, (_, dst_row) in enumerate(cmp_write_records):
            mapping[i] = dst_row
        return mapping

    return [
        TensorSpec("x", [MAX_TOKENS, D], torch.bfloat16, init_value=init_x),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("hadamard", [IDX_HEAD_DIM, IDX_HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        TensorSpec("inner_kv_state", [MAX_REQS, INNER_STATE_LEN, INNER_OUT_DIM], torch.float32, init_value=init_inner_state),
        TensorSpec("inner_score_state", [MAX_REQS, INNER_STATE_LEN, INNER_OUT_DIM], torch.float32, init_value=init_inner_state),
        TensorSpec("inner_wkv", [D, INNER_OUT_DIM], torch.bfloat16, init_value=init_inner_wkv),
        TensorSpec("inner_wgate", [D, INNER_OUT_DIM], torch.bfloat16, init_value=init_inner_wgate),
        TensorSpec("inner_ape", [COMPRESS_RATIO, INNER_OUT_DIM], torch.float32, init_value=init_inner_ape),
        TensorSpec("inner_norm_w", [INNER_HEAD_DIM], torch.float32, init_value=init_inner_norm_w),
        TensorSpec("idx_kv_cache", [PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], torch.bfloat16, init_value=init_idx_kv_cache, is_output=True),
        TensorSpec("score", [MAX_TOKENS, INDEXER_SCORE_CAP], torch.float32, is_output=True),
        TensorSpec("topk_idxs", [MAX_TOKENS, INDEXER_SCORE_CAP], torch.int32, is_output=True),
        TensorSpec("token_to_request", [MAX_TOKENS], torch.int32, init_value=init_token_to_request),
        TensorSpec("position_ids", [MAX_TOKENS], torch.int32, init_value=init_position_ids),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
        ScalarSpec("num_cmp_writes", torch.int32, len(cmp_write_records)),
        TensorSpec("cmp_write_token_ids", [MAX_CMP_WRITES], torch.int32, init_value=init_cmp_write_token_ids),
        TensorSpec("cmp_slot_mapping", [MAX_CMP_WRITES], torch.int32, init_value=init_cmp_slot_mapping),
    ]


if __name__ == "__main__":
    import argparse
    import torch
    from golden import ratio_allclose, run_jit, topk_pair_compare

    parser = argparse.ArgumentParser(description="Standalone token-major DeepSeek V4 prefill indexer validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--start-pos", type=int, default=START_POS,
                        help="Fixture-only absolute position for token 0; lowered into position_ids for the kernel.")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    def topk_idxs_compare(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        score = actual_outputs["score"]
        a_top = actual[..., :IDX_TOPK]
        e_top = expected[..., :IDX_TOPK]
        invalid_top = a_top < INDEXER_OFFSET
        a_orig = (a_top.long() - INDEXER_OFFSET).clamp(min=0, max=score.shape[-1] - 1)
        paired = torch.gather(score, dim=-1, index=a_orig)
        paired = torch.where(invalid_top, torch.full_like(paired, -torch.inf), paired)
        synth_actual = {**actual_outputs, "_topk_paired_scores": paired}
        return topk_pair_compare("_topk_paired_scores")(
            a_top, e_top,
            actual_outputs=synth_actual,
            expected_outputs=expected_outputs,
            inputs=inputs,
            rtol=rtol, atol=atol,
        )
    topk_idxs_compare.__name__ = "topk_pair_compare"

    result = run_jit(
        fn=prefill_indexer_test,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_prefill_indexer,
        runtime_cfg=dict(platform=args.platform, device_id=args.device, enable_l2_swimlane=args.enable_l2_swimlane),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "score": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "topk_idxs": topk_idxs_compare,
            "idx_kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
