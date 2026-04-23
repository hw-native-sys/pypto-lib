# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
DeepSeek V3.2-EXP single-layer decode FRONT — Scope 3: indexer score + topk.

Pipeline: scope1 (qkv proj/RoPE) → scope2 (indexer proj/RoPE + q_idx aggregate
+ write k_cache_idx) → scope3 (this file: score then topk).

Scoring uses the linear reduction
    score[b, s] = sum_h w[b, h] * (q[b, h] dot k[b, s])
                = (sum_h w[b, h] * q[b, h]) dot k[b, s]
                = q_idx[b] dot k_cache_idx[b, s]
so scope2 collapses 64 heads into a single query vector, and scope3 only has
to do [1, INDEX_HEAD_DIM] x [ctx_len, INDEX_HEAD_DIM] per batch.

Topk is done by sort32 + 4-way mrgsort merge, then gather to split sorted
(val, idx) pairs, then GM reload with valid_shape+fillpad to mark idx slots
past ctx_len. Outputs (topk_vals_out, topk_idx_out) = top-INDEX_TOPK entries
per batch; invalid tail idx = INT32_MIN (< 0), compatible with scope4's
`topk_pos >= 0` filter.
"""


import pypto.language as pl


BATCH = 16
MAX_SEQ = 4096
INDEX_HEAD_DIM = 128
INDEX_TOPK = 2048
CACHE_ROWS_IDX = BATCH * MAX_SEQ

SEQ_TILE = 64
MAX_SEQ_BLOCKS = (MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE

# Q pad: a2a3 TExtract requires row % 16 == 0, so pad the 1-row query to 16.
Q_VALID = 1
Q_PAD = 16

# sort32 + 4 mrgsort iterations (block_len 64,256,1024,4096) sort SORT_LEN=8192.
# SORT_LEN > MAX_SEQ so the full sort buffer is pre-filled with -inf (Stage 0)
# and only [0, ctx_len) contains real scores; tail stays -inf.
SORT_LEN = 8192
MRGSORT_ITERS = 4

# -inf sentinel for score tail. FP32 lowest, since ptoas rejects literal -inf.
FP32_NEG_INF = -3.4028234663852886e38


def build_deepseek_v3_2_decode_front_scope3_program():
    @pl.program
    class DeepSeekV32DecodeFrontScope3:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v3_2_decode_front_scope3(
            self,
            q_idx: pl.Tensor[[BATCH, INDEX_HEAD_DIM], pl.BF16],
            k_cache_idx: pl.Tensor[[CACHE_ROWS_IDX, INDEX_HEAD_DIM], pl.BF16],
            seq_lens: pl.Tensor[[BATCH], pl.INT32],
            idx_init: pl.Tensor[[1, SORT_LEN], pl.UINT32],
            topk_vals_out: pl.Tensor[[BATCH, INDEX_TOPK], pl.FP32],
            topk_idx_out: pl.Tensor[[BATCH, INDEX_TOPK], pl.INT32],
        ) -> tuple[
            pl.Tensor[[BATCH, INDEX_TOPK], pl.FP32],
            pl.Tensor[[BATCH, INDEX_TOPK], pl.INT32],
        ]:
            # Pad q_idx to [BATCH * Q_PAD, 128] with zero rows so QK matmul
            # has row=16 (required by a2a3 TExtract).
            q_padded = pl.create_tensor([BATCH * Q_PAD, INDEX_HEAD_DIM], dtype=pl.BF16)
            with pl.at(level=pl.Level.CORE_GROUP):
                for b in pl.range(BATCH):
                    q_row = pl.slice(q_idx, [1, INDEX_HEAD_DIM], [b, 0])
                    q_padded = pl.assemble(q_padded, q_row, [b * Q_PAD, 0])
                    q_padded = pl.assemble(
                        q_padded,
                        pl.cast(
                            pl.full(
                                [Q_PAD - Q_VALID, INDEX_HEAD_DIM],
                                dtype=pl.FP32,
                                value=0.0,
                            ),
                            target_type=pl.BF16,
                        ),
                        [b * Q_PAD + Q_VALID, 0],
                    )

            # Transient GM buffers.
            # scores is [BATCH, SORT_LEN]: Stage 0 fills the full row with -inf
            # so the [MAX_SEQ, SORT_LEN) tail is always -inf for the sort.
            scores = pl.create_tensor([BATCH, SORT_LEN], dtype=pl.FP32)
            sorted_gm = pl.create_tensor([BATCH, 2 * SORT_LEN], dtype=pl.FP32)
            raw_idx_gm = pl.create_tensor([BATCH, INDEX_TOPK], dtype=pl.INT32)

            for b in pl.range(0, BATCH, 1):
                ctx_len = pl.tensor.read(seq_lens, [b])
                ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE

                # Stage 0: pre-fill scores[b, 0:SORT_LEN] with -inf so both the
                # untouched ctx tail and the [MAX_SEQ, SORT_LEN) pad are -inf.
                with pl.at(level=pl.Level.CORE_GROUP):
                    neg_inf_row = pl.full([1, SORT_LEN], dtype=pl.FP32, value=FP32_NEG_INF)
                    scores = pl.assemble(scores, neg_inf_row, [b, 0])

                # Stage 1: tiled QK matmul into all_scores[sb*Q_PAD, 0] (row 0 valid).
                all_scores = pl.create_tensor(
                    [MAX_SEQ_BLOCKS * Q_PAD, SEQ_TILE], dtype=pl.FP32
                )
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for sb in pl.parallel(ctx_blocks, chunk=MAX_SEQ_BLOCKS):
                        s0 = sb * SEQ_TILE
                        cache_row0 = b * MAX_SEQ + s0
                        q_b = pl.slice(q_padded, [Q_PAD, INDEX_HEAD_DIM], [b * Q_PAD, 0])
                        k_tile = pl.slice(
                            k_cache_idx, [SEQ_TILE, INDEX_HEAD_DIM], [cache_row0, 0]
                        )
                        score_tile = pl.matmul(q_b, k_tile, b_trans=True, out_dtype=pl.FP32)
                        all_scores = pl.assemble(all_scores, score_tile, [sb * Q_PAD, 0])

                # Stage 2: fillpad each tile's tail and write row 0 to scores[b, s0].
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for sb in pl.parallel(ctx_blocks, chunk=MAX_SEQ_BLOCKS):
                        s0 = sb * SEQ_TILE
                        valid_len = pl.min(SEQ_TILE, ctx_len - s0)
                        tile_valid = pl.slice(
                            all_scores,
                            [1, SEQ_TILE],
                            [sb * Q_PAD, 0],
                            valid_shape=[1, valid_len],
                        )
                        tile_padded = pl.fillpad(tile_valid, pad_value=pl.PadValue.min)
                        scores = pl.assemble(scores, tile_padded, [b, s0])

                # Stage 3: sort32 + 4 mrgsort iterations (tensor-level). Operates
                # directly on GM slices; result is [1, 2*SORT_LEN] interleaved
                # (val, idx). Stored to sorted_gm for gather in Stage 4.
                with pl.at(level=pl.Level.CORE_GROUP):
                    score_row = pl.slice(scores, [1, SORT_LEN], [b, 0])
                    sorted_t = pl.tensor.sort32(score_row, idx_init)
                    sorted_t = pl.tensor.mrgsort(sorted_t, block_len=64)
                    sorted_t = pl.tensor.mrgsort(sorted_t, block_len=256)
                    sorted_t = pl.tensor.mrgsort(sorted_t, block_len=1024)
                    sorted_t = pl.tensor.mrgsort(sorted_t, block_len=4096)
                    sorted_gm = pl.assemble(sorted_gm, sorted_t, [b, 0])

                # Stage 4: gather P0101/P1010 to split vals / idx bits from the
                # first INDEX_TOPK pairs (2*INDEX_TOPK cols) in sorted_gm.
                with pl.at(level=pl.Level.CORE_GROUP):
                    topk_pairs = pl.slice(sorted_gm, [1, 2 * INDEX_TOPK], [b, 0])
                    topk_v = pl.tensor.gather(
                        topk_pairs, mask_pattern=pl.tile.MaskPattern.P0101
                    )
                    topk_i_raw = pl.tensor.gather(
                        topk_pairs,
                        mask_pattern=pl.tile.MaskPattern.P1010,
                        output_dtype=pl.INT32,
                    )
                    topk_vals_out = pl.assemble(topk_vals_out, topk_v, [b, 0])
                    raw_idx_gm = pl.assemble(raw_idx_gm, topk_i_raw, [b, 0])

                # Stage 5: GM reload + valid_shape fillpad to mark idx slots past
                # ctx_len with PadValue.min (= INT32_MIN < 0).
                with pl.at(level=pl.Level.CORE_GROUP):
                    valid_topk = pl.min(INDEX_TOPK, ctx_len)
                    idx_valid = pl.slice(
                        raw_idx_gm,
                        [1, INDEX_TOPK],
                        [b, 0],
                        valid_shape=[1, valid_topk],
                    )
                    idx_padded = pl.fillpad(idx_valid, pad_value=pl.PadValue.min)
                    topk_idx_out = pl.assemble(topk_idx_out, idx_padded, [b, 0])

            return topk_vals_out, topk_idx_out

    return DeepSeekV32DecodeFrontScope3


def golden_decode_front_scope3(tensors):
    import torch  # type: ignore[import]

    q_idx = tensors["q_idx"].float()
    k_cache_idx = tensors["k_cache_idx"].float()
    seq_lens = tensors["seq_lens"]
    topk_vals_out = tensors["topk_vals_out"]
    topk_idx_out = tensors["topk_idx_out"]

    scores = torch.full((BATCH, SORT_LEN), FP32_NEG_INF, dtype=torch.float32)
    for b in range(BATCH):
        ctx_len = int(seq_lens[b].item())
        q_b = q_idx[b : b + 1]
        k_b = k_cache_idx[b * MAX_SEQ : b * MAX_SEQ + ctx_len]
        scores[b, :ctx_len] = (q_b @ k_b.T).squeeze(0)

    # Rare BF16 ties can swap adjacent idx entries between kernel and
    # torch.sort; vals stay identical so downstream attention is unaffected.
    vals, idx = torch.topk(scores, INDEX_TOPK, dim=1, largest=True, sorted=True)
    topk_vals_out.copy_(vals)
    # Kernel fillpads idx tail past ctx_len with INT32_MIN.
    idx = idx.to(torch.int32)
    for b in range(BATCH):
        ctx_len = int(seq_lens[b].item())
        valid_topk = min(INDEX_TOPK, ctx_len)
        idx[b, valid_topk:] = torch.iinfo(torch.int32).min
    topk_idx_out.copy_(idx)


def build_tensor_specs():
    import torch  # type: ignore[import]
    from golden import TensorSpec

    # ctx_len in [1, MAX_SEQ]; kernel pads idx tail past ctx_len with
    # PadValue.min (= INT32_MIN), and scope4 filters on `topk_pos >= 0`.
    seq_lens_data = torch.randint(1, MAX_SEQ + 1, (BATCH,), dtype=torch.int32)

    def init_q_idx():
        return torch.rand(BATCH, INDEX_HEAD_DIM) - 0.5

    def init_k_cache_idx():
        return torch.rand(CACHE_ROWS_IDX, INDEX_HEAD_DIM) - 0.5

    def init_idx_init():
        return torch.arange(SORT_LEN, dtype=torch.int32).unsqueeze(0)

    def init_topk_vals_out():
        return torch.zeros((BATCH, INDEX_TOPK), dtype=torch.float32)

    def init_topk_idx_out():
        return torch.zeros((BATCH, INDEX_TOPK), dtype=torch.int32)

    return [
        TensorSpec("q_idx", [BATCH, INDEX_HEAD_DIM], torch.bfloat16, init_value=init_q_idx),
        TensorSpec(
            "k_cache_idx",
            [CACHE_ROWS_IDX, INDEX_HEAD_DIM],
            torch.bfloat16,
            init_value=init_k_cache_idx,
        ),
        TensorSpec("seq_lens", [BATCH], torch.int32, init_value=seq_lens_data),
        TensorSpec("idx_init", [1, SORT_LEN], torch.int32, init_value=init_idx_init),
        TensorSpec(
            "topk_vals_out",
            [BATCH, INDEX_TOPK],
            torch.float32,
            init_value=init_topk_vals_out,
            is_output=True,
        ),
        TensorSpec(
            "topk_idx_out",
            [BATCH, INDEX_TOPK],
            torch.int32,
            init_value=init_topk_idx_out,
            is_output=True,
        ),
    ]


if __name__ == "__main__":
    import argparse
    from golden import RunConfig, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        program=build_deepseek_v3_2_decode_front_scope3_program(),
        tensor_specs=build_tensor_specs(),
        golden_fn=golden_decode_front_scope3,
        config=RunConfig(
            rtol=1e-3,
            atol=1e-3,
            compile=dict(dump_passes=True),
            runtime=dict(
                platform=args.platform,
                device_id=args.device,
                runtime_profiling=args.runtime_profiling,
            ),
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
