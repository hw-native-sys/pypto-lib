# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Row-wise top-k via the ``sort32`` + ``mrgsort`` instruction combo.

``sort32`` sorts 32-element runs into ``(value, index)`` pairs; chained
``mrgsort`` 4-way-merges them until the whole row is sorted (runs of
64 -> 256 positions for N=512). The leading ``2*K`` pairs are split with a
mask ``gather``: ``P0101`` lifts the value lanes, ``P1010`` the index lanes.
Both primitives need a single row, so the ``pl.spmd`` block loops rows with
``pl.range``.
"""

import pypto.language as pl

ROWS = 64
N = 512                 # columns sorted per row
K = 16                  # top-k width
ROW_TILE = 8            # rows handled per pl.spmd block


@pl.jit
def topk(
    scores: pl.Tensor[[ROWS, N], pl.FP32],
    topk_vals: pl.Out[pl.Tensor[[ROWS, K], pl.FP32]],
    topk_idx: pl.Out[pl.Tensor[[ROWS, K], pl.INT32]],
):
    for blk in pl.spmd(ROWS // ROW_TILE, name_hint="topk_block"):
        r0 = blk * ROW_TILE
        idx_init = pl.arange(0, [1, N], dtype=pl.UINT32)
        for ri in pl.range(ROW_TILE):
            r = r0 + ri
            score_row = scores[r : r + 1, :]
            s = pl.sort32(score_row, idx_init)                      # [1, 2N], 32-runs sorted
            s = pl.mrgsort(s, block_len=64)                         # 4-way merge -> 256-pos runs
            s = pl.mrgsort(s, block_len=256)                        # 4-way merge -> whole row, descending
            pairs = s[:, 0 : 2 * K]                                 # K largest (value, index) pairs
            topk_vals[r : r + 1, :] = pl.gather(pairs, mask_pattern=pl.tile.MaskPattern.P0101)
            topk_idx[r : r + 1, :] = pl.gather(
                pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
            )
    return topk_vals, topk_idx


def build_tensor_specs():
    import torch

    from golden import TensorSpec

    return [
        TensorSpec("scores",    [ROWS, N], torch.float32, init_value=torch.randn),
        TensorSpec("topk_vals", [ROWS, K], torch.float32, is_output=True),
        TensorSpec("topk_idx",  [ROWS, K], torch.int32,   is_output=True),
    ]


def golden_topk(tensors):
    import torch

    vals, idx = torch.topk(tensors["scores"].float(), K, dim=-1, largest=True, sorted=True)
    tensors["topk_vals"][:] = vals
    tensors["topk_idx"][:] = idx.to(torch.int32)


if __name__ == "__main__":
    import argparse

    from golden import run_jit, topk_pair_compare

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=topk,
        specs=build_tensor_specs(),
        golden_fn=golden_topk,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-5,
        atol=1e-5,
        # Tie scores can swap legal indices; adjudicate idx via the paired vals.
        compare_fn={"topk_idx": topk_pair_compare("topk_vals")},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
