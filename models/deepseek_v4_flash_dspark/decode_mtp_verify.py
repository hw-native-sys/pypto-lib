# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device-side DeepSeek-V4 MTP draft verification and committed-window packing."""

import pypto.language as pl

from config import DECODE_BATCH, DECODE_SEQ, TP
from lm_head import MAX_LOGIT_ROWS, SAMPLED_IDS_PAD


B = DECODE_BATCH // TP
S = DECODE_SEQ
T = B * S

assert S == 2, "MTP verification requires decode_seq=2"
assert MAX_LOGIT_ROWS >= T


@pl.jit.inline
def verify_and_pack_mtp_tokens(
    main_input_ids: pl.Tensor[[T], pl.INT64],
    main_position_ids: pl.Tensor[[T], pl.INT32],
    main_sampled_ids: pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32],
    tail_token_ids: pl.Tensor[[B], pl.INT64],
    tail_positions: pl.Tensor[[B], pl.INT32],
    tail_slot_ids: pl.Tensor[[B], pl.INT32],
    mtp_input_ids: pl.Tensor[[T], pl.INT64],
    mtp_position_ids: pl.Tensor[[T], pl.INT32],
    accepted_counts: pl.Tensor[[B], pl.INT32],
):
    """Verify one draft per row and pack the two-token MTP committed window."""
    # Keep the tightly packed scalar stores on one core. Multiple SPMD cores
    # writing adjacent scalar addresses can race through overlapping DMA units.
    for verify_core in pl.spmd(1, name_hint="mtp_verify_and_pack"):
        for request in pl.range(verify_core, B):
            row0 = request * S
            row1 = row0 + 1
            slot = pl.read(tail_slot_ids, [request])
            if slot >= 0:
                draft = pl.read(main_input_ids, [row1])
                main0 = pl.cast(pl.read(main_sampled_ids, [row0, 0]), pl.INT64)
                main1 = pl.cast(pl.read(main_sampled_ids, [row1, 0]), pl.INT64)
                accepted = pl.cast(1, pl.INT32)
                committed0 = pl.read(tail_token_ids, [request])
                committed1 = main0
                position0 = pl.read(tail_positions, [request])
                position1 = pl.read(main_position_ids, [row0])
                if draft == main0:
                    accepted = pl.cast(2, pl.INT32)
                    committed0 = main0
                    committed1 = main1
                    position0 = pl.read(main_position_ids, [row0])
                    position1 = pl.read(main_position_ids, [row1])
                pl.write(accepted_counts, [request], accepted)
                pl.write(mtp_input_ids, [row0], committed0)
                pl.write(mtp_input_ids, [row1], committed1)
                pl.write(mtp_position_ids, [row0], position0)
                pl.write(mtp_position_ids, [row1], position1)
            else:
                pl.write(accepted_counts, [request], pl.cast(1, pl.INT32))
                pl.write(mtp_input_ids, [row0], pl.read(main_input_ids, [row0]))
                pl.write(mtp_input_ids, [row1], pl.read(main_input_ids, [row1]))
                pl.write(
                    mtp_position_ids,
                    [row0],
                    pl.read(main_position_ids, [row0]),
                )
                pl.write(
                    mtp_position_ids,
                    [row1],
                    pl.read(main_position_ids, [row1]),
                )
    return mtp_input_ids, mtp_position_ids, accepted_counts


@pl.jit
def mtp_verify_and_pack(
    main_input_ids: pl.Tensor[[T], pl.INT64],
    main_position_ids: pl.Tensor[[T], pl.INT32],
    main_sampled_ids: pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32],
    tail_token_ids: pl.Tensor[[B], pl.INT64],
    tail_positions: pl.Tensor[[B], pl.INT32],
    tail_slot_ids: pl.Tensor[[B], pl.INT32],
    mtp_input_ids: pl.Out[pl.Tensor[[T], pl.INT64]],
    mtp_position_ids: pl.Out[pl.Tensor[[T], pl.INT32]],
    accepted_counts: pl.Out[pl.Tensor[[B], pl.INT32]],
):
    """Standalone validation entry for device-side MTP verification."""
    return verify_and_pack_mtp_tokens(
        main_input_ids,
        main_position_ids,
        main_sampled_ids,
        tail_token_ids,
        tail_positions,
        tail_slot_ids,
        mtp_input_ids,
        mtp_position_ids,
        accepted_counts,
    )


def _test_inputs():
    import torch

    # Four acceptance regimes, tiled up to B: draft accepted, draft rejected,
    # draft accepted, and no tail slot at all.
    regime_input = torch.tensor([10, 20, 11, 21, 12, 22, 13, 23], dtype=torch.int64)
    regime_sampled = torch.tensor([20, 30, 99, 31, 22, 32, 23, 33], dtype=torch.int32)
    regime_tail_tokens = torch.tensor([9, 8, 7, 6], dtype=torch.int64)
    reps = (B + 3) // 4

    main_input_ids = regime_input.repeat(reps)[:T]
    main_sampled_ids = torch.zeros((MAX_LOGIT_ROWS, SAMPLED_IDS_PAD), dtype=torch.int32)
    main_sampled_ids[:T, 0] = regime_sampled.repeat(reps)[:T]
    # One distinct position base per request; the tail sits one step before it.
    position_bases = (torch.arange(B, dtype=torch.int32) + 1) * 100
    main_position_ids = torch.stack([position_bases, position_bases + 1], dim=1).reshape(T)
    # Every request owns its own tail slot; the 4th regime has none.
    tail_slot_ids = torch.arange(B, dtype=torch.int32)
    tail_slot_ids[3::4] = -1
    return {
        "main_input_ids": main_input_ids,
        "main_position_ids": main_position_ids,
        "main_sampled_ids": main_sampled_ids,
        "tail_token_ids": regime_tail_tokens.repeat(reps)[:B],
        "tail_positions": position_bases - 1,
        "tail_slot_ids": tail_slot_ids,
    }


def golden_mtp_verify_and_pack(tensors):
    for request in range(B):
        row0 = request * S
        row1 = row0 + 1
        if int(tensors["tail_slot_ids"][request]) < 0:
            tensors["accepted_counts"][request] = 1
            tensors["mtp_input_ids"][row0 : row1 + 1].copy_(tensors["main_input_ids"][row0 : row1 + 1])
            tensors["mtp_position_ids"][row0 : row1 + 1].copy_(tensors["main_position_ids"][row0 : row1 + 1])
            continue
        draft = int(tensors["main_input_ids"][row1])
        main0 = int(tensors["main_sampled_ids"][row0, 0])
        main1 = int(tensors["main_sampled_ids"][row1, 0])
        if draft == main0:
            tensors["accepted_counts"][request] = 2
            tensors["mtp_input_ids"][row0] = main0
            tensors["mtp_input_ids"][row1] = main1
            tensors["mtp_position_ids"][row0 : row1 + 1].copy_(tensors["main_position_ids"][row0 : row1 + 1])
        else:
            tensors["accepted_counts"][request] = 1
            tensors["mtp_input_ids"][row0] = tensors["tail_token_ids"][request]
            tensors["mtp_input_ids"][row1] = main0
            tensors["mtp_position_ids"][row0] = tensors["tail_positions"][request]
            tensors["mtp_position_ids"][row1] = tensors["main_position_ids"][row0]


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    inputs = _test_inputs()
    specs = [
        TensorSpec(name, list(value.shape), value.dtype, init_value=value) for name, value in inputs.items()
    ]
    specs.extend(
        (
            TensorSpec("mtp_input_ids", [T], torch.int64, is_output=True),
            TensorSpec("mtp_position_ids", [T], torch.int32, is_output=True),
            TensorSpec("accepted_counts", [B], torch.int32, is_output=True),
        )
    )
    return specs


if __name__ == "__main__":
    import argparse

    from golden import run_jit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-p",
        "--platform",
        default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    result = run_jit(
        fn=mtp_verify_and_pack,
        specs=build_tensor_specs(),
        golden_fn=golden_mtp_verify_and_pack,
        compile_only=args.compile_only,
        runtime_cfg={"platform": args.platform, "device_id": args.device},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
