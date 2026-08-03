# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Standalone device operators for DeepSeek-V4 Flash decode input packing."""

import pypto.language as pl

from config import DECODE_BATCH, DECODE_SEQ, DECODE_TOKENS, FLASH as M
from lm_head import SAMPLED_IDS_PAD


VOCAB_DYN = pl.dynamic("PACK_X_HC_VOCAB_DYN")

D = M.hidden_size
HC_MULT = M.hc_mult

X_HC_HIDDEN_TILE = 512
MTP_HIDDEN_TILE = 1024
SPMD_BLOCKS = 48

assert DECODE_SEQ == 2, "pack_mtp_hidden requires decode_seq=2"
assert D % X_HC_HIDDEN_TILE == 0
assert D % MTP_HIDDEN_TILE == 0


@pl.jit
def pack_mtp_inputs(
    main_sampled_ids: pl.Tensor[[DECODE_TOKENS, SAMPLED_IDS_PAD], pl.INT32],
    main_position_ids: pl.Tensor[[DECODE_TOKENS], pl.INT32],
    accepted_counts: pl.Tensor[[DECODE_BATCH], pl.INT32],
    tail_slot_ids: pl.Tensor[[DECODE_BATCH], pl.INT32],
    tail_token_pool: pl.InOut[pl.Tensor[[DECODE_BATCH], pl.INT64]],
    tail_position_pool: pl.InOut[pl.Tensor[[DECODE_BATCH], pl.INT32]],
    mtp_input_ids: pl.Out[pl.Tensor[[DECODE_TOKENS], pl.INT64]],
    mtp_position_ids: pl.Out[pl.Tensor[[DECODE_TOKENS], pl.INT32]],
):
    for batch_core in pl.spmd(
        1,
        name_hint="pack_mtp_inputs",
    ):
        for batch_idx in pl.range(batch_core, DECODE_BATCH):
            row0 = batch_idx * DECODE_SEQ
            row1 = row0 + 1
            slot = pl.cast(
                pl.read(tail_slot_ids, [batch_idx]),
                target_type=pl.INDEX,
            )
            accepted_count = pl.read(accepted_counts, [batch_idx])
            sampled0 = pl.cast(
                pl.read(main_sampled_ids, [row0, 0]),
                target_type=pl.INT64,
            )
            sampled1 = pl.cast(
                pl.read(main_sampled_ids, [row1, 0]),
                target_type=pl.INT64,
            )
            position0 = pl.read(main_position_ids, [row0])
            position1 = pl.read(main_position_ids, [row1])
            if accepted_count == 1:
                pl.write(
                    mtp_input_ids,
                    [row0],
                    pl.read(tail_token_pool, [slot]),
                )
                pl.write(
                    mtp_position_ids,
                    [row0],
                    pl.read(tail_position_pool, [slot]),
                )
                pl.write(mtp_input_ids, [row1], sampled0)
                pl.write(mtp_position_ids, [row1], position0)
                pl.write(tail_token_pool, [slot], sampled0)
                pl.write(tail_position_pool, [slot], position0)
            else:
                pl.write(mtp_input_ids, [row0], sampled0)
                pl.write(mtp_position_ids, [row0], position0)
                pl.write(mtp_input_ids, [row1], sampled1)
                pl.write(mtp_position_ids, [row1], position1)
                pl.write(tail_token_pool, [slot], sampled1)
                pl.write(tail_position_pool, [slot], position1)
    return mtp_input_ids, mtp_position_ids


@pl.jit.inline
def pack_x_hc(
    input_ids: pl.Tensor[[DECODE_TOKENS], pl.INT64],
    embed_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    x_hc: pl.Tensor[[DECODE_TOKENS, HC_MULT, D], pl.FP32],
) -> pl.Tensor[[DECODE_TOKENS, HC_MULT, D], pl.FP32]:
    x_hc_flat = pl.reshape(x_hc, [DECODE_TOKENS * HC_MULT, D])
    for block in pl.spmd(SPMD_BLOCKS, name_hint="pack_x_hc"):
        for work_idx in pl.range(
            block,
            DECODE_TOKENS * (D // X_HC_HIDDEN_TILE),
            SPMD_BLOCKS,
        ):
            token_idx = work_idx // (D // X_HC_HIDDEN_TILE)
            hidden_offset = (work_idx % (D // X_HC_HIDDEN_TILE)) * X_HC_HIDDEN_TILE
            token_id = pl.tensor.read(input_ids, [token_idx])
            token_row = pl.cast(token_id, target_type=pl.INDEX)
            hidden_chunk = pl.cast(
                embed_weight[
                    token_row : token_row + 1,
                    hidden_offset : hidden_offset + X_HC_HIDDEN_TILE,
                ],
                target_type=pl.FP32,
            )
            for hc_idx in pl.range(HC_MULT):
                x_hc_row = token_idx * HC_MULT + hc_idx
                x_hc_flat[
                    x_hc_row : x_hc_row + 1,
                    hidden_offset : hidden_offset + X_HC_HIDDEN_TILE,
                ] = hidden_chunk
    return x_hc


@pl.jit.inline
def pack_mtp_hidden(
    main_pre_hc_hidden: pl.Tensor[[DECODE_TOKENS, HC_MULT, D], pl.FP32],
    tail_pre_hc_pool: pl.Tensor[[DECODE_BATCH, HC_MULT, D], pl.FP32],
    accepted_counts: pl.Tensor[[DECODE_BATCH], pl.INT32],
    tail_slot_ids: pl.Tensor[[DECODE_BATCH], pl.INT32],
    fallback_hidden: pl.Tensor[[DECODE_SEQ, HC_MULT, D], pl.FP32],
    packed_hidden: pl.Tensor[[DECODE_TOKENS, HC_MULT, D], pl.FP32],
) -> pl.Tensor[[DECODE_TOKENS, HC_MULT, D], pl.FP32]:
    for block in pl.spmd(SPMD_BLOCKS, name_hint="pack_mtp_hidden"):
        for work_idx in pl.range(
            block,
            DECODE_BATCH * HC_MULT * (D // MTP_HIDDEN_TILE),
            SPMD_BLOCKS,
        ):
            batch_idx = work_idx // (HC_MULT * (D // MTP_HIDDEN_TILE))
            local_idx = work_idx % (HC_MULT * (D // MTP_HIDDEN_TILE))
            hc_idx = local_idx // (D // MTP_HIDDEN_TILE)
            hidden_offset = (local_idx % (D // MTP_HIDDEN_TILE)) * MTP_HIDDEN_TILE
            row0 = batch_idx * DECODE_SEQ
            row1 = row0 + 1

            slot_raw = pl.read(tail_slot_ids, [batch_idx])
            if slot_raw >= 0:
                accepted_count = pl.read(accepted_counts, [batch_idx])
                last_row = row0 + pl.cast(accepted_count, target_type=pl.INDEX) - 1
                last_hidden = main_pre_hc_hidden[
                    last_row : last_row + 1,
                    hc_idx : hc_idx + 1,
                    hidden_offset : hidden_offset + MTP_HIDDEN_TILE,
                ]
                slot = pl.cast(slot_raw, target_type=pl.INDEX)
                if accepted_count == 1:
                    packed_hidden[
                        row0 : row0 + 1,
                        hc_idx : hc_idx + 1,
                        hidden_offset : hidden_offset + MTP_HIDDEN_TILE,
                    ] = tail_pre_hc_pool[
                        slot : slot + 1,
                        hc_idx : hc_idx + 1,
                        hidden_offset : hidden_offset + MTP_HIDDEN_TILE,
                    ]
                else:
                    packed_hidden[
                        row0 : row0 + 1,
                        hc_idx : hc_idx + 1,
                        hidden_offset : hidden_offset + MTP_HIDDEN_TILE,
                    ] = main_pre_hc_hidden[
                        row0 : row0 + 1,
                        hc_idx : hc_idx + 1,
                        hidden_offset : hidden_offset + MTP_HIDDEN_TILE,
                    ]
                packed_hidden[
                    row1 : row1 + 1,
                    hc_idx : hc_idx + 1,
                    hidden_offset : hidden_offset + MTP_HIDDEN_TILE,
                ] = last_hidden
                tail_pre_hc_pool[
                    slot : slot + 1,
                    hc_idx : hc_idx + 1,
                    hidden_offset : hidden_offset + MTP_HIDDEN_TILE,
                ] = last_hidden
            else:
                for seq_idx in pl.range(DECODE_SEQ):
                    packed_hidden[
                        row0 + seq_idx : row0 + seq_idx + 1,
                        hc_idx : hc_idx + 1,
                        hidden_offset : hidden_offset + MTP_HIDDEN_TILE,
                    ] = fallback_hidden[
                        seq_idx : seq_idx + 1,
                        hc_idx : hc_idx + 1,
                        hidden_offset : hidden_offset + MTP_HIDDEN_TILE,
                    ]
    return packed_hidden


def golden_pack_mtp_inputs(tensors):
    sampled_ids = tensors["main_sampled_ids"][:, 0].to(dtype=tensors["tail_token_pool"].dtype)
    for batch_idx in range(DECODE_BATCH):
        row0 = batch_idx * DECODE_SEQ
        row1 = row0 + 1
        slot = int(tensors["tail_slot_ids"][batch_idx])
        accepted_count = int(tensors["accepted_counts"][batch_idx])
        if accepted_count == 1:
            tensors["mtp_input_ids"][row0] = tensors["tail_token_pool"][slot]
            tensors["mtp_position_ids"][row0] = tensors["tail_position_pool"][slot]
            tensors["mtp_input_ids"][row1] = sampled_ids[row0]
            tensors["mtp_position_ids"][row1] = tensors["main_position_ids"][row0]
            tensors["tail_token_pool"][slot] = sampled_ids[row0]
            tensors["tail_position_pool"][slot] = tensors["main_position_ids"][row0]
        else:
            tensors["mtp_input_ids"][row0] = sampled_ids[row0]
            tensors["mtp_position_ids"][row0] = tensors["main_position_ids"][row0]
            tensors["mtp_input_ids"][row1] = sampled_ids[row1]
            tensors["mtp_position_ids"][row1] = tensors["main_position_ids"][row1]
            tensors["tail_token_pool"][slot] = sampled_ids[row1]
            tensors["tail_position_pool"][slot] = tensors["main_position_ids"][row1]


def validate_handoff_fixture(
    accepted_counts,
    tail_slot_ids,
    main_sampled_ids,
    tail_token_pool,
):
    if not bool(((accepted_counts == 1) | (accepted_counts == 2)).all()):
        raise ValueError("accepted counts must be 1 or 2")
    slots = tail_slot_ids.tolist()
    if len(set(slots)) != DECODE_BATCH:
        raise ValueError("tail slots must be unique")
    if not all(0 <= slot < DECODE_BATCH for slot in slots):
        raise ValueError("tail slots must be in the decode batch")
    sampled_tokens = main_sampled_ids[:, 0]
    if not bool(((sampled_tokens >= 0) & (sampled_tokens < M.vocab_size)).all()):
        raise ValueError("sampled tokens must be in the model vocabulary")
    if not bool(((tail_token_pool >= 0) & (tail_token_pool < M.vocab_size)).all()):
        raise ValueError("tail tokens must be in the model vocabulary")


def _build_handoff_fixture():
    import torch

    sampled = torch.full((8, 8), -777, dtype=torch.int32)
    sampled[:, 0] = torch.tensor(
        [100, 101, 200, 201, 300, 301, 400, 401],
        dtype=torch.int32,
    )
    return {
        "main_sampled_ids": sampled,
        "main_position_ids": torch.tensor(
            [10, 11, 20, 21, 30, 31, 40, 41],
            dtype=torch.int32,
        ),
        "accepted_counts": torch.tensor([1, 2, 1, 2], dtype=torch.int32),
        "tail_slot_ids": torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        "tail_token_pool": torch.tensor([900, 901, 902, 903], dtype=torch.int64),
        "tail_position_pool": torch.tensor([9, 19, 29, 39], dtype=torch.int32),
    }


def build_handoff_tensor_specs():
    import torch
    from golden import TensorSpec

    fixture = _build_handoff_fixture()
    return [
        TensorSpec("main_sampled_ids", [8, 8], torch.int32, init_value=fixture["main_sampled_ids"]),
        TensorSpec("main_position_ids", [8], torch.int32, init_value=fixture["main_position_ids"]),
        TensorSpec("accepted_counts", [4], torch.int32, init_value=fixture["accepted_counts"]),
        TensorSpec("tail_slot_ids", [4], torch.int32, init_value=fixture["tail_slot_ids"]),
        TensorSpec(
            "tail_token_pool",
            [4],
            torch.int64,
            init_value=fixture["tail_token_pool"],
            is_output=True,
        ),
        TensorSpec(
            "tail_position_pool",
            [4],
            torch.int32,
            init_value=fixture["tail_position_pool"],
            is_output=True,
        ),
        TensorSpec("mtp_input_ids", [8], torch.int64, is_output=True),
        TensorSpec("mtp_position_ids", [8], torch.int32, is_output=True),
    ]


if __name__ == "__main__":
    import argparse

    from golden import run_jit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-p",
        "--platform",
        type=str,
        default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    fixture = _build_handoff_fixture()
    validate_handoff_fixture(
        fixture["accepted_counts"],
        fixture["tail_slot_ids"],
        fixture["main_sampled_ids"],
        fixture["tail_token_pool"],
    )
    result = run_jit(
        fn=pack_mtp_inputs,
        specs=build_handoff_tensor_specs(),
        golden_fn=golden_pack_mtp_inputs,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
        ),
        rtol=0,
        atol=0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
