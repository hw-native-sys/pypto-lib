# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Request-slot token feedback for Qwen decode."""

from __future__ import annotations

import pypto.language as pl

from config import QWEN3_14B as M

# Dynamic shape variables.
BATCH_DYN = pl.dynamic("QWEN_TOKEN_BATCH_DYN")
STATE_SLOTS_DYN = pl.dynamic("QWEN_TOKEN_STATE_SLOTS_DYN")

# model config
ROW_WIDTH = M.sampled_ids_pad

# State row: token, generation, valid.
STATE_TOKEN = 0
STATE_GENERATION = 1
STATE_VALID = 2

# Control row: slot, generation, use_feedback, active, seed_token.
CONTROL_SLOT = 0
CONTROL_GENERATION = 1
CONTROL_FEEDBACK = 2
CONTROL_ACTIVE = 3
CONTROL_SEED = 4

# Token/snapshot row: token, valid, generation.
TOKEN_VALID = 1
TOKEN_GENERATION = 2


@pl.jit.inline
def load_request_tokens(
    state: pl.Tensor[[STATE_SLOTS_DYN, ROW_WIDTH], pl.INT32],
    control: pl.Tensor[[BATCH_DYN, ROW_WIDTH], pl.INT32],
    tokens: pl.Tensor[[BATCH_DYN, ROW_WIDTH], pl.INT32],
) -> pl.Tensor[[BATCH_DYN, ROW_WIDTH], pl.INT32]:
    batch = pl.tensor.dim(control, 0)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="request_token_load"):
        for row in pl.range(batch):
            generation = pl.read(control, [row, CONTROL_GENERATION])
            token = pl.read(control, [row, CONTROL_SEED])
            valid = pl.cast(1, pl.INT32)
            feedback = pl.read(control, [row, CONTROL_FEEDBACK])
            if feedback == 1:
                slot = pl.cast(pl.read(control, [row, CONTROL_SLOT]), pl.INDEX)
                saved = state[slot : slot + 1, :]
                saved_generation = pl.read(saved, [0, STATE_GENERATION])
                saved_valid = pl.read(saved, [0, STATE_VALID])
                token = pl.cast(0, pl.INT32)
                valid = pl.cast(0, pl.INT32)
                if saved_generation == generation:
                    if saved_valid == 1:
                        token = pl.read(saved, [0, STATE_TOKEN])
                        valid = pl.cast(1, pl.INT32)
            token_row = pl.full([1, ROW_WIDTH], dtype=pl.INT32, value=0)
            pl.write(token_row, [0, 0], token)
            pl.write(token_row, [0, TOKEN_VALID], valid)
            pl.write(token_row, [0, TOKEN_GENERATION], generation)
            tokens[row : row + 1, :] = token_row
    return tokens


@pl.jit.inline
def commit_request_tokens(
    state: pl.Tensor[[STATE_SLOTS_DYN, ROW_WIDTH], pl.INT32],
    control: pl.Tensor[[BATCH_DYN, ROW_WIDTH], pl.INT32],
    tokens: pl.Tensor[[BATCH_DYN, ROW_WIDTH], pl.INT32],
    sampled_ids: pl.Tensor[[BATCH_DYN, ROW_WIDTH], pl.INT32],
) -> tuple[
    pl.Tensor[[STATE_SLOTS_DYN, ROW_WIDTH], pl.INT32],
    pl.Tensor[[BATCH_DYN, ROW_WIDTH], pl.INT32],
]:
    batch = pl.tensor.dim(control, 0)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="request_token_commit"):
        for row in pl.range(batch):
            active = pl.read(control, [row, CONTROL_ACTIVE])
            generation = pl.read(control, [row, CONTROL_GENERATION])
            valid = pl.read(tokens, [row, TOKEN_VALID])
            snapshot = sampled_ids[row : row + 1, :]
            pl.write(snapshot, [0, TOKEN_VALID], valid)
            pl.write(snapshot, [0, TOKEN_GENERATION], generation)
            sampled_ids[row : row + 1, :] = snapshot
            if active == 1:
                if valid == 1:
                    slot = pl.cast(pl.read(control, [row, CONTROL_SLOT]), pl.INDEX)
                    token = pl.read(snapshot, [0, 0])
                    next_state = pl.full([1, ROW_WIDTH], dtype=pl.INT32, value=0)
                    pl.write(next_state, [0, STATE_TOKEN], token)
                    pl.write(next_state, [0, STATE_GENERATION], generation)
                    pl.write(next_state, [0, STATE_VALID], pl.cast(1, pl.INT32))
                    state[slot : slot + 1, :] = next_state
    return state, sampled_ids


@pl.jit
def request_token_state_fwd(
    state: pl.InOut[pl.Tensor[[STATE_SLOTS_DYN, ROW_WIDTH], pl.INT32]],
    control: pl.Tensor[[BATCH_DYN, ROW_WIDTH], pl.INT32],
    next_sample: pl.Tensor[[BATCH_DYN, ROW_WIDTH], pl.INT32],
    tokens: pl.Out[pl.Tensor[[BATCH_DYN, ROW_WIDTH], pl.INT32]],
    snapshot: pl.Out[pl.Tensor[[BATCH_DYN, ROW_WIDTH], pl.INT32]],
):
    state.bind_dynamic(0, STATE_SLOTS_DYN)
    control.bind_dynamic(0, BATCH_DYN)
    next_sample.bind_dynamic(0, BATCH_DYN)
    tokens.bind_dynamic(0, BATCH_DYN)
    snapshot.bind_dynamic(0, BATCH_DYN)
    tokens = load_request_tokens(state, control, tokens)
    batch = pl.tensor.dim(control, 0)
    for row in pl.parallel(batch):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="request_token_snapshot"):
            snapshot[row : row + 1, :] = next_sample[row : row + 1, :]
    state, snapshot = commit_request_tokens(state, control, tokens, snapshot)
    return state, tokens, snapshot


def build_tensor_specs(case):
    import torch

    from golden import TensorSpec

    state = torch.zeros((8, ROW_WIDTH), dtype=torch.int32)
    state[6, :3] = torch.tensor([101, 7, 1], dtype=torch.int32)
    state[1, :3] = torch.tensor([202, 10, 1], dtype=torch.int32)
    state[4, :3] = torch.tensor([303, 4, 1], dtype=torch.int32)
    state[3, :3] = torch.tensor([404, 9, 0], dtype=torch.int32)
    control = torch.zeros((16, ROW_WIDTH), dtype=torch.int32)
    control[:, CONTROL_SEED] = 13
    control[:4, :5] = torch.tensor(
        [[6, 7, 1, 1, 13], [1, 11, 0, 1, 29], [4, 5, 1, 1, 17], [3, 9, 1, 1, 19]],
        dtype=torch.int32,
    )
    sample_base = 1000
    if case == "padding":
        control[:] = control[0].clone()
        control[1:, CONTROL_ACTIVE] = 0
    elif case == "reordered":
        state[6, :3] = torch.tensor([1000, 7, 1], dtype=torch.int32)
        state[1, :3] = torch.tensor([1001, 11, 1], dtype=torch.int32)
        control[:, CONTROL_ACTIVE] = 0
        control[:2, :5] = torch.tensor([[1, 11, 1, 1, 29], [6, 7, 1, 1, 13]], dtype=torch.int32)
        sample_base = 2000
    elif case == "replacement":
        control[:, CONTROL_ACTIVE] = 0
        control[0, :5] = torch.tensor([6, 8, 0, 1, 42], dtype=torch.int32)
        sample_base = 3000
    elif case == "stale":
        state[6, :3] = torch.tensor([3000, 8, 1], dtype=torch.int32)
        control[:, CONTROL_ACTIVE] = 0
        control[0, :5] = torch.tensor([6, 7, 1, 1, 13], dtype=torch.int32)
        sample_base = 4000
    samples = torch.zeros((16, ROW_WIDTH), dtype=torch.int32)
    samples[:, 0] = torch.arange(sample_base, sample_base + 16, dtype=torch.int32)
    return [
        TensorSpec("state", list(state.shape), torch.int32, init_value=lambda: state.clone()),
        TensorSpec("control", list(control.shape), torch.int32, init_value=lambda: control.clone()),
        TensorSpec("next_sample", list(samples.shape), torch.int32, init_value=lambda: samples.clone()),
        TensorSpec("tokens", [16, ROW_WIDTH], torch.int32),
        TensorSpec("snapshot", [16, ROW_WIDTH], torch.int32),
    ]


def golden_request_token_state(values):
    state = values["state"]
    control = values["control"]
    values["tokens"].zero_()
    values["snapshot"].copy_(values["next_sample"])
    for row in range(control.shape[0]):
        slot, generation, feedback, active, seed = (int(x) for x in control[row, :5])
        token, valid = seed, 1
        if feedback:
            saved_token, saved_generation, saved_valid = (int(x) for x in state[slot, :3])
            valid = int(saved_generation == generation and saved_valid == 1)
            token = saved_token if valid else 0
        values["tokens"][row, :3] = values["tokens"].new_tensor([token, valid, generation])
        values["snapshot"][row, 1:3] = values["snapshot"].new_tensor([valid, generation])
    for row in range(control.shape[0]):
        slot, generation, _, active, _ = (int(x) for x in control[row, :5])
        if active and int(values["tokens"][row, TOKEN_VALID]):
            state[slot].zero_()
            state[slot, :3] = state.new_tensor([int(values["snapshot"][row, 0]), generation, 1])


if __name__ == "__main__":
    import argparse

    from pypto import passes

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    args = parser.parse_args()
    failed = False
    for case in ["mixed", "padding", "reordered", "replacement", "stale"]:
        with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
            result = run(
                fn=request_token_state_fwd,
                specs=build_tensor_specs(case),
                golden_fn=golden_request_token_state,
                config=dict(
                    backend_type=args.backend_type,
                    execution_mode=args.execution_mode,
                    device_id=args.device,
                ),
                rtol=0,
                atol=0,
            )
        print(f"request-token-state {case}: {'PASS' if result.passed else 'FAIL'}")
        failed = failed or not result.passed
        if result.error:
            print(result.error)
    raise SystemExit(int(failed))
