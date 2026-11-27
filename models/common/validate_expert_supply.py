# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Standalone simulator validation of expert cache planning and packed-field supply."""

import argparse

import torch
import pypto.language as pl

from golden import TensorSpec, run
from models.common.expert_cache import initial_expert_cache
from models.common.expert_supply import mapped_copy_workers, mapped_expert_copy, supply_packed_wave


FIELD_BYTES = (1 << 20) + 64
SLOT_BYTES = FIELD_BYTES + 128
COPY_WORKERS = mapped_copy_workers(FIELD_BYTES)
FIELDS = ("w1", "s1", "w2", "s2", "w3", "s3")


@pl.jit
def supply_test(
    counts: pl.Tensor,
    state: pl.InOut[pl.Tensor],
    descriptors: pl.Tensor,
    slots: pl.InOut[pl.Tensor],
    misses: pl.InOut[pl.Tensor],
    w1: pl.InOut[pl.Tensor],
    s1: pl.InOut[pl.Tensor],
    w2: pl.InOut[pl.Tensor],
    s2: pl.InOut[pl.Tensor],
    w3: pl.InOut[pl.Tensor],
    s3: pl.InOut[pl.Tensor],
    layer: pl.Scalar[pl.INDEX],
):
    inputs_ready = pl.system.task_dummy(deps=[])
    slots, w1, s1, w2, s2, w3, s3, ready = supply_packed_wave(
        counts, state, descriptors, slots, misses, w1, s1, w2, s2, w3, s3,
        layer, 0, 4, inputs_ready, COPY_WORKERS, COPY_WORKERS,
    )
    return state, slots, misses, w1, s1, w2, s2, w3, s3


@pl.jit
def copy_view_test(
    descriptor: pl.Tensor,
    misses: pl.Tensor,
    bank: pl.InOut[pl.Tensor],
):
    view = bank[16:32, :]
    view, ready = pl.submit(mapped_expert_copy, descriptor, misses, view, 0, -1, 1)
    return bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=("a2a3", "a5"), default="a5")
    args = parser.parse_args()
    config = dict(backend_type=args.platform, execution_mode=True)
    for case in ("cold", "warm", "protect_hit", "cross_layer"):
        specs, golden = supply_fixture(case)
        result = run(fn=supply_test, specs=specs, golden_fn=golden, config=config, rtol=0, atol=0)
        print(f"{case}: {result}")
        if not result.passed:
            raise SystemExit(1)
    for dtype in (torch.uint8, torch.float32):
        specs, golden = view_fixture(dtype)
        result = run(fn=copy_view_test, specs=specs, golden_fn=golden, config=config, rtol=0, atol=0)
        print(f"typed subview {dtype}: {result}")
        if not result.passed:
            raise SystemExit(1)


def supply_fixture(case):
    from golden import ScalarSpec

    state = initial_expert_cache(2, 2, 4)
    if case != "cold":
        state[1:3, :2] = torch.tensor([[0, 3], [0, 0]])
        state[3:].flatten()[[3, 0]] = torch.tensor([0, 1], dtype=torch.int32)
    counts = torch.tensor([1, 0, 0, 1] if case == "warm" else [0, 1, 0, 1], dtype=torch.int32)
    layer = int(case == "cross_layer")
    # Literal transitions exercise the planner without reproducing its LRU algorithm.
    transitions = {
        "cold": ([-1, 0, -1, 1], [-1, 0, -1, 1], [[0, 1, 0, 2, 0, 1], [0, 3, 1, 0, 0, 1]], 2, 1),
        "warm": ([1, -1, -1, 0], [-1, -1, -1, -1], [[0, 3, 2, 0, 1, 0], [0, 0, 0, 1, 1, 0]], 1, 2),
        "protect_hit": ([-1, 1, -1, 0], [-1, 1, -1, -1], [[0, 3, 0, 2, 1, 0], [0, 1, 1, 0, 0, 1]], 2, 1),
        "cross_layer": ([-1, 0, -1, 1], [-1, 0, -1, 1], [[1, 1, 0, 2, 0, 1], [1, 3, 1, 0, 0, 1]], 2, 1),
    }
    slots, misses, rows, tail, head = transitions[case]
    expected = initial_expert_cache(2, 2, 4)
    expected[0, 2:4] = torch.tensor([tail, head])
    expected[1:3] = torch.tensor(rows)
    for expert, slot in enumerate(slots):
        expected[3:].flatten()[layer * 4 + expert] = slot
    pattern = torch.arange(FIELD_BYTES + 64).reshape(1, -1) + torch.arange(4).reshape(-1, 1) * 17
    sources = [[(pattern + i * 31 + source_layer * 73).byte() for i in range(6)] for source_layer in range(2)]
    descriptors = torch.tensor([
        [[source.data_ptr(), FIELD_BYTES + 64, FIELD_BYTES, SLOT_BYTES] for source in fields]
        for fields in sources
    ], dtype=torch.int64)
    specs = [
        TensorSpec("counts", (4,), torch.int32, init_value=counts),
        TensorSpec("state", state.shape, torch.int32, init_value=state),
        TensorSpec("descriptors", descriptors.shape, torch.int64, init_value=descriptors),
        TensorSpec("slots", (4,), torch.int32, init_value=-9),
        TensorSpec("misses", (4,), torch.int32, init_value=-9),
        *[TensorSpec(name, (2, SLOT_BYTES), torch.uint8, init_value=173) for name in FIELDS],
        ScalarSpec("layer", torch.int64, layer),
    ]

    def golden(values):
        values["state"].copy_(expected)
        values["slots"].copy_(torch.tensor(slots))
        values["misses"].copy_(torch.tensor(misses))
        for name, source in zip(FIELDS, sources[layer]):
            for expert, slot in enumerate(misses):
                if slot >= 0:
                    values[name][slot, :FIELD_BYTES].copy_(source[expert, :FIELD_BYTES])

    return specs, golden


def view_fixture(dtype):
    source = torch.arange(16 * 32).to(dtype).reshape(16, 32)
    descriptor = torch.tensor([source.data_ptr(), source.nbytes, source.nbytes, source.nbytes])
    specs = [
        TensorSpec("descriptor", (4,), torch.int64, init_value=descriptor),
        TensorSpec("misses", (1,), torch.int32, init_value=1),
        TensorSpec("bank", (32, 32), dtype, init_value=17),
    ]

    def golden(values):
        values["bank"][16:].copy_(source)

    return specs, golden


if __name__ == "__main__":
    main()
