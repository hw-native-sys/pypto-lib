# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Serving contract for packed A3 requests."""

import ctypes
from types import SimpleNamespace

import pytest
import torch

from models.deepseek_v4_1_flash_a3 import config as c
from models.deepseek_v4_1_flash_a3.host import prepare_prefill_inputs
from models.deepseek_v4_1_flash_a3.packed import PackedSession
from models.deepseek_v4_1_flash_a3.weights import ENGRAM_K


def test_packed_prefill_returns_completed_requests_and_expires_released_handles():
    capacity = c.BLOCK_SIZE
    context = 2 * c.BLOCK_SIZE
    slots = 2
    template = prepare_prefill_inputs(
        torch.empty(0, c.FLASH.hidden_size, dtype=torch.bfloat16),
        capacity, max_seq_len=context, request_slots=slots,
    )
    fields = (
        "embedding", "position_ids", "logit_rows", "engram_lookup",
        "c1a_index_block_table", "c2a_index_block_table", "state_block_table",
    )
    inputs = {name: template[name] for name in fields}
    staging = {"prefill_" + name: torch.stack((value, value)) for name, value in inputs.items()}
    device_inputs = [{name: value.clone() for name, value in inputs.items()} for _ in range(c.TP_SIZE)]
    device_logits = torch.zeros(slots, 2, dtype=torch.float32)
    ranks = [{name: SimpleNamespace(data_ptr=field.data_ptr()) for name, field in image.items()}
             for image in device_inputs]
    ranks[0]["logits"] = SimpleNamespace(data_ptr=device_logits.data_ptr())

    def copy_to(dst, src, size, *, worker_id, dst_offset=0):
        ctypes.memmove(dst + dst_offset, src, size)

    def copy_from(dst, src, size, *, worker_id, src_offset=0):
        ctypes.memmove(dst, src + src_offset, size)

    def run(program, *, config):
        owner = device_inputs[0]
        for index, row in enumerate(owner["logit_rows"][:slots].tolist()):
            if row >= 0:
                device_logits[index, 0] = owner["position_ids"][0, row]
                device_logits[index, 1] = owner["embedding"][row, 0]

    def embed(ids):
        return ids[:, None].expand(-1, c.FLASH.hidden_size).to(torch.bfloat16)

    graphs = SimpleNamespace(
        host_io={"idle_inputs": (inputs, {}), "logits": torch.empty(1, 2), **staging},
        host_cache=[{} for _ in range(c.TP_SIZE)], ranks=[ranks],
        runtime=SimpleNamespace(copy_to=copy_to, copy_from=copy_from, run=run),
        weights=SimpleNamespace(lookup_rows=embed),
        engram=SimpleNamespace(lookup=lambda ids, n, offset, **kw: torch.zeros(2, n, ENGRAM_K), history={}),
        programs=("prefill",), arguments=({},), names=((),), config=None,
    )
    session = PackedSession(graphs, slots, context, (capacity, 16))
    a = session.admit("a", 0, capacity + 1)
    b = session.admit("b", 0, 2)
    assert session.advance_batch("prefill", [(a, [11] * capacity)]) == {}
    with pytest.raises(ValueError, match="completed Prefill"):
        session.advance_batch("decode", [(a, [99])])
    completed = session.advance_batch("prefill", [(a, [33]), (b, [22, 23])])
    assert list(completed) == [a, b]
    assert completed[a].tolist() == [capacity, 33]
    assert completed[b].tolist() == [1, 23]
    page_table = device_inputs[0]["c1a_index_block_table"]
    assert page_table.tolist() == [[2, -1], [0, 1]]
    session.release(a)
    replacement = session.admit("a", 0, 1)
    with pytest.raises(ValueError, match="Expired"):
        session.advance_batch("decode", [(a, [1])])
    assert session.advance_batch("prefill", [(replacement, [44])])[replacement].tolist() == [0, 44]
    assert page_table.tolist() == [[0, -1], [-1, -1]]
