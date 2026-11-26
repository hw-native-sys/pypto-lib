# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Execute caller-selected batches on the resident A3 Prefill and Decode graphs."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from . import config as C
from .utils import paged_slots, prepare_decode_inputs, window_metadata


@dataclass(frozen=True, eq=False)
class RequestHandle:
    """Opaque session-local identity. A reused name or slot receives a new handle."""

    request_id: str
    dp: int
    slot: int


@dataclass
class _History:
    handle: RequestHandle
    prompt_length: int
    committed: int = 0
    pages: list[int] = field(default_factory=list)


class PackedSession:
    """A synchronous execution context for one scheduler thread."""

    def __init__(self, graphs, slots, context, capacities):
        self.graphs = graphs
        self.slots = slots
        self.context = context
        self.capacities = capacities
        self.history = {}
        self._idle_values = {}
        self._active_dps = [set(), set()]
        self.failed = False
        self._pages_per_slot = (context + C.BLOCK_SIZE - 1) // C.BLOCK_SIZE

    def _ready(self):
        if self.failed:
            raise RuntimeError("Close the packed session after a failed device operation")

    def admit(self, request_id, dp, prompt_length):
        self._ready()
        if not isinstance(request_id, str) or not request_id or request_id in self.history:
            raise ValueError("Expected a unique live request ID")
        if type(dp) is not int or not 0 <= dp < C.DP_SIZE:
            raise ValueError("Invalid DP owner")
        if type(prompt_length) is not int or not 1 <= prompt_length <= self.context:
            raise ValueError("Invalid prompt length")
        occupied = {r.handle.slot for r in self.history.values() if r.handle.dp == dp}
        slot = next((i for i in range(self.slots) if i not in occupied), None)
        if slot is None:
            raise ValueError("DP request slots exhausted")
        handle = RequestHandle(request_id, dp, slot)
        self.history[request_id] = _History(handle, prompt_length)
        return handle

    def _lookup(self, handle):
        record = self.history.get(handle.request_id)
        if record is None or record.handle is not handle:
            raise ValueError("Expired or foreign request handle")
        return record

    def release(self, handle):
        """Finish or cancel one request after its last synchronous dispatch."""
        self._ready()
        record = self._lookup(handle)
        g = self.graphs
        def reset_rank(rank):
            for name, initial in g.host_cache[rank].items():
                target = g.ranks[0][rank][name]
                blocks = [handle.slot] if name == "state_cache" else record.pages
                page_image = g.host_io.get("reset_" + name)
                for source in range(initial.shape[0]):
                    for block in blocks:
                        region = initial[source, block]
                        image = region if page_image is None else page_image
                        offset = region.data_ptr() - initial.data_ptr()
                        g.runtime.copy_to(target.data_ptr, image.data_ptr(), region.nbytes,
                                          dst_offset=offset, worker_id=rank)
        try:
            owner = handle.dp * C.TP_SIZE
            with ThreadPoolExecutor(max_workers=C.TP_SIZE) as workers:
                list(workers.map(reset_rank, range(owner, owner + C.TP_SIZE)))
        except BaseException:
            self.failed = True
            raise
        g.engram.history.pop(handle, None)
        del self.history[handle.request_id]

    def advance_batch(self, phase, queries):
        """Run (handle, token IDs) pairs in caller order. Return sampled rows keyed by handle."""
        self._ready()
        if phase not in ("prefill", "decode"):
            raise ValueError("Expected prefill or decode")
        index = int(phase == "decode")
        capacity = self.capacities[index]
        groups = [[] for _ in range(C.DP_SIZE)]
        seen = set()
        queries = list(queries)
        if not queries:
            raise ValueError("Empty dispatch")
        for handle, tokens in queries:
            record = self._lookup(handle)
            tokens = tuple(tokens)
            if handle in seen or not tokens or any(type(t) is not int or not 0 <= t < C.FLASH.vocab_size for t in tokens):
                raise ValueError("Invalid or repeated query")
            end = record.committed + len(tokens)
            if index and (len(tokens) != 1 or record.committed < record.prompt_length):
                raise ValueError("Decode requires completed Prefill and one new token")
            if end > self.context or (not index and end > record.prompt_length):
                raise ValueError("Query exceeds the admitted history")
            groups[handle.dp].append((record, tokens))
            seen.add(handle)
        if any(sum(len(ids) for _, ids in group) > capacity for group in groups):
            raise ValueError("Batch exceeds the artifact token capacity on a DP owner")
        # All admission errors above leave the session reusable. Failure during staging or execution
        # may have changed Engram history or device caches. Do not retry against that state.
        try:
            result = self._dispatch(groups, index, capacity)
        except BaseException:
            self.failed = True
            raise
        for group in groups:
            for record, ids in group:
                record.committed += len(ids)
        return {handle: result[handle] for handle, _ in queries if handle in result}

    def _inputs(self, group, index, capacity):
        g = self.graphs
        # Prefill and Decode share the same packed positions, page maps and cache allocations.
        template = g.host_io["idle_inputs"][0]
        table = torch.full_like(template["c1a_index_block_table"], -1)
        half = torch.full_like(template["c2a_index_block_table"], -1)
        state = torch.full_like(template["state_block_table"], -1)
        lengths, ids, positions, owners, lookups, sampled = [], [], [], [], [], []
        if not index:
            # Equal-length rows retain caller order. Sample selection remains keyed by handle.
            group = sorted(group, key=lambda item: len(item[1]), reverse=True)
        for row, (record, tokens) in enumerate(group):
            handle = record.handle
            start = record.committed
            end = start + len(tokens)
            needed = (end + C.BLOCK_SIZE - 1) // C.BLOCK_SIZE
            # Keep each request's physical pages in chronological order.
            base = handle.slot * self._pages_per_slot
            record.pages.extend(range(base + len(record.pages), base + needed))
            physical = torch.tensor(record.pages, dtype=torch.int32)
            table[row, :needed] = physical
            n_half = (end // 2 + C.BLOCK_SIZE - 1) // C.BLOCK_SIZE
            half[row, :n_half] = physical[:n_half]
            state[row, 0] = handle.slot
            lookups.append(g.engram.lookup(tokens, len(tokens), start, request_id=handle))
            ids.extend(tokens)
            positions.extend(range(start, end))
            owners.extend([row] * len(tokens))
            lengths.append(len(ids))
            if end >= record.prompt_length:
                sampled.append((handle, len(ids) - 1))
        count = len(ids)
        values = dict(g.host_io["idle_inputs"][0 if not index else 1])
        pos = torch.tensor(positions, dtype=torch.int32)
        req = torch.tensor(owners, dtype=torch.int32)
        slots, window, _ = window_metadata(pos, req, table)
        compressed = paged_slots(pos, req, half, C.BLOCK_SIZE, 2, True)
        def pad(value, fill=-1):
            out = torch.full((capacity, *value.shape[1:]), fill, dtype=value.dtype)
            out[:value.shape[0]] = value
            return out
        starts = torch.tensor([0, *lengths, *([count] * (self.slots - len(group)))], dtype=torch.int32)
        rows = torch.full((16,), -1, dtype=torch.int32)
        for row, (_, source) in enumerate(sampled):
            rows[row] = source
        emb = g.weights.lookup_rows(torch.tensor(ids, dtype=torch.int64))
        initial = torch.zeros((capacity, C.FLASH.hc_mult), dtype=torch.float32)
        initial[:, 0] = 1
        engram = torch.zeros((2, capacity, template["engram_lookup"].shape[-1]), dtype=torch.bfloat16)
        if lookups:
            engram[:, :count] = torch.cat(lookups, dim=1)
        values.update(embedding=pad(emb, 0), initial_pre=initial, engram_lookup=engram,
                      logit_rows=rows, c2a_index_block_table=half, c1a_index_block_table=table,
                      state_block_table=state, c1a_compressed_slots=pad(slots),
                      c2a_compressed_slots=pad(compressed))
        for name, value in (("window_slots", pad(slots)), ("window_indices", pad(window)),
                            ("position_ids", pad(pos)), ("token_to_req_indices", pad(req)),
                            ("query_start_loc", starts)):
            values[name] = torch.stack((value, value))
        if index:
            values.update(freqs_cos=g.rope_tables[0], freqs_sin=g.rope_tables[1])
            values = prepare_decode_inputs(values)
        else:
            order = pad(torch.arange(count, dtype=torch.int32))
            values.update(counts=torch.tensor([count, count], dtype=torch.int32),
                          identity_rows=order, replay_rows=order)
        return values, sampled, count

    def _dispatch(self, groups, index, capacity):
        g = self.graphs
        selections = []
        with ThreadPoolExecutor(max_workers=C.TP_SIZE) as workers:
            for dp, group in enumerate(groups):
                owner = dp * C.TP_SIZE
                # Both graph images start idle. Clear a DP exactly once after
                # it leaves a batch; an idle DP has no live rows or output slots.
                if not group and dp not in self._active_dps[index]:
                    continue
                if group:
                    values, sampled, count = self._inputs(group, index, capacity)
                else:
                    if index not in self._idle_values:
                        self._idle_values[index] = self._inputs([], index, capacity)
                    values, sampled, count = self._idle_values[index]
                selections.extend((handle, owner, row) for row, (handle, _) in enumerate(sampled))
                if index:
                    for image in (1, 2):
                        for name, field in g.decode_inputs.fields[image].items():
                            if image == 1 and name == "logit_row_indices":
                                field.fill_(-1)
                            else:
                                field.copy_(values[name])
                    for name in ("moe_num_tokens", "attention_num_tokens"):
                        g.arguments[index][name][owner:owner + C.TP_SIZE] = count
                    list(workers.map(lambda rank: g.decode_inputs.upload(g.runtime, rank, owner),
                                     range(owner, owner + C.TP_SIZE)))
                else:
                    self._upload_prefill_group(owner, values, workers)
        self._active_dps[index] = {dp for dp, group in enumerate(groups) if group}
        args = [g.arguments[index][name] for name in g.names[index]]
        g.runtime.run(g.programs[index], *args, config=g.config)
        result = {}
        for handle, rank, row in selections:
            logits = g.host_io["logits"][0]
            dst = g.ranks[index][rank]["logits"]
            g.runtime.copy_from(logits.data_ptr(), dst.data_ptr, logits.nbytes, src_offset=row * logits.nbytes,
                                worker_id=rank)
            result[handle] = logits.clone()
        return result

    def _upload_prefill_group(self, owner, values, workers):
        g = self.graphs
        inputs = {}
        for name in g.host_io["idle_inputs"][0]:
            field = g.host_io["prefill_" + name][1]
            value = values[name]
            if tuple(field.shape) != tuple(value.shape) or field.dtype != value.dtype:
                raise ValueError(f"Packed tensor {name} differs from its bound graph")
            field.copy_(value)
            inputs[name] = field
        idle_rows = g.host_io["prefill_logit_rows"][0].fill_(-1)

        def upload_rank(rank):
            for name, field in inputs.items():
                value = idle_rows if name == "logit_rows" and rank != owner else field
                dst = g.ranks[0][rank][name]
                g.runtime.copy_to(dst.data_ptr, value.data_ptr(), value.nbytes, worker_id=rank)

        # All copies finish before the next DP group stages its inputs in the same buffers.
        list(workers.map(upload_rank, range(owner, owner + C.TP_SIZE)))


@contextmanager
def open_packed_session(args, devices, weights, engram, memory, report, *, request_slots):
    """Bind a matched graph pair. All request operations must use the same worker thread."""
    from .driver import load_manifest
    from .generation import open_resident_graphs

    manifests = [load_manifest(path, mode=phase, capacity=capacity, request_slots=request_slots)
                 for path, phase, capacity in ((args.runtime_dir, "prefill", args.capacity),
                                               (args.decode_runtime_dir, "decode", args.decode_capacity))]
    if any(m["history_capacity"] != args.context_capacity for m in manifests):
        raise ValueError("History capacity differs from the compiled graph pair")
    with open_resident_graphs(
        args, devices, weights, engram, memory, report,
        prefill_slab_rows=manifests[0]["moe_slab_rows"], request_slots=request_slots,
    ) as graphs:
        session = PackedSession(graphs, request_slots, args.context_capacity, (args.capacity, args.decode_capacity))
        try:
            yield session
        finally:
            session.failed = True
            for record in session.history.values():
                engram.history.pop(record.handle, None)
