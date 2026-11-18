# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepared prompt geometry and typed, shared metadata upload arenas."""

from dataclasses import dataclass
from math import prod

import torch

from .prefill_metadata import prepare_prefill_metadata


RANKS = 8
ALIGNMENT_BYTES = 64
PLAN_VERSION = 1


@dataclass(frozen=True)
class MetadataField:
    name: str
    shape: tuple
    dtype: torch.dtype
    element_offset: int

    @property
    def elements(self):
        return prod(self.shape)

    @property
    def byte_offset(self):
        return self.element_offset * torch.empty((), dtype=self.dtype).element_size()


class TypedMetadataArena:
    """Own one mutable shared CPU allocation per dtype, with aligned field views."""

    def __init__(self, specs):
        self.fields = {}
        self.elements = {}
        self.copying = False
        for spec in specs:
            name = spec["name"]
            shape = tuple(spec["per_rank_shape"])
            dtype = getattr(torch, spec["dtype"])
            if name in self.fields or not shape or any(type(n) is not int or n <= 0 for n in shape):
                raise ValueError("Metadata fields require unique names and positive per-rank shapes")
            if dtype not in (torch.int16, torch.int32, torch.int64, torch.bfloat16, torch.float32):
                raise ValueError("Unsupported metadata arena dtype")
            alignment = ALIGNMENT_BYTES // torch.empty((), dtype=dtype).element_size()
            start = (self.elements.get(dtype, 0) + alignment - 1) // alignment * alignment
            field = MetadataField(name, shape, dtype, start)
            self.fields[name] = field
            self.elements[dtype] = start + field.elements
        for dtype, elements in self.elements.items():
            alignment = ALIGNMENT_BYTES // torch.empty((), dtype=dtype).element_size()
            self.elements[dtype] = (elements + alignment - 1) // alignment * alignment
        self.roots = {dtype: torch.zeros((RANKS, n), dtype=dtype).share_memory_()
                      for dtype, n in self.elements.items()}
        self.views = {}
        for name, field in self.fields.items():
            root = self.roots[field.dtype]
            self.views[name] = root[:, field.element_offset:field.element_offset + field.elements]
            self.views[name] = self.views[name].view(RANKS, *field.shape)

    @classmethod
    def from_tensors(cls, tensors):
        specs = [{"name": name, "per_rank_shape": list(value.shape[1:]),
                  "dtype": str(value.dtype).removeprefix("torch.")}
                 for name, value in tensors.items()]
        result = cls(specs)
        result.copy_from(tensors)
        return result

    def copy_from(self, tensors):
        if self.copying:
            raise RuntimeError("Metadata upload must retire before host refill")
        if tensors.keys() != self.views.keys():
            raise ValueError("Metadata fields differ from the prepared arena")
        for name, value in tensors.items():
            target = self.views[name]
            if value.shape != target.shape or value.dtype != target.dtype or value.device.type != "cpu":
                raise ValueError("Metadata shape or dtype differs: " + name)
            target.copy_(value)

    def begin_copy(self):
        if self.copying:
            raise RuntimeError("Metadata copy is already active")
        self.copying = True

    def end_copy(self):
        if not self.copying:
            raise RuntimeError("No metadata copy is active")
        self.copying = False

    @property
    def bytes_per_rank(self):
        return sum(root[0].numel() * root.element_size() for root in self.roots.values())

    def allocation_specs(self):
        return [{"name": "metadata_arena_" + str(dtype).removeprefix("torch."),
                 "per_rank_shape": [n], "dtype": str(dtype).removeprefix("torch.")}
                for dtype, n in self.elements.items()]


def _tensor_stamp(value):
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise ValueError("Prepared metadata sources must be CPU tensors")
    return (value.data_ptr(), tuple(value.shape), tuple(value.stride()), str(value.dtype), value._version)


def _signature(owner, revision, token_count, start_pos, terminal, tables, caches, rope_cos, rope_sin,
               max_logit_rows, block_counts, extras):
    if not isinstance(owner, str) or not owner or not isinstance(revision, str) or not revision:
        raise ValueError("An explicit cache owner and metadata revision are required")
    cache_specs = tuple((name, tuple(value.shape), str(value.dtype)) for name, value in sorted(caches.items()))
    table_specs = tuple((name, _tensor_stamp(value)) for name, value in sorted(tables.items()))
    extra_specs = tuple((name, _tensor_stamp(value)) for name, value in sorted(extras.items()))
    return (PLAN_VERSION, owner, revision, token_count, start_pos, terminal, max_logit_rows,
            cache_specs, table_specs, _tensor_stamp(rope_cos), _tensor_stamp(rope_sin),
            _tensor_stamp(block_counts), extra_specs)


class PreparedInputPlan:
    """Retain prompt-independent geometry; fill actual token IDs during each request."""

    def __init__(self, *, owner, revision, token_count, start_pos, terminal, tables, caches,
                 rope_cos, rope_sin, max_logit_rows, block_counts, extras):
        self.context = dict(owner=owner, revision=revision, token_count=token_count, start_pos=start_pos,
                            terminal=terminal, tables=tables, caches=caches, rope_cos=rope_cos,
                            rope_sin=rope_sin, max_logit_rows=max_logit_rows,
                            block_counts=block_counts, extras=extras)
        self.signature = _signature(**self.context)
        placeholder = torch.zeros(token_count, dtype=torch.int64)
        values = prepare_prefill_metadata(placeholder, start_pos=start_pos, terminal=terminal,
                                          tables=tables, caches=caches, rope_cos=rope_cos, rope_sin=rope_sin,
                                          max_logit_rows=max_logit_rows, block_counts=block_counts)
        if values.keys() & extras.keys():
            raise ValueError("Extra metadata must not replace native geometry fields")
        values.update(extras)
        self.arena = TypedMetadataArena.from_tensors(values)
        self.generation = 0

    def validate(self, **changes):
        context = {**self.context, **changes}
        if _signature(**context) != self.signature:
            raise ValueError("Prepared input owner/page/position/shape/RoPE/grammar/revision changed")

    def fill(self, tokens, *, projection_epochs, causal_epoch, **changes):
        if self.arena.copying:
            raise RuntimeError("Metadata upload must retire before filling token IDs")
        self.validate(**changes)
        if (not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.int64
                or tokens.device.type != "cpu" or tokens.ndim != 1
                or tokens.numel() != self.context["token_count"]):
            raise ValueError("Actual INT64 token IDs must match the prepared token extent")
        if bool(((tokens < 0) | (tokens >= 129280)).any()):
            raise ValueError("Token ID outside the vocabulary")
        if type(causal_epoch) is not int or not 0 <= causal_epoch < 2**31:
            raise ValueError("Causal epoch must be a nonnegative INT32")
        if len(projection_epochs) != 6 or any(type(v) is not int or not 0 <= v < 2**31 for v in projection_epochs):
            raise ValueError("Projection epochs must be six nonnegative INT32 values")
        views = self.arena.views
        full = views["input_ids_full"]
        full[:, :tokens.numel()].copy_(tokens.unsqueeze(0))
        full[:, tokens.numel():].zero_()
        views["input_ids"].copy_(full[0].reshape(RANKS, -1))
        views["projection_epoch_bases"].copy_(torch.tensor(projection_epochs, dtype=torch.int32).unsqueeze(0))
        views["causal_epoch_bases"].fill_(causal_epoch)
        self.generation += 1
        return views


class DeviceMetadataArena:
    """Own real typed device roots and borrow same-dtype public field views."""

    def __init__(self, worker, layout):
        from pypto.runtime import StackedDeviceTensor

        self.worker = worker
        self.busy = False
        self.roots = {}
        self.fields = {}
        for dtype, n in layout.elements.items():
            shards = [worker.alloc_tensor((n,), dtype, worker_id=rank) for rank in range(RANKS)]
            self.roots[dtype] = StackedDeviceTensor(shards, (RANKS, n), range(RANKS))
        for name, field in layout.fields.items():
            root = self.roots[field.dtype]
            shards = [worker.view_tensor(shard, field.shape, byte_offset=field.byte_offset, worker_id=rank)
                      for rank, shard in enumerate(root.shards)]
            self.fields[name] = StackedDeviceTensor(shards, (RANKS, *field.shape), range(RANKS))

    def require_idle(self, worker):
        if worker is not self.worker:
            raise ValueError("Metadata arenas belong to a different Worker")
        if self.busy:
            raise RuntimeError("Native metadata consumer must finish before arena reuse")

    def begin_consume(self, worker):
        self.require_idle(worker)
        self.busy = True

    def end_consume(self, worker):
        if worker is not self.worker or not self.busy:
            raise RuntimeError("No native metadata consumer is active for this Worker")
        self.busy = False
