# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Deterministic standalone CSA inputs with Native physical cache layouts.

No checkpoint, vLLM model, custom operator, or NPU is required to construct a
CPU fixture. Device fixtures preserve the cache page padding and guard bytes.
"""

from dataclasses import dataclass

import torch

from golden import TensorSpec


@dataclass(frozen=True)
class Case:
    batch: int = 1
    history: int = 8192
    query: int = 6
    mixed: bool = False
    seed: int = 20261102

    def __post_init__(self):
        if self.batch < 1 or self.history < 8192 or not 1 <= self.query <= 6:
            raise ValueError("CSA requires batch >= 1, history >= 8192 and 1 <= query <= 6")
        if self.history + self.query + self.batch > 1048576:
            raise ValueError("CSA sequence length exceeds the supported descriptor range")

    @property
    def queries(self):
        return [max(1, self.query - i % 3) if self.mixed else self.query for i in range(self.batch)]

    @property
    def histories(self):
        offsets = (0, 3, 1, 2)
        return [self.history + offsets[i % 4] if self.mixed else self.history for i in range(self.batch)]

    @property
    def tokens(self):
        valid = sum(self.queries)
        return ((valid + 7) // 8 * 8 + 8) if self.mixed else valid

    @property
    def requests(self):
        padding = self.tokens - sum(self.queries)
        return self.batch + (padding + self.query - 1) // self.query

    @property
    def max_length(self):
        return max(h + q for h, q in zip(self.histories, self.queries))


def _rope(positions):
    inv = 160000.0 ** (-torch.arange(0, 64, 2).float() / 64)
    angle = positions.float()[:, None] * inv[None, :]
    return angle.cos().repeat_interleave(2, -1), angle.sin().repeat_interleave(2, -1)


def build_tensor_specs(case):
    """Build independent seeded weight/input initializers in logical ND layout."""
    generator = torch.Generator().manual_seed(case.seed)

    def normal(shape, scale=1):
        return lambda: torch.randn(shape, generator=generator) * scale

    specs = []

    def add(name, shape, dtype=torch.bfloat16, init=None):
        specs.append(TensorSpec(name, shape, dtype, init_value=init))

    add("x_hc", (case.tokens, 4, 4096), init=normal((case.tokens, 4, 4096), 0.5))
    add("x_out", (case.tokens, 4, 4096))
    add("hc_attn_fn", (24, 16384), torch.float32, normal((24, 16384), 0.05))
    add("hc_attn_scale", (3,), torch.float32, torch.tensor([0.076099, 0.032597, 0.226994]))
    add(
        "hc_attn_base",
        (24,),
        torch.float32,
        torch.tensor(
            [
                5.9166,
                -3.6223,
                -2.9324,
                -3.3124,
                -3.9100,
                -0.9384,
                -3.3256,
                -2.5240,
                2.0706,
                -2.5728,
                0.1424,
                -3.9453,
                -3.8859,
                3.4634,
                -3.3799,
                -2.6077,
                -2.7191,
                -2.4846,
                2.0395,
                -0.5010,
                -3.5992,
                -2.7520,
                -3.3493,
                3.1587,
            ]
        ),
    )
    for name, width in [
        ("attn_norm_w", 4096),
        ("gamma_cq", 1024),
        ("gamma_ckv", 512),
        ("cmp_norm_w", 512),
        ("inner_norm_w", 128),
    ]:
        add(name, (width,), init=1)
    for name, shape in [
        ("wq_a", (1024, 4096)),
        ("wkv", (4096, 512)),
        ("cmp_wkv", (1024, 4096)),
        ("cmp_wgate", (1024, 4096)),
        ("inner_wkv", (256, 4096)),
        ("inner_wgate", (256, 4096)),
        ("wo_a", (8, 4096, 1024)),
    ]:
        add(name, shape, init=normal(shape, 4096**-0.5))
    for name, shape in [("wq_b", (1024, 32768)), ("idx_wq_b", (1024, 8192)), ("wo_b", (8192, 4096))]:
        # Quantized weights and scale are nonzero and exercise all channels.
        add(
            name,
            shape,
            torch.int8,
            lambda shape=shape: torch.randint(-64, 65, shape, generator=generator, dtype=torch.int8),
        )
        add(name + "_scale", (shape[-1],), torch.float32, 1 / (32 * shape[0] ** 0.5))
    add("weights_proj", (4096, 64), init=normal((4096, 64), 0.1))
    add("cmp_ape", (4, 1024), torch.float32, normal((4, 1024), 0.1))
    add("inner_ape", (4, 256), torch.float32, normal((4, 256), 0.1))
    add("attn_sink", (64,), torch.float32, normal((64,), 0.1))
    hadamard = torch.ones(1, 1)
    while hadamard.shape[0] < 128:
        hadamard = torch.cat((torch.cat((hadamard, hadamard), 1), torch.cat((hadamard, -hadamard), 1)), 0)
    add("hadamard_idx", (128, 128), init=hadamard)
    return specs


def make_fixture(case):
    native = {spec.name: spec.create_tensor() for spec in build_tensor_specs(case)}
    native["x_hc"][sum(case.queries) :].zero_()
    positions = torch.zeros(case.tokens, dtype=torch.int64)
    bounds, lengths = [0], []
    for history, query in zip(case.histories, case.queries):
        start = bounds[-1]
        positions[start : start + query] = torch.arange(history, history + query)
        bounds.append(start + query)
        lengths.append(history + query)
    if case.mixed:
        while bounds[-1] < case.tokens:
            bounds.append(min(bounds[-1] + case.query, case.tokens))
            lengths.append(0)
    bounds = torch.tensor(bounds, dtype=torch.int32)
    lengths = torch.tensor(lengths, dtype=torch.int32)
    native.update(
        position_ids=positions,
        kv_seq_lens=lengths,
        cmp_seq_lens=lengths.clone(),
        cmp_query_start_loc=bounds,
        idx_query_start_loc=bounds.clone(),
        idx_topk=torch.full((case.tokens, 512), -1, dtype=torch.int32),
        idx_topk_scores=torch.zeros(case.tokens, 512),
    )
    native["freqs_cos"], native["freqs_sin"] = _rope(positions)
    layouts = {
        "swa": (32, 1, 32768, 512, torch.bfloat16, "kv_cache", "ori"),
        "compressed": (32, 4, 32768, 512, torch.bfloat16, "cmp_kv", "cmp"),
        "state": (2, 1, 32768, 2048, torch.float32, "compress_state", "state"),
        "indexer": (32, 4, 4160, 128, torch.int8, "idx_native_kv_cache", "idx"),
        "indexer_state": (2, 1, 4160, 512, torch.float32, "inner_compress_state", "inner_state"),
    }
    generator = torch.Generator().manual_seed(case.seed + 1)
    groups = {}
    compact = sum((h + q) // 4 - h // 4 for h, q in zip(case.histories, case.queries))
    compact_capacity = max(1, compact)
    for name, (block, ratio, page_bytes, width, dtype, key, prefix) in layouts.items():
        columns = (case.max_length + block * ratio - 1) // (block * ratio) + 1
        owned = columns if ratio == 4 else 9
        pages = case.requests * owned + 1
        allocation = torch.full((pages * page_bytes + 256,), 37, dtype=torch.int8)
        raw = allocation[128:-128]
        itemsize = torch.empty((), dtype=dtype).element_size()
        view = allocation.view(dtype).as_strided(
            (pages, block, width), (page_bytes // itemsize, width, 1), 128 // itemsize
        )
        if dtype == torch.int8:
            view.copy_(torch.randint(-64, 65, view.shape, generator=generator, dtype=dtype))
            scales = allocation.view(torch.float16).as_strided(
                (pages, 32, 1), (page_bytes // 2, 1, 1), (128 + 4096) // 2
            )
            scales.copy_(torch.rand(scales.shape, generator=generator) * 0.01 + 0.004)
            native[key] = raw.view(pages, page_bytes)
        else:
            view.copy_(torch.randn(view.shape, generator=generator) * 0.5)
            native[key] = raw.view(dtype).view(pages, -1) if "state" in name else view
        table = torch.zeros((case.requests, columns), dtype=torch.int32)
        for request in range(case.batch):
            table[request] = (
                1 + (case.batch - request - 1) * owned + (owned - 1 - torch.arange(columns)) % owned
            )
        slots = torch.full((compact_capacity if ratio == 4 else case.tokens, 2), -1, dtype=torch.int32)
        compact_positions = torch.zeros(compact_capacity, dtype=torch.int64)
        cursor = 0
        for request, (history, query) in enumerate(zip(case.histories, case.queries)):
            if ratio == 4:
                for pos in range(history // 4, (history + query) // 4):
                    slots[cursor] = torch.tensor([table[request, pos // block], pos % block])
                    compact_positions[cursor] = pos * 4
                    cursor += 1
            else:
                for offset in range(query):
                    pos = history + offset
                    slots[bounds[request] + offset] = torch.tensor(
                        [table[request, pos // block], pos % block]
                    )
        native[prefix + "_block_table"] = table
        native[prefix + "_slot_mapping"] = slots
        if ratio == 4:
            cos, sin = _rope(compact_positions)
            cos[cursor:].zero_()
            sin[cursor:].zero_()
            if name in ("compressed", "indexer"):
                rope_prefix = "cmp" if name == "compressed" else "inner"
                native[rope_prefix + "_freqs_cos"], native[rope_prefix + "_freqs_sin"] = cos, sin
        allowed = torch.zeros_like(allocation, dtype=torch.bool)
        for page, row in slots.tolist():
            if page < 0:
                continue
            start = 128 + page * page_bytes + row * width * itemsize
            allowed[start : start + width * itemsize] = True
            if name == "indexer":
                start = 128 + page * page_bytes + 4096 + row * 2
                allowed[start : start + 2] = True
        groups[key] = dict(
            allocation=allocation,
            allowed=allowed,
            page_bytes=page_bytes,
            shape=tuple(native[key].shape),
            dtype=dtype,
        )
    return native, groups


def to_device(native, groups, device):
    import torch_npu

    result = {name: value.to(device) for name, value in native.items() if name not in groups}
    guards = {}
    for name, group in groups.items():
        allocation = group["allocation"].to(device)
        result[name] = allocation[128:-128].view(group["dtype"]).view(group["shape"])
        guards[name] = allocation
    for name in ("wq_a", "wq_b", "wo_a", "wo_b"):
        result[name] = torch_npu.npu_format_cast(result[name], 29)
    return result, guards


def outputs(native):
    cache = native["idx_native_kv_cache"].detach().cpu()
    pages = cache.shape[0]
    return {
        "x_out": native["x_out"].detach().cpu().clone(),
        "idx_topk": native["idx_topk"].detach().cpu().clone(),
        "idx_topk_scores": native["idx_topk_scores"].detach().cpu().clone(),
        "swa": native["kv_cache"].detach().cpu().clone(),
        "compressed": native["cmp_kv"].detach().cpu().clone(),
        "state": native["compress_state"].detach().cpu().clone(),
        "indexer_key": cache[:, :4096].reshape(pages, 32, 128).clone(),
        "indexer_scale": cache[:, 4096:4160].contiguous().view(torch.float16).clone(),
        "indexer_state": native["inner_compress_state"].detach().cpu().clone(),
    }
