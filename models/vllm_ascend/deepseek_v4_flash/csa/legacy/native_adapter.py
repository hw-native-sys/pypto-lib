# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CSA Native-storage binding, with weight/layout helpers reused by HCA.

The caller retains Native metadata waits and cache lifecycle hooks. Allocation,
weight preparation and operator registration happen before graph capture.
"""

from dataclasses import dataclass
from typing import Any

import torch

from ..common.config import DECODE_BATCH
from .decode_csa import (
    csa_compile_constants,
    decode_csa_tp1_layer_hbg,
    decode_csa_tp1_layer_test,
)
from .host_metadata import CSAHostMetadata
from ..common.native_storage import indexer_storage, physical_pages, table_storage

@dataclass(frozen=True)
class CSAOperators:
    attention: Any
    attention_single_m16: Any = None

    @classmethod
    def register(cls, kernel=decode_csa_tp1_layer_test, *, namespace="dsv4_csa", config=None) -> "CSAOperators":
        import pypto.torch

        from ..common.reduction import validate_reduction_mode

        validate_reduction_mode()

        def register(name, constants=None):
            return pypto.torch.register(kernel, f"{namespace}::{name}", constexpr=constants, config=config)

        if kernel is decode_csa_tp1_layer_test:
            return cls(
                register("attention", {"SINGLE_M16_QA": False}),
                register("attention_single_m16", {"SINGLE_M16_QA": True}),
            )
        return cls(register("attention"))

    def for_tokens(self, tokens):
        if self.attention_single_m16 is not None and csa_compile_constants(tokens)["SINGLE_M16_QA"]:
            return self.attention_single_m16
        return self.attention


class NativeCSACall:
    """Fixed-address eager/graph call for uniform six-token target requests.

    整档中可以含补位请求：它们由 kernel 内的 seq_lens 判据屏蔽，此处只校验真实
    部分仍是完整六行请求。模型前向的派发仍由各自的闸门决定。
    compact_metadata contains the release Native producer's device tensors;
    this descriptor binds them without materializing an expanded buffer.
    """

    def __init__(self, ops, weights, hidden, positions, groups, *, layer_name: str, compact_metadata, buffers=None,
                 kernel=decode_csa_tp1_layer_test, host_args=None):
        # Each entry contains its own metadata and Native cache views. No shared
        # synthetic page table can stand in for another cache group.
        self.ops = ops
        self.groups = groups
        self.positions = positions
        self.req = {name: value[0].decode for name, value in groups.items()}
        self.views = {name: value[1] for name, value in groups.items()}
        batch = self.req["swa"].seq_lens.numel()
        tokens = hidden.shape[0]
        if not 1 <= batch <= DECODE_BATCH or tokens != batch * 6:
            raise ValueError(f"CSA requires 1 <= batch <= {DECODE_BATCH} and six unpadded rows per request")
        # kernel 现在从 mHC 的残差流进、也从它出，入参是层间的 [T, HC_MULT, D]。
        if tuple(hidden.shape[1:]) != (4, 4096) or hidden.dtype != torch.bfloat16 or not hidden.is_contiguous():
            raise ValueError("CSA expects a contiguous BF16 hc residual stream [T, 4, 4096]")
        if positions.dtype != torch.int64 or tuple(positions.shape) != (tokens,):
            raise ValueError("CSA expects the Native INT64 target position vector")
        for name, (metadata, _) in groups.items():
            # 补位档位下 num_actual_tokens 是**实际** token 数，小于 hidden 的整档
            # 行数（实测 18/24）；补位请求本身由 kernel 内的 seq_lens 判据屏蔽，
            # 这里只要求真实部分是完整的六行请求。
            if metadata.num_prefills or metadata.num_actual_tokens > tokens:
                raise ValueError(f"{name}: requires target decode metadata without prefill rows")
            if metadata.num_actual_tokens % 6:
                raise ValueError(f"{name}: CSA requires whole six-token target requests")
            req = self.req[name]
            if req.query_start_loc.numel() != batch + 1 or req.seq_lens.numel() != batch:
                raise ValueError(f"{name}: inconsistent Native request capacity")
            if req.ori_win_right not in (None, 0):
                raise ValueError("Noncausal drafter windows are outside the target CSA contract")

        def empty(name, shape, dtype):
            if buffers is None:
                return torch.empty(shape, dtype=dtype, device=hidden.device)
            value = buffers[name]
            if tuple(value.shape) != shape or value.dtype != dtype or value.device != hidden.device:
                raise ValueError(f"Invalid prepared CSA buffer {name}")
            return value

        for name, width in (("state", 2048), ("indexer_state", 512)):
            view = self.views[name][0]
            if view.dtype != torch.float32 or tuple(view.shape[1:]) != (2, 1, width):
                raise ValueError(f"{name}: CSA expects Native FP32 two-token state pages with row width {width}")
        self.state_storage = {name: physical_pages(self.views[name][0]) for name in ("state", "indexer_state")}
        self.tables = {name: table_storage(req.block_table) for name, req in self.req.items()}
        self.args = dict(weights)
        self.args.update(host_args or {})
        self.args.update(
            x_hc=hidden,
            kv_cache=self.views["swa"][0],
            cmp_kv=self.views["compressed"][0],
            **self._indexer_cache_arguments(),
            cmp_block_table=table_storage(self.req["compressed"].block_table),
            idx_block_table=table_storage(self.req["indexer"].block_table),
            kv_seq_lens=self.req["indexer"].seq_lens,
            position_ids=positions,
            ori_slot_mapping=self.req["swa"].slot_mapping,
            state_slot_mapping=self.req["state"].slot_mapping,
            inner_state_slot_mapping=self.req["indexer_state"].slot_mapping,
            ori_block_table=table_storage(self.req["swa"].block_table),
            compress_state=self.state_storage["state"],
            inner_compress_state=self.state_storage["indexer_state"],
            state_block_table=self.tables["state"],
            inner_state_block_table=self.tables["indexer_state"],
            idx_topk_scores=empty("idx_topk_scores", (tokens, 512), torch.float32),
            idx_topk=empty("idx_topk", (tokens, 512), torch.int32),
            x_out=empty("x_out", (tokens, 4, 4096), torch.bfloat16),
        )
        for name, slot_name, rope_name in (
            ("compressed", "cmp_slot_mapping", "cmp_freqs"),
            ("indexer", "idx_slot_mapping", "inner_freqs"),
        ):
            cos, sin, slots = compact_metadata[name]
            self.args[slot_name] = slots
            self.args[f"{rope_name}_cos"] = cos.view(-1, 64)
            self.args[f"{rope_name}_sin"] = sin.view(-1, 64)
        self.args["cmp_query_start_loc"] = self.req["compressed"].query_start_loc
        self.args["cmp_seq_lens"] = self.req["compressed"].seq_lens
        self.args["idx_query_start_loc"] = self.req["indexer"].query_start_loc
        for name in ("kv_cache", "cmp_kv"):
            if not self.args[name].is_contiguous() or self.args[name].dtype != torch.bfloat16:
                raise ValueError(f"{name}: Native BF16 32-token pages must have no additional page padding")
        main = self.req["compressed"]
        # Native 的 decode RoPE 是常驻缓冲的切片视图，按**实际** token 数切
        # （`vllm_ascend/ops/rope_dsv4.py` 的 use_cache 分支）。图捕获发生在无补位的
        # 满档 dummy 上，捕获到的是整档视图；补位步只回写前若干行，尾部保留上一步
        # 的值，与 positions 同机制，不是越界。这里显式要求视图覆盖整档，
        # 免得在 view 上抛出难以定位的形状错误。
        cos, sin = main.cos[layer_name], main.sin[layer_name]
        if cos.shape[0] < tokens or sin.shape[0] < tokens:
            raise ValueError("CSA requires Native decode RoPE rows covering the whole padded bucket")
        self.native_cos = cos[:tokens].view(tokens, 64)
        self.native_sin = sin[:tokens].view(tokens, 64)
        # All CSA consumers use Native interleaved FP32 frequency columns.
        # Keep the Native buffers and their producer waits; no device conversion.
        self.args["freqs_cos"] = self.native_cos
        self.args["freqs_sin"] = self.native_sin
        # The dispatcher schema contains runtime parameters only; a JIT source
        # signature may additionally declare constexpr specialization controls.
        schema = getattr(self.ops.attention, "_schema", None)
        self.param_names = tuple(arg.name for arg in schema.arguments) if schema is not None else kernel.param_names
        self.core_args = tuple(self.args[name] for name in self.param_names)

    def _indexer_cache_arguments(self):
        native_cache = indexer_storage(*self.views["indexer"])
        return {"idx_native_kv_cache": native_cache}

    def __call__(self):
        selector = getattr(self.ops, "for_tokens", None)
        entry = self.ops.attention if selector is None else selector(self.args["x_hc"].shape[0])
        entry(*self.core_args)
        return self.args["x_out"]


class HBGCSAOperators(CSAOperators):
    @classmethod
    def register(cls):
        import pypto.torch

        from ..common.reduction import validate_reduction_mode

        validate_reduction_mode()
        return cls(pypto.torch.register(decode_csa_tp1_layer_hbg, "dsv4_csa::attention_hbg"))


class HBGNativeCSACall(NativeCSACall):
    """Separate scalar ABI, sharing all Native tensor/cache/weight bindings.

    ``host_metadata`` is required. The caller obtains it from this step's Native
    CPU metadata and must select the matching topology before NPUGraph replay.
    """

    def __init__(self, *args, host_metadata: CSAHostMetadata, **kwargs):
        if not isinstance(host_metadata, CSAHostMetadata):
            raise TypeError("CSA HBG requires explicit CSAHostMetadata")
        super().__init__(*args, host_args={"host_max_seq_len": host_metadata.max_seq_len}, **kwargs)
        if "host_max_seq_len" not in self.param_names:
            raise TypeError("HBGNativeCSACall requires HBGCSAOperators and its Host scalar ABI")
        self.update_host_metadata(host_metadata)

    def update_host_metadata(self, host_metadata: CSAHostMetadata):
        if not isinstance(host_metadata, CSAHostMetadata):
            raise TypeError("CSA HBG requires explicit CSAHostMetadata")
        self.host_metadata = host_metadata
        self.args["host_max_seq_len"] = host_metadata.max_seq_len
        self.core_args = tuple(self.args[name] for name in self.param_names)
