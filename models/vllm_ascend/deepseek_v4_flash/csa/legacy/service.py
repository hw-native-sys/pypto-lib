# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Per-layer PTO resources and binding of the current Native service buffers."""

import torch

from vllm_ascend.ops.pypto.variant import csa_runtime

from .host_metadata import CSAHostMetadata
from .native_adapter import HBGNativeCSACall, NativeCSACall
from .decode_csa import _decode_csa_tp1_layer
from ..common.weights import prepare_weights
from ..common.service_base import AttentionServiceBase
from ..common.service_config import MAX_BATCH_SIZE, QUERY_TOKENS

class CSAServiceRuntime(AttentionServiceBase):
    def __init__(self, attention, operators, max_num_seqs, layer=None):
        self.wrapper = attention.dsa_attn
        self.layer_name = self.wrapper.dsa_attn.layer_name
        self.operators = operators
        self.is_hbg = csa_runtime() == "host_build_graph"
        # layer 提供 mHC 的门控权重与 attention 的 input_layernorm：
        # 这两段现在也在 PTO kernel 里，见 decode_csa._decode_csa_tp1_layer。
        self.weights = prepare_weights(attention, None, layer, root_function=_decode_csa_tp1_layer)
        self.prefixes = {
            "swa": self.wrapper.swa_cache_layer.prefix,
            "compressed": self.layer_name,
            "state": attention.compressor.state_cache.prefix,
            "indexer": attention.indexer.k_cache.prefix,
            "indexer_state": attention.indexer.compressor.state_cache.prefix,
        }
        self.batch_capacity = min(max_num_seqs, MAX_BATCH_SIZE)
        tokens = self.batch_capacity * QUERY_TOKENS
        device = attention.wq_a.weight.device
        # Allocate once after weight loading; these buffers are private to one
        # layer. Output belongs to the Native forward call / captured graph.
        self.scores = torch.empty((tokens, 512), dtype=torch.float32, device=device)
        self.topk = torch.empty((tokens, 512), dtype=torch.int32, device=device)
        self._hadamard = None

    def __call__(self, context, hidden, positions, output, kv_cache):
        from vllm_ascend.attention.utils import (
            maybe_save_kv_layer_to_connector,
            notify_kv_cache_written,
            wait_for_kv_layer_from_connector,
        )
        from vllm_ascend.memcache_comm_fence import record_attention_compute_start

        metadata = {name: context.attn_metadata[prefix] for name, prefix in self.prefixes.items()}
        # Native creates this Hadamard table when it builds attention metadata.
        # Transform once in an ordinary warmup call, never during capture/replay.
        hadamard = metadata["indexer"].hadamard
        if hadamard is None:
            raise ValueError("Native indexer metadata did not provide the Hadamard matrix")
        if self._hadamard is not hadamard:
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("PTO CSA requires Native metadata warmup before graph capture")
            self.weights["hadamard_idx"] = hadamard.detach().T.to(torch.bfloat16).contiguous()
            self._hadamard = hadamard

        wait_for_kv_layer_from_connector(self.layer_name)
        # Release Native creates compact rows at the consumer on this stream.
        # Pass those exact device tensors to CSA, without expanding them or
        # introducing the main-branch DeviceMetadataExecutor API.
        compact = {
            name: self._compact_metadata(context, metadata[name].decode)
            for name in ("compressed", "indexer")
        }
        compressed, swa, state, indexer_state, indexer_key, indexer_scale = kv_cache
        groups = {
            "swa": (metadata["swa"], (swa,)),
            "compressed": (metadata["compressed"], (compressed,)),
            "state": (metadata["state"], (state,)),
            "indexer": (metadata["indexer"], (indexer_key, indexer_scale)),
            "indexer_state": (metadata["indexer_state"], (indexer_state,)),
        }
        tokens = hidden.shape[0]
        call_type = HBGNativeCSACall if self.is_hbg else NativeCSACall
        host_kwargs = {}
        if self.is_hbg:
            host = CSAHostMetadata.from_native(metadata["indexer"].decode)
            host_kwargs["host_metadata"] = host
            # An enclosing NPUGraph freezes the Host scalar. Record precisely
            # which Native CPU value its replay guard must check; per-request
            # lengths remain live device inputs inside the captured tasks.
            context.additional_kwargs.setdefault("pto_csa_hbg_graph_metadata", {})[
                self.prefixes["indexer"]
            ] = host
        call = call_type(
            self.operators, self.weights, hidden, positions, groups, layer_name=self.layer_name,
            compact_metadata=compact,
            buffers={"idx_topk_scores": self.scores[:tokens], "idx_topk": self.topk[:tokens], "x_out": output},
            **host_kwargs,
        )
        # A single fused call publishes all KV writes. Notify the connector on
        # the same stream after that call; no global synchronization is needed.
        record_attention_compute_start()
        call()
        notify_kv_cache_written(self.layer_name)
        maybe_save_kv_layer_to_connector(self.layer_name, list(kv_cache))
