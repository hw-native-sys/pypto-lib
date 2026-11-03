# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared TND CSA service lifecycle for the SPMD and MPMD implementations.

Mode-specific services supply weights, policy and prepared bindings. This
module owns Native metadata validation, output/workspace reuse, graph replay
contracts, and cache connector lifecycle without importing either mode.
"""

from abc import ABC, abstractmethod

import torch

from .common.service_base import AttentionServiceBase
from .common.graph_contract import TNDGraphContract, tensor_descriptor
from .common.native_adapter import bind_native_inputs, token_capacity_view
from .common.tnd_config import MAX_QUERY_TOKENS, supported_descriptor


def _descriptor_key(native):
    """只用描述符作工作区键；地址可以rebind，Tensor内容不得进入Host控制流。"""
    return tuple(
        (name, tuple(value.shape), value.stride(), value.dtype, value.device) for name, value in sorted(native.items())
    )


def _rope_tensor(request, kind, layer_name):
    """Native uses RopeDataProxy rather than a dict; both expose subscription."""
    mapping = getattr(request, kind, None)
    if mapping is None:
        return None
    try:
        value = mapping[layer_name]
    except (KeyError, TypeError):
        return None
    return value if isinstance(value, torch.Tensor) else None


def _native_token_views(requests, layer_name, tokens):
    """Validate the initialized Native capacity behind shortened decode views."""
    result = []
    for name in ("swa", "state", "indexer_state"):
        slots = requests[name].slot_mapping
        if slots is None or slots.dtype != torch.int32:
            raise ValueError("Native slots须为INT32")
        result.append((name + ".slots", token_capacity_view(slots, tokens, 2)))
    for kind in ("cos", "sin"):
        value = _rope_tensor(requests["compressed"], kind, layer_name)
        if value is None or value.dtype != torch.float32:
            raise ValueError("Native RoPE须为FP32")
        result.append((kind, token_capacity_view(value, tokens, 64)))
    return result


class TNDServiceRuntime(AttentionServiceBase, ABC):
    @staticmethod
    @abstractmethod
    def _prepare_weights(attention, layer):
        """Prepare weights using the implementation's root signature."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _select_policy(**descriptors):
        """Select the implementation's shape-dependent policy."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _prepare_binding(native, **options):
        """Allocate or bind the implementation's private workspaces."""
        raise NotImplementedError

    def __init__(self, attention, operators, max_num_seqs, layer=None):
        # operators参数保留模型准备钩子的统一调用约定；本实现不注册多task算子。
        if operators is not None:
            raise ValueError("TND CSA直接提交，不接受旧BSH CSAOperators")
        self.wrapper = attention.dsa_attn
        self.layer_name = self.wrapper.dsa_attn.layer_name
        self.weights = self._prepare_weights(attention, layer)
        self.prefixes = {
            "swa": self.wrapper.swa_cache_layer.prefix,
            "compressed": self.layer_name,
            "state": attention.compressor.state_cache.prefix,
            "indexer": attention.indexer.k_cache.prefix,
            "indexer_state": attention.indexer.compressor.state_cache.prefix,
        }
        self.batch_capacity = max_num_seqs
        self._hadamard = None
        self._outputs = {}
        self._bindings = {}
        self.submissions = 0
        self.empty_calls = 0
        self.last_binding = None

    def record_graph_route(self, context, hidden, positions):
        """先记下未确认路由；只有实际单SPMD提交路径才能覆盖为可重放合同。"""
        if getattr(context, "capturing", False):
            context.additional_kwargs.setdefault("pto_csa_tnd_graph_contracts", {})[self.layer_name] = TNDGraphContract(
                self, hidden.shape[0], None
            )

    def graph_signature(self, context, tokens):
        """不执行metadata生产者或读取Tensor内容，避免旧图绕过真实服务判定。"""
        from vllm_ascend.ops.dsa import _build_kv_cache

        metadata = context.attn_metadata
        if metadata is None or getattr(context, "is_draft_model", False):
            return None
        descriptors = []
        reference_counts = None
        for name, prefix in self.prefixes.items():
            item = metadata.get(prefix)
            if item is None or item.num_prefills or item.decode is None:
                return None
            req = item.decode
            requests = req.seq_lens.numel()
            if not supported_descriptor(
                tokens=tokens,
                requests=requests,
                actual_tokens=item.num_actual_tokens,
                max_query_tokens=req.max_seqlen_q,
                max_seq_len=req.max_seq_lens,
                batch_capacity=self.batch_capacity,
            ):
                return None
            if (
                item.num_decodes != requests
                or req.query_start_loc.shape != (requests + 1,)
                or req.seq_lens.dtype != torch.int32
                or req.query_start_loc.dtype != torch.int32
                or req.seq_lens.device != req.query_start_loc.device
            ):
                return None
            if req.num_reqs_actual is not None and not 1 <= req.num_reqs_actual <= requests:
                return None
            counts = (requests, item.num_actual_tokens, req.max_seqlen_q, req.max_seq_lens)
            if reference_counts is not None and counts != reference_counts:
                return None
            reference_counts = counts
            if req.ori_win_right not in (None, 0) or req.dspark_swa_indices is not None:
                return None
            descriptors.append(
                (
                    name,
                    tuple(
                        (field, tensor_descriptor(getattr(req, field, None)))
                        for field in (
                            "seq_lens",
                            "query_start_loc",
                            "block_table",
                            "slot_mapping",
                            "input_positions",
                            "start_pos",
                        )
                    ),
                )
            )
            if name in ("compressed", "indexer"):
                # Native compressor_metadata also captures these Host scalars
                # and table roots. Changed capacity/request count needs redispatch.
                descriptors.append(
                    (
                        name + ".compact_producer",
                        tuple(
                            getattr(req, field, None)
                            for field in (
                                "num_reqs_actual",
                                "num_compressed_tokens",
                                "block_size",
                            )
                        ),
                        tensor_descriptor(getattr(req, "full_compress_cos", None)),
                        tensor_descriptor(getattr(req, "full_compress_sin", None)),
                    )
                )
        try:
            views = _native_token_views(
                {name: metadata[prefix].decode for name, prefix in self.prefixes.items()},
                self.layer_name,
                tokens,
            )
        except ValueError:
            return None
        descriptors.extend((name, tensor_descriptor(value)) for name, value in views)
        if metadata[self.prefixes["swa"]].decode.ori_win_left not in (None, 127):
            return None
        indexer = metadata[self.prefixes["indexer"]]
        descriptors.append(("hadamard", tensor_descriptor(indexer.hadamard)))
        kv_cache = _build_kv_cache(self.wrapper, context)
        descriptors.append(("kv_cache", tuple(tensor_descriptor(value) for value in kv_cache)))
        policy = self._select_policy(
            tokens=tokens,
            requests=indexer.decode.seq_lens.numel(),
            max_query_tokens=MAX_QUERY_TOKENS,
            max_seq_len=indexer.decode.max_seq_lens,
        )
        return (tuple(descriptors), policy)

    def handle_empty_call(self, context, hidden, positions, output):
        """空decode不启动全核协议，不访问或更新Native cache/state。"""
        if hidden.numel() == 0:
            output.copy_(hidden)
            self.empty_calls += 1
            return True
        metadata = context.attn_metadata
        if metadata is None or getattr(context, "is_draft_model", False):
            return False
        for prefix in self.prefixes.values():
            item = metadata.get(prefix)
            if item is None or item.num_prefills or item.num_actual_tokens != 0:
                return False
        # dummy输出不作为有效token使用；保留输入值便于检查无额外状态副作用。
        output.copy_(hidden)
        self.empty_calls += 1
        return True

    def eligible(self, context, hidden, positions):
        metadata = context.attn_metadata
        if metadata is None or getattr(context, "is_draft_model", False):
            return False
        if hidden.ndim != 3 or hidden.shape[1:] != (4, 4096):
            return False
        if hidden.dtype != torch.bfloat16 or not hidden.is_contiguous():
            return False
        tokens = hidden.shape[0]
        if (
            positions.shape != (tokens,)
            or positions.dtype != torch.int64
            or positions.device != hidden.device
            or not positions.is_contiguous()
        ):
            return False
        reference_counts = None
        for prefix in self.prefixes.values():
            item = metadata.get(prefix)
            if item is None or item.num_prefills or item.decode is None:
                return False
            req = item.decode
            requests = req.seq_lens.numel()
            if not supported_descriptor(
                tokens=tokens,
                requests=requests,
                actual_tokens=item.num_actual_tokens,
                max_query_tokens=req.max_seqlen_q,
                max_seq_len=req.max_seq_lens,
                batch_capacity=self.batch_capacity,
            ):
                return False
            if item.num_decodes != requests or req.query_start_loc.shape != (requests + 1,):
                return False
            if req.seq_lens.dtype != torch.int32 or req.query_start_loc.dtype != torch.int32:
                return False
            if req.seq_lens.device != hidden.device or req.query_start_loc.device != hidden.device:
                return False
            counts = (requests, item.num_actual_tokens, req.max_seqlen_q, req.max_seq_lens)
            if reference_counts is not None and counts != reference_counts:
                return False
            reference_counts = counts
            if req.num_reqs_actual is not None and not 1 <= req.num_reqs_actual <= requests:
                return False
            if req.ori_win_right not in (None, 0) or req.dspark_swa_indices is not None:
                return False
        try:
            _native_token_views(
                {name: metadata[prefix].decode for name, prefix in self.prefixes.items()},
                self.layer_name,
                tokens,
            )
        except ValueError:
            return False
        swa = metadata[self.prefixes["swa"]].decode
        return swa.ori_win_left in (None, 127)

    def __call__(self, context, hidden, positions, output, kv_cache):
        from vllm_ascend.attention.utils import (
            maybe_save_kv_layer_to_connector,
            notify_kv_cache_written,
            wait_for_kv_layer_from_connector,
        )
        from vllm_ascend.memcache_comm_fence import record_attention_compute_start

        metadata = {name: context.attn_metadata[prefix] for name, prefix in self.prefixes.items()}
        capturing = torch.npu.is_current_stream_capturing()
        hadamard = metadata["indexer"].hadamard
        if hadamard is None:
            raise ValueError("Native Indexer未提供Hadamard")
        if self._hadamard is not hadamard:
            if capturing:
                raise RuntimeError("TND CSA须在捕获前完成Native Hadamard warmup")
            self.weights["hadamard_idx"] = hadamard.detach().T.to(torch.bfloat16).contiguous()
            self._hadamard = hadamard

        wait_for_kv_layer_from_connector(self.layer_name)
        compact = {name: self._compact_metadata(context, metadata[name].decode) for name in ("compressed", "indexer")}
        compressed, swa, state, indexer_state, indexer_key, indexer_scale = kv_cache
        groups = {
            "swa": (metadata["swa"], (swa,)),
            "compressed": (metadata["compressed"], (compressed,)),
            "state": (metadata["state"], (state,)),
            "indexer": (metadata["indexer"], (indexer_key, indexer_scale)),
            "indexer_state": (metadata["indexer_state"], (indexer_state,)),
        }
        tokens = hidden.shape[0]
        output_key = (tokens, hidden.device)
        if output_key not in self._outputs:
            if capturing:
                raise RuntimeError("TND CSA须在捕获前准备当前token容量的Top-K工作区")
            self._outputs[output_key] = (
                torch.empty((tokens, 512), dtype=torch.int32, device=hidden.device),
                torch.empty((tokens, 512), dtype=torch.float32, device=hidden.device),
            )
        topk, scores = self._outputs[output_key]
        native = bind_native_inputs(
            self.weights,
            hidden,
            positions,
            groups,
            layer_name=self.layer_name,
            compact_metadata=compact,
            output=output,
            topk=topk,
            topk_scores=scores,
        )
        requests = native["kv_seq_lens"].numel()
        maximum = metadata["indexer"].decode.max_seq_lens
        # 固定的是本服务允许的query上界，真实Q/L仍可在1..6之间逐请求变化。
        # 不因当前max_q较小而把图容量误判为T > B*max_q。
        policy = self._select_policy(
            tokens=tokens,
            requests=requests,
            max_query_tokens=MAX_QUERY_TOKENS,
            max_seq_len=maximum,
        )
        key = (_descriptor_key(native), policy)
        if key not in self._bindings:
            if capturing:
                raise RuntimeError("TND CSA须在捕获前warmup当前描述符与策略")
            self._bindings[key] = self._prepare_binding(
                native,
                max_query_tokens=MAX_QUERY_TOKENS,
                max_seq_len=maximum,
                policy=policy,
            )
        else:
            self._bindings[key].rebind(native)
        prepared = self._bindings[key]
        if capturing:
            signature = self.graph_signature(context, tokens)
            if signature is None:
                raise RuntimeError("TND CSA捕获输入未满足可重放合同")
            context.additional_kwargs.setdefault("pto_csa_tnd_graph_contracts", {})[self.layer_name] = TNDGraphContract(
                self, tokens, signature
            )
        record_attention_compute_start()
        prepared()
        self.submissions += 1
        self.last_binding = prepared
        notify_kv_cache_written(self.layer_name)
        maybe_save_kv_layer_to_connector(self.layer_name, list(kv_cache))
