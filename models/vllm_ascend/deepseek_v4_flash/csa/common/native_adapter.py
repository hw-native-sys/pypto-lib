# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""TND CSA 的 Native 根 Tensor 绑定，不经过旧版均匀 S6 入口。

只检查描述符并建立同存储的 view。Q/L 的内容、每个请求的范围和 C4 输出数
继续由设备端解释；本层不回读设备元数据、不按 T/B 构造请求、不创建 B×6
中间输入。权重和 Top-K/output 工作区由服务持有并在捕获前准备。

五个 cache group 保留各自的 Native 页表、slots 和 state，主/Indexer 的
紧凑 metadata 也分别绑定。后续 binding 对完整根参数和工作区再作 ABI 校验。
"""

from collections.abc import Mapping

import torch

from .native_storage import indexer_storage, physical_pages, table_storage


def token_capacity_view(value: torch.Tensor, tokens: int, width: int) -> torch.Tensor:
    """Expose Native's existing capacity buffer behind its shorter decode view.

    Native builds RoPE and slot storage for num_input_tokens, then returns views
    sliced to num_decode_tokens. Padding can therefore shorten the descriptor
    while the initialized backing buffer still covers the input capacity.
    No new allocation, copy or tensor-value read is performed here.
    """
    if value.ndim < 2 or value.numel() != value.shape[0] * width or not value.is_contiguous():
        raise ValueError("Native token行须是连续且具有固定行宽的视图")
    needed_bytes = (value.storage_offset() + tokens * width) * value.element_size()
    if needed_bytes > value.untyped_storage().nbytes():
        raise ValueError("Native token视图的已有存储不足以覆盖token容量")
    if value.shape == (tokens, width):
        return value
    return value.as_strided((tokens, width), (width, 1), value.storage_offset())


def bind_native_inputs(
    weights: Mapping[str, torch.Tensor],
    hidden: torch.Tensor,
    positions: torch.Tensor,
    groups: Mapping,
    *,
    layer_name: str,
    compact_metadata: Mapping,
    output: torch.Tensor,
    topk: torch.Tensor,
    topk_scores: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """绑定一个 Native decode 调用；不分配业务 Tensor，也不消费 metadata 值。"""
    if hidden.ndim != 3 or hidden.shape[1:] != (4, 4096):
        raise ValueError("TND CSA入口要求hidden[T_capacity,4,4096]")
    tokens = hidden.shape[0]
    if hidden.dtype != torch.bfloat16 or not hidden.is_contiguous():
        raise ValueError("TND CSA要求连续BF16 hidden")
    if positions.shape != (tokens,) or positions.dtype != torch.int64:
        raise ValueError("TND CSA要求Native INT64 positions覆盖token容量")
    if output.shape != hidden.shape or output.dtype != hidden.dtype or not output.is_contiguous():
        raise ValueError("TND CSA输出须与hidden描述符一致")
    if topk.shape != (tokens, 512) or topk.dtype != torch.int32:
        raise ValueError("TND CSA Top-K要求INT32[T_capacity,512]")
    if topk_scores.shape != (tokens, 512) or topk_scores.dtype != torch.float32:
        raise ValueError("TND CSA scores要求FP32[T_capacity,512]")
    if any(value.device != hidden.device for value in (positions, output, topk, topk_scores)):
        raise ValueError("TND CSA根输入输出须处于同一设备")

    req = {name: item[0].decode for name, item in groups.items()}
    views = {name: item[1] for name, item in groups.items()}
    requests = req["indexer"].seq_lens.numel()
    if requests <= 0 or tokens <= 0:
        raise ValueError("空请求由服务执行安全回退，不构造SPMD工作区")
    for name in ("swa", "compressed", "state", "indexer", "indexer_state"):
        item = req[name]
        if item.seq_lens.shape != (requests,) or item.query_start_loc.shape != (requests + 1,):
            raise ValueError(f"{name}: Native请求容量不一致")
        if item.seq_lens.dtype != torch.int32 or item.query_start_loc.dtype != torch.int32:
            raise ValueError(f"{name}: Native Q/L须为INT32")

    native = dict(weights)
    native.update(
        x_hc=hidden,
        x_out=output,
        position_ids=positions,
        kv_cache=views["swa"][0],
        cmp_kv=views["compressed"][0],
        idx_native_kv_cache=indexer_storage(*views["indexer"]),
        kv_seq_lens=req["indexer"].seq_lens,
        cmp_query_start_loc=req["compressed"].query_start_loc,
        cmp_seq_lens=req["compressed"].seq_lens,
        idx_query_start_loc=req["indexer"].query_start_loc,
        compress_state=physical_pages(views["state"][0]),
        inner_compress_state=physical_pages(views["indexer_state"][0]),
        idx_topk=topk,
        idx_topk_scores=topk_scores,
    )
    for name, prefix in (
        ("swa", "ori"),
        ("compressed", "cmp"),
        ("indexer", "idx"),
        ("state", "state"),
        ("indexer_state", "inner_state"),
    ):
        native[prefix + "_block_table"] = table_storage(req[name].block_table)
    for name, prefix in (("swa", "ori"), ("state", "state"), ("indexer_state", "inner_state")):
        slots = req[name].slot_mapping
        if slots.dtype != torch.int32:
            raise ValueError(f"{name}: Native slots须为INT32[T_capacity,2]")
        native[prefix + "_slot_mapping"] = token_capacity_view(slots, tokens, 2)
    for name, prefix, slot_prefix in (("compressed", "cmp", "cmp"), ("indexer", "inner", "idx")):
        cos, sin, slots = compact_metadata[name]
        native[prefix + "_freqs_cos"] = cos.view(-1, 64)
        native[prefix + "_freqs_sin"] = sin.view(-1, 64)
        native[slot_prefix + "_slot_mapping"] = slots
    for kind in ("cos", "sin"):
        value = getattr(req["compressed"], kind)[layer_name]
        native["freqs_" + kind] = token_capacity_view(value, tokens, 64)
    return native
