# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared Native eligibility and per-step compact metadata caching."""

import torch

from .service_config import QUERY_TOKENS

_COMPACT_METADATA_CACHE = "pto_csa_compact_compressor_metadata"


class AttentionServiceBase:
    """CSA性能版与HCA共用的入口判定和metadata缓存；不注册算术实现。"""

    def eligible(self, context, hidden, positions):
        metadata = context.attn_metadata
        if metadata is None or getattr(context, "is_draft_model", False):
            return False
        # 整层入口：进出都是层间的 mHC 残差流 [T, HC_MULT, D]。
        if hidden.ndim != 3 or tuple(hidden.shape[1:]) != (4, 4096) or hidden.dtype != torch.bfloat16:
            return False
        tokens = hidden.shape[0]
        if tokens % QUERY_TOKENS or not 1 <= tokens // QUERY_TOKENS <= self.batch_capacity:
            return False
        if not hidden.is_contiguous() or positions.dtype != torch.int64 or positions.shape != (tokens,):
            return False
        batch = tokens // QUERY_TOKENS
        for prefix in self.prefixes.values():
            item = metadata.get(prefix)
            # 补位档位下这两个计数不再相等：num_actual_tokens 是**实际** token 数，
            # num_decodes 是**补齐后**的请求数（实测 18 与 4，整档 24）。补位请求
            # 由 kernel 内的 seq_lens 判据屏蔽，这里只要求真实部分是完整六行请求。
            if item is None or item.num_prefills or item.num_actual_tokens > tokens:
                return False
            if item.num_actual_tokens % QUERY_TOKENS or item.num_decodes != batch:
                return False
            req = item.decode
            if req is None or req.seq_lens.numel() != batch or req.query_start_loc.numel() != batch + 1:
                return False
            actual = item.num_actual_tokens // QUERY_TOKENS
            if req.max_seqlen_q != QUERY_TOKENS or req.num_reqs_actual not in (None, actual):
                return False
            if req.ori_win_right not in (None, 0) or req.dspark_swa_indices is not None:
                return False
        main = metadata[self.prefixes["compressed"]]
        if main.num_actual_tokens < tokens and main.decode.cos[self.layer_name].shape[0] < tokens:
            # Native 的 decode RoPE 视图按实际 token 切片。图捕获发生在无补位的满档
            # dummy 上，捕获到的就是整档视图，重放不受影响；只有 eager 下真出现补位
            # 才会切短。那种情况下整档算不出来，让本层回退 Native，
            # 而不是另建一份冗余缓冲去凑。
            return False
        swa = metadata[self.prefixes["swa"]].decode
        return swa.ori_win_left in (None, 127)

    def _compact_metadata(self, context, req):
        """取这一步该 KV cache group 的 (cos, sin, slot_mapping)，同 step 内跨层复用。

        缓存挂在 forward context 的 additional_kwargs 上：它每次前向都由
        `vllm_ascend/platform.py` 的 set_additional_forward_context 新建，
        所以不会跨 step 残留。aclgraph 下被捕获进图的同样只有第一层那一次调用，
        重放时后面各层读的是同一块固定地址的输出，语义与逐层重算完全一致。
        """
        cache = context.additional_kwargs.setdefault(_COMPACT_METADATA_CACHE, {})
        # id() 在对象回收后可能被复用，所以连同对象本身一起存下来比对。
        cached = cache.get(id(req))
        if cached is not None and cached[0] is req:
            return cached[1]
        value = self.wrapper.dsa_attn.impl._compute_compressor_metadata(req)
        cache[id(req)] = (req, value)
        return value
