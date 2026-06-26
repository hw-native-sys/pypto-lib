# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared TensorSpec builders and weight-initialisation helpers for Qwen3 test harnesses.

Usage (in a model file's ``build_tensor_specs``):

    from models.shared.specs import (
        build_qwen3_decode_specs, build_qwen3_prefill_specs,
        init_rand, init_qkv_weight, init_down_weight, init_ones,
    )

    specs = build_qwen3_decode_specs(
        batch=BATCH, hidden=HIDDEN, num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        intermediate=INTERMEDIATE, use_max_seq=args.max_seq,
    )

Files with additional tensors (paged cache, per-head QK-norm, LM head, etc.)
call the appropriate builder and then ``extend()`` with extra specs.
"""

from __future__ import annotations

import torch
from golden import TensorSpec


# ── Init helpers (return callables suitable as TensorSpec init_value) ──


def init_rand(*shape, scale: float = 1.0, offset: float = 0.0):
    """``torch.rand(*shape) * scale + offset``."""
    return lambda: torch.rand(*shape) * scale + offset


def init_rand_half(*shape):
    """``(torch.rand(*shape) - 0.5)`` — activations, cache, rope."""
    return lambda: torch.rand(*shape) - 0.5


def init_qkv_weight(*shape, hidden: float):
    """Q/K/V weights: ``torch.rand(...) / sqrt(hidden)`` (unsigned, no offset).

    Matches ``qwen3_32b_decode.py`` init_wq/wk/wv pattern.  The ``wo``,
    ``w_gate``, ``w_up`` projections use signed inits in the existing code
    (``(torch.rand - 0.5) / sqrt(hidden)``) — use ``init_proj_weight`` for
    those.
    """
    denom = hidden ** 0.5
    return lambda: torch.rand(*shape) / denom


def init_proj_weight(*shape, hidden: float):
    """Output/gate/up projection weights: ``(torch.rand - 0.5) / sqrt(hidden)``
    (signed, zero-mean).  Matches ``init_wo`` / ``init_w_gate`` / ``init_w_up``
    in the existing code.
    """
    denom = hidden ** 0.5
    return lambda: (torch.rand(*shape) - 0.5) / denom


def init_down_weight(*shape, intermediate: float):
    """Down-projection weight: ``(torch.rand - 0.5) / sqrt(intermediate)``
    (signed, zero-mean).  Matches ``init_w_down`` in the existing code.
    """
    denom = intermediate ** 0.5
    return lambda: (torch.rand(*shape) - 0.5) / denom


def init_linear_weight(*shape, sqrt_denom: float):
    """Generic linear weight: ``torch.rand(...) / sqrt_denom`` (unsigned)."""
    return lambda: torch.rand(*shape) / sqrt_denom


def init_ones(*shape):
    """Constant ``torch.ones(*shape)`` (for norm/rms weights)."""
    return lambda: torch.ones(*shape)


# ── Parameterized spec builders ──


def build_qwen3_decode_specs(
    *,
    batch: int = 16,
    max_seq: int = 4096,
    hidden: int = 8192,
    num_heads: int = 64,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    intermediate: int = 25600,
    use_max_seq: bool = False,
) -> list:
    """Build TensorSpec list for a single-layer decode golden test.

    Returns specs with 2D flat tensors (the common ``32b/qwen3_32b_decode.py``
    layout).  Callers that need additional tensors (paged cache, per-head
    QK-norm, LM head, 4D layout) can ``specs.extend(...)`` or build manually
    using the init helpers above.
    """
    kv_hidden = num_kv_heads * head_dim
    cache_rows = batch * num_kv_heads * max_seq

    def init_seq_lens():
        if use_max_seq:
            return torch.full((batch,), max_seq, dtype=torch.int32)
        return torch.randint(1, max_seq + 1, (batch,), dtype=torch.int32)

    return [
        TensorSpec("hidden_states", [batch, hidden], torch.bfloat16,
                   init_value=init_rand_half(batch, hidden)),
        TensorSpec("input_rms_weight", [1, hidden], torch.float32,
                   init_value=init_rand_half(1, hidden)),
        TensorSpec("wq", [hidden, hidden], torch.bfloat16,
                   init_value=init_qkv_weight(hidden, hidden, hidden=hidden)),
        TensorSpec("wk", [hidden, kv_hidden], torch.bfloat16,
                   init_value=init_qkv_weight(hidden, kv_hidden, hidden=hidden)),
        TensorSpec("wv", [hidden, kv_hidden], torch.bfloat16,
                   init_value=init_qkv_weight(hidden, kv_hidden, hidden=hidden)),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=init_seq_lens),
        TensorSpec("rope_cos", [max_seq, head_dim], torch.float32,
                   init_value=init_rand_half(max_seq, head_dim)),
        TensorSpec("rope_sin", [max_seq, head_dim], torch.float32,
                   init_value=init_rand_half(max_seq, head_dim)),
        TensorSpec("k_cache", [cache_rows, head_dim], torch.bfloat16,
                   init_value=init_rand_half(cache_rows, head_dim)),
        TensorSpec("v_cache", [cache_rows, head_dim], torch.bfloat16,
                   init_value=init_rand_half(cache_rows, head_dim)),
        TensorSpec("wo", [hidden, hidden], torch.bfloat16,
                   init_value=init_proj_weight(hidden, hidden, hidden=hidden)),
        TensorSpec("post_rms_weight", [1, hidden], torch.float32,
                   init_value=init_ones(1, hidden)),
        TensorSpec("w_gate", [hidden, intermediate], torch.bfloat16,
                   init_value=init_proj_weight(hidden, intermediate, hidden=hidden)),
        TensorSpec("w_up", [hidden, intermediate], torch.bfloat16,
                   init_value=init_proj_weight(hidden, intermediate, hidden=hidden)),
        TensorSpec("w_down", [intermediate, hidden], torch.bfloat16,
                   init_value=init_down_weight(intermediate, hidden, intermediate=intermediate)),
        TensorSpec("out", [batch, hidden], torch.bfloat16, is_output=True),
    ]


def build_qwen3_prefill_specs(
    *,
    batch: int = 16,
    max_seq: int = 256,
    hidden: int = 8192,
    num_heads: int = 64,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    intermediate: int = 25600,
    use_max_seq: bool = False,
    tok_tile: int = 64,
) -> list:
    """Build TensorSpec list for a single-layer prefill golden test (flat cache, no paging).

    Returns specs with 3D ``[batch, max_seq, hidden]`` hidden states (prefill
    layout).  Callers that need paged cache or chunked prefill should build
    manually or extend.
    """
    kv_hidden = num_kv_heads * head_dim
    cache_rows = batch * num_kv_heads * max_seq

    def init_seq_lens():
        if use_max_seq:
            return torch.full((batch,), max_seq, dtype=torch.int32)
        n_blocks = max_seq // tok_tile
        blocks = torch.randint(1, n_blocks + 1, (batch,), dtype=torch.int32)
        return blocks * tok_tile

    return [
        TensorSpec("hidden_states", [batch, max_seq, hidden], torch.bfloat16,
                   init_value=init_rand_half(batch, max_seq, hidden)),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=init_seq_lens),
        TensorSpec("input_rms_weight", [1, hidden], torch.float32,
                   init_value=init_rand_half(1, hidden)),
        TensorSpec("wq", [hidden, hidden], torch.bfloat16,
                   init_value=init_qkv_weight(hidden, hidden, hidden=hidden)),
        TensorSpec("wk", [hidden, kv_hidden], torch.bfloat16,
                   init_value=init_qkv_weight(hidden, kv_hidden, hidden=hidden)),
        TensorSpec("wv", [hidden, kv_hidden], torch.bfloat16,
                   init_value=init_qkv_weight(hidden, kv_hidden, hidden=hidden)),
        TensorSpec("rope_cos", [max_seq, head_dim], torch.float32,
                   init_value=init_rand_half(max_seq, head_dim)),
        TensorSpec("rope_sin", [max_seq, head_dim], torch.float32,
                   init_value=init_rand_half(max_seq, head_dim)),
        TensorSpec("k_cache", [cache_rows, head_dim], torch.bfloat16,
                   init_value=init_rand_half(cache_rows, head_dim)),
        TensorSpec("v_cache", [cache_rows, head_dim], torch.bfloat16,
                   init_value=init_rand_half(cache_rows, head_dim)),
        TensorSpec("wo", [hidden, hidden], torch.bfloat16,
                   init_value=init_proj_weight(hidden, hidden, hidden=hidden)),
        TensorSpec("post_rms_weight", [1, hidden], torch.float32,
                   init_value=init_ones(1, hidden)),
        TensorSpec("w_gate", [hidden, intermediate], torch.bfloat16,
                   init_value=init_proj_weight(hidden, intermediate, hidden=hidden)),
        TensorSpec("w_up", [hidden, intermediate], torch.bfloat16,
                   init_value=init_proj_weight(hidden, intermediate, hidden=hidden)),
        TensorSpec("w_down", [intermediate, hidden], torch.bfloat16,
                   init_value=init_down_weight(intermediate, hidden, intermediate=intermediate)),
        TensorSpec("out", [batch, max_seq, hidden], torch.bfloat16, is_output=True),
    ]
