# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F401,F403,F405,F821
"""DeepSeek-V4 prefill sparse_attn scaffold.

Kernel body is intentionally empty; golden follows the torch reference for this stage.
"""

import pypto.language as pl

from decode_sparse_attn import *  # noqa: F401,F403
from decode_sparse_attn import build_tensor_specs as _build_tensor_specs
from decode_sparse_attn import _int8_quant_per_row


@pl.jit
def prefill_sparse_attn(
    q: pl.Tensor[[T, H, HEAD_DIM],                               pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],       pl.BF16],
    ori_block_table: pl.Tensor[[B, ORI_MAX_BLOCKS],                            pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],       pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS],                            pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, TOPK],                                      pl.INT32],
    attn_sink: pl.Tensor[[H],                                            pl.FP32],
    seqused_kv: pl.Tensor[[B],                                            pl.INT32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM],                                  pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM],                                  pl.BF16],
    even_select_local: pl.Tensor[[ROPE_INTERLEAVE_CHUNK, ROPE_CHUNK],            pl.BF16],
    odd_select_local: pl.Tensor[[ROPE_INTERLEAVE_CHUNK, ROPE_CHUNK],            pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN],                 pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA],                         pl.INT8],
    wo_b_scale: pl.Tensor[[D],                                            pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T, D],                                  pl.BF16]],
):
    # TODO: kernel implementation
    return attn_out


def golden_prefill_sparse_attn(tensors):
    """Non-distributed prefill sparse attention.

    Official prefill uses the full prompt KV cache with causal masking. This
    golden mirrors that single-rank behavior with the existing standalone
    sparse-attn tensor contract: ``ori_kv`` is treated as the full prompt cache
    for each batch, and each query token attends to positions ``[0, s]``.
    """
    import torch

    q = tensors["q"].float()
    ori_kv = tensors["ori_kv"].float()
    ori_block_table = tensors["ori_block_table"]
    attn_sink = tensors["attn_sink"].float()
    cos = tensors["freqs_cos"].float()
    sin = tensors["freqs_sin"].float()
    wo_a = tensors["wo_a"].float()
    wo_b_i8 = tensors["wo_b"]
    wo_b_scale = tensors["wo_b_scale"].float()

    o = torch.zeros(T, H, HEAD_DIM)
    for t in range(T):
        b = t // S
        s = t % S
        gathered = []
        for raw in range(s + 1):
            blk_id = int(ori_block_table[b, raw // BLOCK_SIZE].item())
            intra = raw % BLOCK_SIZE
            gathered.append(ori_kv[blk_id, intra, 0])
        kv_b = torch.stack(gathered, dim=0)
        scores = (q[t] @ kv_b.T) * SOFTMAX_SCALE
        score_max = scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - score_max)
        oi_num = exp_scores @ kv_b
        li = exp_scores.sum(dim=-1, keepdim=True)
        denom = li + torch.exp(attn_sink.unsqueeze(-1) - score_max)
        o[t] = oi_num / denom

    rope_pair = o[..., NOPE_DIM:].unflatten(-1, (-1, 2))
    rope_even = rope_pair[..., 0]
    rope_odd = rope_pair[..., 1]
    cos_half = cos[:, :HALF_ROPE].unsqueeze(1)
    sin_half = sin[:, :HALF_ROPE].unsqueeze(1)
    inv_even = (rope_even * cos_half + rope_odd * sin_half).to(torch.bfloat16).float()
    inv_odd = (rope_odd * cos_half - rope_even * sin_half).to(torch.bfloat16).float()
    o_rope = torch.stack([inv_even, inv_odd], dim=-1).flatten(-2)
    o = torch.cat([o[..., :NOPE_DIM], o_rope], dim=-1).to(torch.bfloat16)

    o_model = o.float().view(B, S, O_GROUPS, O_GROUP_IN)
    o_r = torch.einsum("bsgd,grd->bsgr", o_model, wo_a)
    o_r = o_r.to(torch.bfloat16).float()
    o_r_q = o_r.flatten(2).view(T, O_GROUPS * O_LORA)
    o_r_i8, o_r_scale = _int8_quant_per_row(o_r_q)
    acc = o_r_i8.to(torch.int32) @ wo_b_i8.to(torch.int32).T
    out = acc.float() * o_r_scale * wo_b_scale.unsqueeze(0)

    tensors["attn_out"][:] = out.to(torch.bfloat16)


def build_tensor_specs(*args, **kwargs):
    return _build_tensor_specs(*args, **kwargs)
