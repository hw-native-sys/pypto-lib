# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Independent Torch reference for the fused DeepSeek V4 Flash CSA contract.

Weights use logical ND matrices. All accumulation is float32 except the
specified integer products and BF16/FP16 quantization boundaries. Physical
Native cache pages and metadata are inputs, not synthesized from NPU results.
"""

import math

import torch

from ..config import FLASH


def _bf(value):
    return value.to(torch.bfloat16).float()


def _rms(value, gamma=1):
    value = value.float()
    return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6) * gamma


def _quant(value):
    factor = 127 / value.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    return torch.round(value * factor).clamp(-127, 127).to(torch.int8), factor.reciprocal()


def _rope(value, cos, sin, inverse=False):
    value = value.float().clone()
    cos = cos[..., ::2]
    sin = sin[..., ::2]
    while cos.ndim < value.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    even, odd = value[..., -64::2].clone(), value[..., -63::2].clone()
    sign = -1 if inverse else 1
    value[..., -64::2] = even * cos - sign * odd * sin
    value[..., -63::2] = odd * cos + sign * even * sin
    return value


def _cache_rows(cache, table, request, positions):
    return cache[table[request, positions // 32].long(), positions % 32]


def _compress(values, normed, case, indexer=False):
    prefix = "inner" if indexer else "cmp"
    head = 128 if indexer else 512
    state = values["inner_compress_state" if indexer else "compress_state"]
    state_prefix = "inner_state" if indexer else "state"
    state_table = values[state_prefix + "_block_table"]
    state_slots = values[state_prefix + "_slot_mapping"]
    slots = values[("idx" if indexer else "cmp") + "_slot_mapping"]
    cos = values[("inner" if indexer else "cmp") + "_freqs_cos"]
    sin = values[("inner" if indexer else "cmp") + "_freqs_sin"]
    projected = normed @ values[prefix + "_wkv"].float().T
    scores = normed @ values[prefix + "_wgate"].float().T
    scores += values[prefix + "_ape"][values["position_ids"] % 4]
    cursor = 0
    begin = 0
    for request, (history, query) in enumerate(zip(case.histories, case.queries)):
        for local in range(query):
            pos = history + local
            if (pos + 1) % 4:
                continue
            token = begin + local
            # Same mathematical eight-row overlap, independently expressed as softmax.
            data, logits = [], []
            for absolute in range(pos - 7, pos + 1):
                half = 0 if absolute < pos - 3 else head
                if absolute >= history:
                    row = begin + absolute - history
                    data.append(projected[row, half : half + head])
                    logits.append(scores[row, half : half + head])
                else:
                    page = int(state_table[request, absolute // 2])
                    offset = absolute % 2 * (4 * head) + half
                    data.append(state[page, offset : offset + head])
                    logits.append(state[page, offset + 2 * head : offset + 3 * head])
            pooled = (torch.softmax(torch.stack(logits), dim=0) * torch.stack(data)).sum(0)
            normalized = _rms(pooled, values[prefix + "_norm_w"].float())
            rotated = _rope(normalized, cos[cursor], sin[cursor])
            page, row = slots[cursor].tolist()
            if indexer:
                had = _bf(_bf(rotated) @ values["hadamard_idx"].float()) / math.sqrt(128)
                key, scale = _quant(had)
                cache = values["idx_native_kv_cache"]
                cache[page, row * 128 : (row + 1) * 128] = key
                cache[page, 4096 + row * 2 : 4098 + row * 2] = scale.half().reshape(1).view(torch.int8)
            else:
                values["cmp_kv"][page, row] = rotated.to(torch.bfloat16)
            cursor += 1
        begin += query
    # Read all historical rows before committing the current decode chunk.
    for token in range(sum(case.queries)):
        page, row = state_slots[token].tolist()
        offset = row * 4 * head
        state[page, offset : offset + 2 * head] = projected[token]
        state[page, offset + 2 * head : offset + 4 * head] = scores[token]


def golden_csa(values, case):
    """Mutate all nine observable outputs; return full Indexer scores for ranking checks."""
    x = values["x_hc"].float()
    tokens = x.shape[0]
    flat = x.flatten(1)
    mix = (flat @ values["hc_attn_fn"].T) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
    base, scale = values["hc_attn_base"], values["hc_attn_scale"]
    pre = torch.sigmoid(mix[:, :4] * scale[0] + base[:4]) + FLASH.hc_eps
    post = 2 * torch.sigmoid(mix[:, 4:8] * scale[1] + base[4:8])
    comb = torch.softmax((mix[:, 8:] * scale[2] + base[8:]).reshape(tokens, 4, 4), -1) + FLASH.hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + FLASH.hc_eps)
    for _ in range(FLASH.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + FLASH.hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + FLASH.hc_eps)
    mixed = _bf((x * pre[:, :, None]).sum(1))
    normed = _bf(_rms(mixed, values["attn_norm_w"].float()))
    qr = _rms(normed @ values["wq_a"].float().T, values["gamma_cq"].float())
    qi, qs = _quant(qr)
    # INT8 dot products are exact in FP32 at this K and avoid slow CPU int GEMM.
    q = (qi.float() @ values["wq_b"].float()) * qs * values["wq_b_scale"]
    q = _bf(_rope(_rms(q.reshape(tokens, 64, 512)), values["freqs_cos"], values["freqs_sin"]))
    kv = _bf(
        _rope(
            _rms(normed @ values["wkv"].float(), values["gamma_ckv"].float()),
            values["freqs_cos"],
            values["freqs_sin"],
        )
    )
    for token in range(sum(case.queries)):
        page, row = values["ori_slot_mapping"][token].tolist()
        values["kv_cache"][page, row] = kv[token]
    _compress(values, normed, case)
    _compress(values, normed, case, True)
    iq = (qi.float() @ values["idx_wq_b"].float()) * qs * values["idx_wq_b_scale"]
    iq = _bf(_rope(iq.reshape(tokens, 64, 128), values["freqs_cos"], values["freqs_sin"]))
    iq = (iq @ values["hadamard_idx"].float()) / math.sqrt(128)
    iq, iscale = _quant(iq)
    weights = (normed @ values["weights_proj"].float()) * FLASH.index_weights_scale
    coeff = (iscale.squeeze(-1).half() * weights.half()).half().float()
    full_scores = torch.full((tokens, case.max_length // 4), torch.finfo(torch.float32).min)
    values["idx_topk"].fill_(-1)
    values["idx_topk_scores"].fill_(torch.finfo(torch.float32).min)
    cache = values["idx_native_kv_cache"]
    keys = cache[:, :4096].reshape(-1, 32, 128)
    scales = cache[:, 4096:4160].contiguous().view(torch.float16).reshape(-1, 32)
    begin = 0
    for request, (history, query) in enumerate(zip(case.histories, case.queries)):
        logical = torch.arange((history + query) // 4)
        pages = values["idx_block_table"][request, logical // 32].long()
        k = keys[pages, logical % 32].float()
        ks = scales[pages, logical % 32].float()
        for token in range(begin, begin + query):
            bound = (int(values["position_ids"][token]) + 1) // 4
            score = ((iq[token].float() @ k.T).relu() / 1024).half().float()
            score = (score * coeff[token, :, None]).sum(0) * ks
            full_scores[token, :bound] = score[:bound]
            top_scores, top_indices = score[:bound].topk(512)
            values["idx_topk"][token] = top_indices.int()
            values["idx_topk_scores"][token] = top_scores
        begin += query
    attention = torch.zeros(tokens, 64, 512)
    begin = 0
    for request, (history, query) in enumerate(zip(case.histories, case.queries)):
        for token in range(begin, begin + query):
            pos = int(values["position_ids"][token])
            logical = torch.arange(max(0, pos - 127), pos + 1)
            original = _cache_rows(values["kv_cache"], values["ori_block_table"], request, logical)
            selected = values["idx_topk"][token].long()
            compressed = _cache_rows(values["cmp_kv"], values["cmp_block_table"], request, selected)
            rows = torch.cat((original, compressed)).float()
            running_max = torch.full((64, 1), -float("inf"))
            denominator = torch.zeros(64, 1)
            numerator = torch.zeros(64, 512)
            for start in range(0, rows.shape[0], 128):
                tile = rows[start : start + 128]
                score = (q[token] @ tile.T) / math.sqrt(512)
                maximum = score.max(-1, keepdim=True).values
                probability = (score - maximum).exp()
                partial = _bf(probability) @ tile
                new_max = torch.maximum(running_max, maximum)
                alpha = (running_max - new_max).exp()
                beta = (maximum - new_max).exp()
                numerator = numerator * alpha + partial * beta
                denominator = denominator * alpha + probability.sum(-1, keepdim=True) * beta
                running_max = new_max
            attention[token] = numerator / (denominator + (values["attn_sink"][:, None] - running_max).exp())
        begin += query
    attention = _bf(_rope(attention, values["freqs_cos"], values["freqs_sin"], inverse=True))
    grouped = attention.reshape(tokens, 8, 4096)
    projected = torch.einsum("tgd,gdr->tgr", grouped, values["wo_a"].float())
    oi, oscale = _quant(projected)
    out = torch.zeros(tokens, 4096)
    for group in range(8):
        out += (oi[:, group].float() @ values["wo_b"][group * 1024 : (group + 1) * 1024].float()) * oscale[
            :, group
        ]
    out = _bf(out * values["wo_b_scale"])
    result = out[:, None, :] * post[:, :, None]
    for incoming in range(4):
        result += x[:, incoming, None, :] * comb[:, incoming, :, None]
    values["x_out"].copy_(result.to(torch.bfloat16))
    return full_scores
