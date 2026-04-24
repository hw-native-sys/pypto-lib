# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import math

import torch

from .kv_cache import KvCacheManager
from .types import DecodeBatch, DecodeResult, LayerWeights, PrefillBatch, PrefillResult, RuntimeModel


class ModelExecutor:
    def __init__(self, kv_cache_manager: KvCacheManager) -> None:
        self._kv_cache_manager = kv_cache_manager

    def lookup_embeddings(self, model: RuntimeModel, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.to(device=model.runtime.device, dtype=torch.long)
        return model.embed_tokens.index_select(0, token_ids.view(-1)).view(*token_ids.shape, model.config.hidden_size)

    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        last_hidden_rows: list[torch.Tensor] = []
        logits_rows: list[torch.Tensor] = []
        for batch_idx, alloc in enumerate(batch.kv_allocations):
            hidden = batch.input_embeddings[batch_idx].to(model.runtime.device).float()
            seq_len = int(batch.seq_lens[batch_idx].item())
            hidden = hidden[:seq_len]
            positions = torch.arange(seq_len, device=model.runtime.device, dtype=torch.long)

            for layer_idx, layer in enumerate(model.layers):
                hidden = self._layer_prefill(
                    model=model,
                    layer_idx=layer_idx,
                    layer=layer,
                    hidden_states=hidden,
                    positions=positions,
                    alloc=alloc,
                )

            last_hidden = hidden[-1]
            last_hidden_rows.append(last_hidden)
            logits_rows.append(self._project_logits(model, last_hidden))
        return PrefillResult(last_hidden=torch.stack(last_hidden_rows), logits=torch.stack(logits_rows))

    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        hidden_rows: list[torch.Tensor] = []
        logits_rows: list[torch.Tensor] = []
        for batch_idx, alloc in enumerate(batch.kv_allocations):
            hidden = batch.hidden_states[batch_idx].to(model.runtime.device).float()
            position = int(batch.seq_lens[batch_idx].item()) - 1

            for layer_idx, layer in enumerate(model.layers):
                hidden = self._layer_decode(
                    model=model,
                    layer_idx=layer_idx,
                    layer=layer,
                    hidden_state=hidden,
                    position=position,
                    alloc=alloc,
                )

            hidden_rows.append(hidden)
            logits_rows.append(self._project_logits(model, hidden))
        return DecodeResult(hidden_states=torch.stack(hidden_rows), logits=torch.stack(logits_rows))

    def _layer_prefill(
        self,
        model: RuntimeModel,
        layer_idx: int,
        layer: LayerWeights,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        alloc,
    ) -> torch.Tensor:
        config = model.config
        normed = self._rms_norm(hidden_states, layer.input_rms_weight, config.rms_norm_eps)
        q = self._linear(normed, layer.wq).view(-1, config.num_attention_heads, config.head_dim)
        k = self._linear(normed, layer.wk).view(-1, config.num_key_value_heads, config.head_dim)
        v = self._linear(normed, layer.wv).view(-1, config.num_key_value_heads, config.head_dim)
        q = self._per_head_rms_norm(q, layer.q_norm_weight, config.rms_norm_eps)
        k = self._per_head_rms_norm(k, layer.k_norm_weight, config.rms_norm_eps)
        q = self._apply_rope(q, positions, config.rope_theta)
        k = self._apply_rope(k, positions, config.rope_theta)
        self._kv_cache_manager.write_tokens(layer_idx, alloc, 0, k.to(model.runtime.device), v.to(model.runtime.device))
        attn_out = self._attention_prefill(q, k, v, config.num_attention_heads, config.num_key_value_heads)
        attn_resid = hidden_states + self._linear(attn_out.reshape(hidden_states.shape[0], -1), layer.wo)
        mlp_normed = self._rms_norm(attn_resid, layer.post_rms_weight, config.rms_norm_eps)
        gate = self._linear(mlp_normed, layer.w_gate)
        up = self._linear(mlp_normed, layer.w_up)
        mlp = torch.nn.functional.silu(gate) * up
        return attn_resid + self._linear(mlp, layer.w_down)

    def _layer_decode(
        self,
        model: RuntimeModel,
        layer_idx: int,
        layer: LayerWeights,
        hidden_state: torch.Tensor,
        position: int,
        alloc,
    ) -> torch.Tensor:
        config = model.config
        normed = self._rms_norm(hidden_state.unsqueeze(0), layer.input_rms_weight, config.rms_norm_eps)
        q = self._linear(normed, layer.wq).view(config.num_attention_heads, config.head_dim)
        k = self._linear(normed, layer.wk).view(config.num_key_value_heads, config.head_dim)
        v = self._linear(normed, layer.wv).view(config.num_key_value_heads, config.head_dim)
        q = self._per_head_rms_norm(q.unsqueeze(0), layer.q_norm_weight, config.rms_norm_eps).squeeze(0)
        k = self._per_head_rms_norm(k.unsqueeze(0), layer.k_norm_weight, config.rms_norm_eps).squeeze(0)
        pos = torch.tensor([position], device=model.runtime.device, dtype=torch.long)
        q = self._apply_rope(q.unsqueeze(0), pos, config.rope_theta).squeeze(0)
        k = self._apply_rope(k.unsqueeze(0), pos, config.rope_theta).squeeze(0)
        self._kv_cache_manager.write_tokens(
            layer_idx,
            alloc,
            position,
            k.unsqueeze(0).to(model.runtime.device),
            v.unsqueeze(0).to(model.runtime.device),
        )
        k_ctx, v_ctx = self._kv_cache_manager.read_context(layer_idx, alloc)
        attn_out = self._attention_decode(q, k_ctx, v_ctx, config.num_attention_heads, config.num_key_value_heads)
        attn_resid = hidden_state + self._linear(attn_out.reshape(1, -1), layer.wo).squeeze(0)
        mlp_normed = self._rms_norm(attn_resid.unsqueeze(0), layer.post_rms_weight, config.rms_norm_eps)
        gate = self._linear(mlp_normed, layer.w_gate)
        up = self._linear(mlp_normed, layer.w_up)
        mlp = torch.nn.functional.silu(gate) * up
        return attn_resid + self._linear(mlp, layer.w_down).squeeze(0)

    def _project_logits(self, model: RuntimeModel, hidden: torch.Tensor) -> torch.Tensor:
        squeeze = hidden.dim() == 1
        hidden_2d = hidden.unsqueeze(0) if squeeze else hidden
        normed = self._rms_norm(hidden_2d, model.final_norm_weight, model.config.rms_norm_eps)
        logits = self._linear(normed, model.lm_head)
        return logits.squeeze(0) if squeeze else logits

    @staticmethod
    def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return x.float() @ weight.float().transpose(0, 1)

    @staticmethod
    def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return x.float() * torch.rsqrt(variance + eps) * weight.float().view(1, -1)

    @staticmethod
    def _per_head_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return x.float() * torch.rsqrt(variance + eps) * weight.float().view(1, 1, -1)

    @staticmethod
    def _apply_rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
        head_dim = x.shape[-1]
        half = head_dim // 2
        device = x.device
        inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
        freqs = torch.outer(positions.float(), inv_freq)
        cos = freqs.cos().unsqueeze(1)
        sin = freqs.sin().unsqueeze(1)
        x_lo = x[..., :half]
        x_hi = x[..., half:]
        return torch.cat([x_lo * cos - x_hi * sin, x_hi * cos + x_lo * sin], dim=-1)

    @staticmethod
    def _attention_prefill(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
    ) -> torch.Tensor:
        q_per_kv = num_heads // num_kv_heads
        k_rep = k.repeat_interleave(q_per_kv, dim=1).permute(1, 0, 2)
        v_rep = v.repeat_interleave(q_per_kv, dim=1).permute(1, 0, 2)
        q_heads = q.permute(1, 0, 2)
        scores = torch.matmul(q_heads, k_rep.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        seq_len = q.shape[0]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v_rep).permute(1, 0, 2)

    @staticmethod
    def _attention_decode(
        q: torch.Tensor,
        k_ctx: torch.Tensor,
        v_ctx: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
    ) -> torch.Tensor:
        k_ctx = k_ctx.float()
        v_ctx = v_ctx.float()
        q_per_kv = num_heads // num_kv_heads
        k_rep = k_ctx.repeat_interleave(q_per_kv, dim=1).permute(1, 0, 2)
        v_rep = v_ctx.repeat_interleave(q_per_kv, dim=1).permute(1, 0, 2)
        q_heads = q.unsqueeze(1)
        scores = torch.matmul(q_heads, k_rep.transpose(-1, -2)).squeeze(1) / math.sqrt(q.shape[-1])
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn.unsqueeze(1), v_rep).squeeze(1)
