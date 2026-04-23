# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import warnings

import torch

from .types import GenerateConfig, SamplingParams


class Sampler:
    def sample(self, logits: torch.Tensor, params: SamplingParams) -> int:
        logits = self._sanitize_logits(logits)
        if params.temperature <= 0.0:
            return self._greedy_token(logits)

        scaled = logits / max(params.temperature, 1e-5)

        if params.top_k is not None and params.top_k > 0 and params.top_k < scaled.numel():
            topk_values, topk_indices = torch.topk(scaled, params.top_k)
            filtered = torch.full_like(scaled, float("-inf"))
            filtered[topk_indices] = topk_values
            scaled = filtered

        probs = torch.softmax(scaled, dim=-1)
        if not self._is_valid_distribution(probs):
            return self._greedy_token(logits)

        if 0.0 < params.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            keep = cumulative <= params.top_p
            keep[0] = True
            filtered_probs = torch.zeros_like(probs)
            filtered_probs[sorted_indices[keep]] = probs[sorted_indices[keep]]
            total = filtered_probs.sum()
            if not torch.isfinite(total) or total.item() <= 0.0:
                return self._greedy_token(logits)
            probs = filtered_probs / total
            if not self._is_valid_distribution(probs):
                return self._greedy_token(logits)

        token = torch.multinomial(probs, num_samples=1)
        return int(token.item())

    @staticmethod
    def from_generate_config(config: GenerateConfig) -> SamplingParams:
        return SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
        )

    @staticmethod
    def _sanitize_logits(logits: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        finite_mask = torch.isfinite(logits)
        if finite_mask.all():
            return logits
        warnings.warn("Sampler received non-finite logits; falling back to sanitized values.", stacklevel=2)
        if not finite_mask.any():
            return torch.zeros_like(logits)
        finite_logits = logits[finite_mask]
        floor = finite_logits.min().item() - 1e4
        return torch.nan_to_num(logits, nan=floor, neginf=floor, posinf=finite_logits.max().item())

    @staticmethod
    def _is_valid_distribution(probs: torch.Tensor) -> bool:
        total = probs.sum()
        return bool(torch.isfinite(probs).all() and torch.all(probs >= 0) and torch.isfinite(total) and total.item() > 0.0)

    @staticmethod
    def _greedy_token(logits: torch.Tensor) -> int:
        return int(torch.argmax(logits).item())
