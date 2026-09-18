# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""BF16 rounding and masking contracts for the paged indexer reference."""

import pytest
import torch

from models.deepseek_v4_1_flash.golden import paged_indexer


@pytest.mark.parametrize("seed", [7, 41])
def test_paged_indexer_bf16_score_boundaries(seed):
    generator = torch.Generator().manual_seed(seed)
    heads, width = 2, 8
    x = torch.randn(2, 4, generator=generator).to(torch.bfloat16)
    query = torch.randn(2, heads * width, generator=generator).to(torch.bfloat16)
    projection = torch.randn(4, heads, generator=generator).to(torch.bfloat16)
    cache = torch.randn(2, 128, 1, width, generator=generator).to(torch.bfloat16)
    request_ids = torch.tensor([0, 1], dtype=torch.int32)
    table = torch.tensor([[1], [0]], dtype=torch.int32)
    lengths = torch.tensor([3, 5], dtype=torch.int32)
    candidates = torch.tensor([[1, 0, 1, 1, 1], [1, 1, 1, 0, 1]], dtype=torch.bool)
    scores, selected = paged_indexer(
        x, query, request_ids, cache, table, lengths,
        torch.eye(heads * width, dtype=torch.bfloat16), None, projection,
        torch.ones(2, 1), torch.zeros(2, 1), candidates, topk=4,
    )

    # Scalar FP64 arithmetic isolates each BF16 tensor boundary from backend GEMMs.
    expected = torch.full((2, 5), -torch.inf)
    scale = width**-0.5 * heads**-0.5
    for token in range(2):
        weights = []
        for head in range(heads):
            projected = (x[token].double() * projection[:, head].double()).sum()
            projected = projected.to(torch.bfloat16).double()
            weights.append((projected * scale).to(torch.bfloat16).double())
        for position in range(int(lengths[token])):
            if not candidates[token, position]:
                continue
            products = []
            for head in range(heads):
                q = query[token, head * width:(head + 1) * width].double()
                k = cache[table[token, 0], position, 0].double()
                dot = (q * k).sum().to(torch.bfloat16).double().clamp_min(0)
                products.append((dot * weights[head]).to(torch.bfloat16).double())
            expected[token, position] = torch.stack(products).sum().to(torch.bfloat16).float()

    torch.testing.assert_close(scores, expected, rtol=0, atol=0)
    assert scores.dtype == torch.float32
    for token in range(2):
        logical = expected[token].topk(4).indices.sort().values
        physical = table[token, 0] * 128 + logical
        valid_rows = physical[torch.isfinite(expected[token, logical])].int()
        padded = torch.full_like(selected[token], -1)
        padded[:valid_rows.numel()] = valid_rows
        torch.testing.assert_close(selected[token], padded, rtol=0, atol=0)
