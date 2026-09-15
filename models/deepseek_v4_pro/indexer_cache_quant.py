# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CANN indexer-cache FP8 reference, after the BF16 Hadamard output boundary."""


def reference_indexer_cache_quant(x):
    """Return per-row E4M3 payload and FP32 power-of-two dequantization scales.

    ``indexer_quant_cache(quant_mode="fp8")`` defaults to ``round_scale=True``.
    Its Normal kernel uses no amax floor and rounds FP8 elements to nearest-even.
    An all-zero row has zero scale and zero payload. The query quantizer is a
    different operator and uses an unrounded FP32 scale.
    """
    import torch

    values = x.to(torch.bfloat16).float()
    maximum = values.abs().amax(dim=-1, keepdim=True)
    linear_scale = maximum * (1.0 / 448.0)
    scale = torch.exp2(torch.ceil(torch.log2(linear_scale.double()))).float()
    normalized = torch.where(scale == 0, values, values / scale)
    return normalized.to(torch.float8_e4m3fn), scale


def fp8_cache_compare(compare_fn):
    """Widen FP8 exactly for CPU comparison, retaining the mapping/tail checks."""
    import functools

    @functools.wraps(compare_fn)
    def compare(actual, expected, **kwargs):
        return compare_fn(actual.float(), expected.float(), **kwargs)

    return compare
