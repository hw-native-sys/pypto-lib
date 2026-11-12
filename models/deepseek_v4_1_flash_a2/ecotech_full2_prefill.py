# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Eco-Tech Full2 Prefill stages with actual intermediate observation ports."""

import pypto.language as pl

from .c1c2 import make_compressed_cache_publish
from .ecotech_full2_observed import (
    make_c2_program, make_indexer_projection_observed, make_indexer_query_quantize_observed,
)
from .ecotech_full2_readback import make_readback_entries
from .ecotech_swa_prefill import GEOMETRY, make_builders
from .hc import make_hc_post_inline
from .decode_indexer import make_empty_selection, make_sparse_controls
from .rmsnorm import make_prefill_norm
from .sparse_consumer_prefill import make_consumer_sparse_core


def make_entries(active, *, compressor_projected=False):
    """Build complete query, Full2 producer, sparse and BF16 output stages."""
    if type(active) is not int or active not in (1, 7, 16):
        raise ValueError("Original bounded active1/7/16 required")
    if type(compressor_projected) is not bool:
        raise ValueError("Select the actual product compressor projection boundary explicitly")
    config = dict(GEOMETRY)
    base = make_builders(16, config)
    names = (
        "wq_a", "q_norm", "wq_b", "q_rope", "wkv", "kv_norm", "kv_rope",
        "inverse_rope", "grouped_wo_a", "wo_b",
    )
    entries = {name: base[name] for name in names}

    @pl.jit
    def narrow_sparse_BF16(
        source: pl.InOut[pl.Tensor[[16, 32768], pl.FP32]],
        output: pl.InOut[pl.Tensor[[16, 32768], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(active * 128, name_hint="L2_sparse_native_BF16_boundary"):
            token = block // 128
            column = block % 128 * 256
            values = source[token : token + 1, column : column + 256]
            output[token : token + 1, column : column + 256] = pl.cast(values, pl.BF16, mode="rint")
        return source, output

    entries.update(
        attention_norm=make_prefill_norm(16, 5120),
        c2_project_pool_publish=make_c2_program(config, project_inputs=not compressor_projected),
        swa_publish=pl.jit(make_compressed_cache_publish(16, 2, 512)._func),
        index_projection=make_indexer_projection_observed(16, config),
        index_quantize=make_indexer_query_quantize_observed(16),
        empty_selection=make_empty_selection(16),
        sparse_controls=make_sparse_controls(16, 1, 2, 2, long_page_rows=64, compress_ratio=2),
        sparse_core=make_consumer_sparse_core(active, config, compressed_blocks=1),
        narrow_sparse_BF16=narrow_sparse_BF16,
        HC_post=pl.jit(make_hc_post_inline(16, config)._func),
    )
    return make_readback_entries(entries, active)
