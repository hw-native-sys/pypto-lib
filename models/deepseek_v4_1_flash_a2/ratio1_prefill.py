# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Ratio1 BF16 compressor and index publication for A2 Prefill.

QKV and SWA use the existing consumer producer. This stage owns the ratio1
latent and long/index caches. The index projection consumes the unrotated
latent, while separate buffers carry its current-position rotary views.
"""

import pypto.language as pl

from .c1c2 import make_compressed_cache_publish
from .layer2_c1c2 import make_indexer_cache_publish_128
from .ops import make_norm, make_rope
from .qkv_proj_rope import make_bf16_projection_staged




def make_ratio1_compress_publish(capacity, config, cache_blocks=2, request_count=1):
    """Project each token once and publish its own-position BF16/INT8 rows."""
    hidden = config["hidden_size"]
    head_dim = config["head_dim"]
    index_width = config["index_head_dim"]
    rope_dim = config["qk_rope_head_dim"]
    pairs = rope_dim // 2
    if head_dim != 512 or index_width != 128:
        raise ValueError("Use the original full512 compressor and128-wide index cache")
    norm_compressed = make_norm(capacity, head_dim, config["rms_norm_eps"])
    project_index = make_bf16_projection_staged(head_dim, index_width)
    norm_index = make_norm(capacity, index_width, config["rms_norm_eps"])
    rope_compressed = make_rope(capacity, 1, head_dim, rope_dim)
    rope_index = make_rope(capacity, 1, index_width, rope_dim)
    publish_compressed = make_compressed_cache_publish(capacity, cache_blocks, head_dim)
    publish_index = make_indexer_cache_publish_128(capacity, cache_blocks)

    @pl.jit
    def ratio1_compress_publish(
        projected: pl.Tensor[[capacity, head_dim], pl.BF16],
        compressor_norm: pl.Tensor[[head_dim], pl.BF16],
        index_wk: pl.Tensor[[head_dim, index_width], pl.BF16],
        index_norm: pl.Tensor[[index_width], pl.BF16],
        source_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        source_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        positions: pl.Tensor[[capacity], pl.INT64],
        requests: pl.Tensor[[capacity], pl.INT32],
        compressed_slots: pl.Tensor[[capacity], pl.INT64],
        compressed: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        long_cache: pl.InOut[pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16]],
        index_key: pl.InOut[pl.Tensor[[capacity, index_width], pl.BF16]],
        key_cache: pl.InOut[pl.Tensor[[cache_blocks, 128, 1, index_width], pl.INT8]],
        key_scale: pl.InOut[pl.Tensor[[cache_blocks, 128, 1, 1], pl.FP16]],
        source_positions: pl.InOut[pl.Tensor[[capacity], pl.INT64]],
        cache_lengths: pl.InOut[pl.Tensor[[request_count, 2], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        norm_compressed(projected, compressor_norm, compressed, active)
        rotated = pl.create_tensor([capacity, head_dim], dtype=pl.BF16)
        rope_compressed(compressed, source_cos, source_sin, rotated, active)
        publish_compressed(rotated, compressed_slots, long_cache, active)
        projected_index = pl.create_tensor([capacity, index_width], dtype=pl.BF16)
        project_index(compressed, index_wk, projected_index, active)
        norm_index(projected_index, index_norm, index_key, active)
        rope_index(index_key, source_cos, source_sin, index_key, active)
        publish_index(index_key, compressed_slots, key_cache, key_scale, active)
        with pl.spmd(1, name_hint="ratio1_lengths_and_sources"):
            block = pl.tile.get_block_idx()
            if block == 0:
                for token in pl.range(active):
                    position = pl.read(positions, [token])
                    request = pl.read(requests, [token])
                    length = pl.cast(position + 1, pl.INT32)
                    pl.write(source_positions, [token], position)
                    pl.write(cache_lengths, [request, 0], length)
                    pl.write(cache_lengths, [request, 1], length)
        return compressed, long_cache, index_key, key_cache, key_scale, source_positions, cache_lengths

    return ratio1_compress_publish
