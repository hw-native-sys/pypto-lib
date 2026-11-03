# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""动态 CSA 主512列压缩后处理：请求内RMSNorm、紧凑RoPE与Native cache写回。

范围与数据流
    本文件只迁移decode_compressor_ratio4.py的归一化、RoPE和压缩cache段。
    输入pooled_kv是在线pool产生的FP32原始token行；gamma保持Native BF16。
    Native Q/L决定每个活动请求的[begin,end)，positions仍是原始TND顺序。
    compact cos/sin及slot按Native关闭组顺序传入，不展开成T行metadata。
    normed_kv保留原FP32中间量；kv_debug只在合法关闭组且合法cache槽时写回，
    与当前canonical的调试输出语义一致，不能把所有token都当作压缩输出。

分工、动态容量与同步
    显式AIV worker按request%WORKERS领取请求，并在请求内部以至多16行遍历。
    即使请求起点不对齐、query数不同，load/store的valid_shape也不跨请求。
    T从positions真实轴取；pooled/normed容量独立动态，均至少覆盖T行。上游
    pool只初始化活动请求的原始行，本段不能遍历未覆盖的padding或inactive行。
    外层在调用前等pool和compact_offsets写回可见，并保证Native RoPE可读；
    helper不提交task、不notify/wait。全部本核wave结束后外层才能发布cache阶段。
    本段不提交state，pool读完→state覆盖的全局依赖仍由外层独立保留。
    Native cache由外层提供同GM的二维[pages*32,512]别名；不在incore内reshape
    整张GM。normed/debug/cache标InOut保留未写区，调用者仍须保证有效写槽唯一。

数学与边界
    平方和保持256→128→64折叠再row_sum，随后eps和recip(sqrt)，不改变归约。
    gamma在UB由BF16无损转FP32；NOPE及RoPE沿原乘法次序，RoPE的sin符号在
    消费端按偶负奇正折入。cache最后FP32→BF16(rint)，debug保留FP32。
    cache副作用先检查active、position的causal一致性和C4关闭条件，再检查
    compact行非负及三份Native表各自容量，之后才读slot。最后独立验证page、
    offset及实际cache行范围；禁止旧safe_row先读表再判断的顺序，C=0也安全。
    当前只做源级迁移和CPU证据；不据此宣称设备精度、同步或性能已经通过。
"""

import pypto.language as pl

from ..common.config import BLOCK_SIZE
from ..common.config import FLASH as M
from .metadata_stages import COMPACT_ROWS_DYN, load_request_compact_rope_rows
from .request_geometry import COMPRESS_RATIO, QUERY_BOUNDS_DYN, REQUESTS_DYN, request_bounds

T_DYN = pl.dynamic("CSA_COMPRESSOR_OUTPUT_T")
POOLED_ROWS_DYN = pl.dynamic("CSA_COMPRESSOR_OUTPUT_POOL_ROWS")
NORMED_ROWS_DYN = pl.dynamic("CSA_COMPRESSOR_OUTPUT_NORM_ROWS")
COMPACT_SLOT_ROWS_DYN = pl.dynamic("CSA_COMPRESSOR_OUTPUT_SLOT_ROWS")
CACHE_ROWS_DYN = pl.dynamic("CSA_COMPRESSOR_OUTPUT_CACHE_ROWS")
HEAD_DIM = M.head_dim
HEAD_DIM_INV = 1.0 / HEAD_DIM
ROPE_HEAD_DIM = M.qk_rope_head_dim
NOPE_HEAD_DIM = M.nope_head_dim
EPS = M.rms_norm_eps
HEAD_TILE = 64
RMS_PAD_TILE = 16


@pl.jit.inline(auto_scope=False)
def compressor_norm_rope_cache_stage(
    pooled_kv: pl.Tensor[[POOLED_ROWS_DYN, HEAD_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[COMPACT_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[COMPACT_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    compact_offsets: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    compact_slots: pl.Tensor[[COMPACT_SLOT_ROWS_DYN, 2], pl.INT32],
    normed_kv: pl.InOut[pl.Tensor[[NORMED_ROWS_DYN, HEAD_DIM], pl.FP32]],
    kv_debug: pl.InOut[pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32]],
    cache_flat: pl.InOut[pl.Tensor[[CACHE_ROWS_DYN, HEAD_DIM], pl.BF16]],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """AIV请求条带；保留未写区，返回三个原GM别名，由外层发布完成信号。"""
    tokens = pl.tensor.dim(positions, 0)
    # 领取请求而非全局M16块：相邻请求可以是2/2/2或4/2，T相同也不会混读。
    for request in pl.range(worker_id, pl.tensor.dim(lengths, 0), WORKERS):
        begin, end, active = request_bounds(bounds, lengths, request, tokens)
        if active != 0:
            length = pl.cast(pl.read(lengths, [request]), pl.INDEX)
            history_start = length - (end - begin)
            compact_offset = pl.cast(pl.read(compact_offsets, [request]), pl.INDEX)
            for first in pl.range(begin, end, RMS_PAD_TILE):
                rows = pl.min(RMS_PAD_TILE, end - first)
                cos_tile, sin_tile = load_request_compact_rope_rows(
                    cos, sin, positions, bounds, lengths, compact_offsets, request, first, rows
                )
                cos_b = pl.set_validshape(cos_tile, rows, ROPE_HEAD_DIM)
                sin_b = pl.set_validshape(sin_tile, rows, ROPE_HEAD_DIM)
                # 保留原平方和的加法树；只缩小有效行，不沿列重排或重新分组。
                rms_low = pl.load(
                    pooled_kv, [first, 0], [RMS_PAD_TILE, HEAD_DIM // 2], valid_shape=[rows, HEAD_DIM // 2]
                )
                rms_high = pl.load(
                    pooled_kv,
                    [first, HEAD_DIM // 2],
                    [RMS_PAD_TILE, HEAD_DIM // 2],
                    valid_shape=[rows, HEAD_DIM // 2],
                )
                folded4 = pl.add(pl.mul(rms_low, rms_low), pl.mul(rms_high, rms_high))
                folded2 = pl.add(folded4[:, 0 : HEAD_DIM // 4], folded4[:, HEAD_DIM // 4 : HEAD_DIM // 2])
                folded1 = pl.add(folded2[:, 0:HEAD_TILE], folded2[:, HEAD_TILE : 2 * HEAD_TILE])
                # 显式load返回Tile，row_sum API需同形临时Tile；这不改变归约算法。
                square_sum = pl.row_sum(folded1, pl.tile.create([RMS_PAD_TILE, HEAD_TILE], dtype=pl.FP32))
                variance = pl.add(pl.mul(square_sum, HEAD_DIM_INV), EPS)
                inv_rms = pl.recip(pl.sqrt(variance))
                for k0 in pl.range(0, NOPE_HEAD_DIM, HEAD_TILE):
                    kv_norm_chunk = pl.load(
                        pooled_kv, [first, k0], [RMS_PAD_TILE, HEAD_TILE], valid_shape=[rows, HEAD_TILE]
                    )
                    # 只对已load的64元素UB tile reshape；不把整个GM norm或pooled变为Tile。
                    gamma = pl.cast(pl.reshape(pl.load(norm_w, [k0], [HEAD_TILE]), [1, HEAD_TILE]), pl.FP32)
                    normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
                    pl.store(pl.set_validshape(normed_chunk, rows, HEAD_TILE), [first, k0], normed_kv)

                kv_rope_norm = pl.load(
                    pooled_kv,
                    [first, NOPE_HEAD_DIM],
                    [RMS_PAD_TILE, ROPE_HEAD_DIM],
                    valid_shape=[rows, ROPE_HEAD_DIM],
                )
                gamma_rope = pl.cast(
                    pl.reshape(pl.load(norm_w, [NOPE_HEAD_DIM], [ROPE_HEAD_DIM]), [1, ROPE_HEAD_DIM]), pl.FP32
                )
                rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
                # 索引运算也用Tile层，避免把Tensor索引与显式load的Tile混用。
                rope_ones = pl.tile.full([RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
                rope_index = pl.tile.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32)
                rope_index_f = pl.cast(rope_index, target_type=pl.FP32)
                rope_col = pl.col_expand_mul(rope_ones, rope_index_f)
                rope_dup_f = pl.cast(
                    pl.cast(pl.mul(rope_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32
                )
                rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))
                rope_swap_idx = pl.cast(pl.sub(pl.add(rope_col, 1.0), pl.mul(rope_lane, 2.0)), target_type=pl.INT32)
                # Tile gather按全Tile寻址，必须补每行64列偏移；原Tensor gather的
                # dim=-1是逐行索引。两者对有效行均读取同一相邻列，不能漏掉行偏移。
                rope_row_offsets = pl.reshape(
                    pl.tile.muls(pl.tile.ci(0, [1, RMS_PAD_TILE], dtype=pl.INT32), ROPE_HEAD_DIM),
                    [RMS_PAD_TILE, 1],
                )
                rope_abs_idx = pl.row_expand_add(rope_swap_idx, rope_row_offsets)
                rope_abs_idx = pl.set_validshape(rope_abs_idx, rows, ROPE_HEAD_DIM)
                swapped = pl.tile.gather(
                    rope_normed, rope_abs_idx, pl.tile.create([RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.INT32)
                )
                sin_signed = pl.mul(sin_b, pl.sub(pl.mul(rope_lane, 2.0), 1.0))
                rope_rot = pl.add(pl.mul(rope_normed, cos_b), pl.mul(swapped, sin_signed))
                pl.store(pl.set_validshape(rope_rot, rows, ROPE_HEAD_DIM), [first, NOPE_HEAD_DIM], normed_kv)

                for inner in pl.range(rows):
                    token = first + inner
                    position = pl.cast(pl.read(positions, [token]), pl.INDEX)
                    # 先拒绝capture遗留position，再判关闭组；禁止读dummy/负compact行的slot。
                    if position == history_start + token - begin and (position + 1) % COMPRESS_RATIO == 0:
                        compact_row = compact_offset + (position + 1) // COMPRESS_RATIO
                        if (
                            compact_row >= 0
                            and compact_row < pl.tensor.dim(compact_slots, 0)
                            and compact_row < pl.tensor.dim(cos, 0)
                            and compact_row < pl.tensor.dim(sin, 0)
                        ):
                            page = pl.cast(pl.read(compact_slots, [compact_row, 0]), pl.INDEX)
                            offset = pl.cast(pl.read(compact_slots, [compact_row, 1]), pl.INDEX)
                            if (
                                page >= 0
                                and page < pl.tensor.dim(cache_flat, 0) // BLOCK_SIZE
                                and offset >= 0
                                and offset < BLOCK_SIZE
                            ):
                                cache_row = page * BLOCK_SIZE + offset
                                if cache_row < pl.tensor.dim(cache_flat, 0):
                                    kv_row_fp32 = normed_kv[token : token + 1, 0:HEAD_DIM]
                                    kv_debug[token : token + 1, :] = kv_row_fp32
                                    cache_flat[cache_row : cache_row + 1, :] = pl.cast(
                                        kv_row_fp32, target_type=pl.BF16, mode="rint"
                                    )
    return normed_kv, kv_debug, cache_flat
