# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""动态 Indexer 压缩输出的前两段：请求内 RMS/RoPE 与真实紧凑行 Hadamard。

整体思路与阅读顺序
    先读 indexer_compressor_norm_rope_stage：AIV 按 Native Q/L 领取完整请求，
    清零该请求本轮真正关闭的 C4 输出行，再把有效池化结果归一化、逆 RoPE，
    按 Native 紧凑前缀直接写 BF16 normed。再读 indexer_compressor_hadamard_stage：
    AIC 对实际 C_valid 行执行原 M16/N64/K128 Hadamard，得到 FP32 结果。
    不再为每请求固定预留 ceil(S/4) 行，不用 T/B 或固定 S 反推 compact 数。

输入输出、布局与动态轴
    pooled 是 FP32[POOL_ROWS,128]，仅活动请求的本轮 token 行由 pool 阶段写入；
    inactive、Q 未覆盖区和容量尾部可以保留 poison。本段只读活动请求区间，
    POOL_ROWS>=T。positions 为 INT64[T]，Q 为 INT32[B+1]、L 为 INT32[B]，
    L 包含本轮 query；offsets 为 metadata 阶段生成的 INT32[B] 请求偏移。
    norm_w 直接接 Native BF16[128]，仅在核内转 FP32；cos/sin 为 Native
    FP32[C_CAPACITY,64] 紧凑表，不展开成 T 行。normed 为 BF16[COMPACT_PAD,128]，
    Hadamard 权重为 ND BF16[128,128]，输出为 FP32[COMPACT_PAD,128]。
    所有容量、请求和 token 轴保留 pl.dynamic，固定 128/64/16 是模型和数学 tile。

真实关闭行、初始化与尾块
    H=L-(Q[r+1]-Q[r])，count=L//4-H//4，prefix=offsets[r]+H//4+1。
    请求内各行先清零，再仅为 causal 且关闭 C4 的 token 覆盖 normed。
    因而有效合同下 [0,C_valid) 每行恰有一位初始化者；position 陈旧时仍留下
    确定零值。不同请求 compact 区间不重叠，无需跨请求初始化 barrier。
    不能把这份零值当作 cache 写入许可；后续 quant/cache 仍检查 active/causal/slot。
    请求内以至多16个原 token为块，每次 load 的 valid_shape 截止到该请求 end，
    不读下一请求或全T中的 dummy。仅取真正关闭行写到密排compact区间。
    C_valid 由同一 Q/L 前缀生产者写 device INT32[1]，外层等待后 read/cast 为
    INDEX 传给 Hadamard；不是 Host 回读，也不是紧凑表的分配容量。
    COMPACT_PAD>=align_up(C_valid,16)。Hadamard只load实际valid行，padding不需
    初始化、也不作为数学输入；C_valid=0时不读normed或Hadamard权重。

数学、所有权与同步
    RMS 保留两段64列平方先相加再row_sum、eps、recip(sqrt)；gamma乘法、
    interleaved逆RoPE和两段最终BF16(rint)顺序不变。Tile gather使用带行偏移
    的扁平索引，等价于原按行的j^1交换，不能把其他行当作当前行。
    AIV按request条带分核；AIC按列块优先的M16/N64网格分核，每个输出块完整
    执行K128，不做split-K。worker_id属于对应角色，要求WORKERS>0且编号有效。
    helper没有spmd/task/notify/wait；norm前等pool及offsets/RoPE，Hadamard前
    等所有norm生产者和C_valid可见。返回Tensor只是GM别名，不是跨核完成。
    本文件不含量化、cache key或64B scale写回；CPU证据不代表整层设备验收。
"""

import pypto.language as pl

from ..common.config import FLASH as M
from .metadata_stages import COMPACT_ROWS_DYN, load_request_compact_rope_rows
from .request_geometry import COMPRESS_RATIO, QUERY_BOUNDS_DYN, REQUESTS_DYN, request_bounds

T_DYN = pl.dynamic("CSA_INDEXER_COMPRESSOR_OUTPUT_T")
POOL_ROWS_DYN = pl.dynamic("CSA_INDEXER_COMPRESSOR_OUTPUT_POOL_ROWS")
COMPACT_PAD_ROWS_DYN = pl.dynamic("CSA_INDEXER_COMPRESSOR_OUTPUT_COMPACT_PAD")
HEAD_DIM = M.index_head_dim
HEAD_DIM_INV = 1.0 / HEAD_DIM
HEAD_TILE = 64
ROPE_HEAD_DIM = M.qk_rope_head_dim
ROPE_DIM_SCALE = float(ROPE_HEAD_DIM)
NOPE_HEAD_DIM = M.index_nope_head_dim
RMS_PAD_TILE = 16
OUT_TILE = 64
EPS = M.rms_norm_eps


@pl.jit.inline(auto_scope=False)
def indexer_compressor_norm_rope_stage(
    pooled: pl.Tensor[[POOL_ROWS_DYN, HEAD_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[COMPACT_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[COMPACT_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    offsets: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    normed: pl.InOut[pl.Tensor[[COMPACT_PAD_ROWS_DYN, HEAD_DIM], pl.BF16]],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """AIV请求所有者先初始化其真实compact行，再消费本请求已经写好的pooled。"""
    for request in pl.range(worker_id, pl.tensor.dim(lengths, 0), WORKERS):
        begin, end, active = request_bounds(bounds, lengths, request, pl.tensor.dim(positions, 0))
        if active != 0:
            length = pl.cast(pl.read(lengths, [request]), pl.INDEX)
            history_start = length - (end - begin)
            count = length // COMPRESS_RATIO - history_start // COMPRESS_RATIO
            offset = pl.cast(pl.read(offsets, [request]), pl.INDEX)
            prefix = offset + history_start // COMPRESS_RATIO + 1
            # 同一request只有一个owner；先清本请求的全部真实输出，随后覆盖不会
            # 与别人的初始化交错。合法前缀保证所有请求连起来恰为[0,C_valid)。
            if count > 0 and prefix >= 0 and prefix + count <= pl.tensor.dim(normed, 0):
                for compact_row in pl.range(prefix, prefix + count):
                    pl.store(pl.tile.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0), [compact_row, 0], normed)
                if end <= pl.tensor.dim(pooled, 0):
                    # Tile gather使用扁平地址：保留原j^1列交换，再加row*64。
                    rope_ones = pl.tile.full([RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
                    rope_index = pl.tile.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32)
                    rope_index_f = pl.cast(rope_index, target_type=pl.FP32)
                    rope_col = pl.col_expand_mul(rope_ones, rope_index_f)
                    rope_dup_f = pl.cast(
                        pl.cast(pl.mul(rope_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32
                    )
                    rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))
                    rope_swap_col = pl.sub(pl.add(rope_col, 1.0), pl.mul(rope_lane, 2.0))
                    row_seed = pl.mul(
                        pl.cast(pl.tile.arange(0, [1, RMS_PAD_TILE], dtype=pl.INT32), target_type=pl.FP32),
                        ROPE_DIM_SCALE,
                    )
                    row_grid = pl.col_expand_mul(
                        pl.tile.full([ROPE_HEAD_DIM, RMS_PAD_TILE], dtype=pl.FP32, value=1.0), row_seed
                    )
                    row_offset = pl.transpose(row_grid, axis1=0, axis2=1)
                    rope_swap_idx = pl.cast(pl.add(rope_swap_col, row_offset), target_type=pl.INT32)
                    reduce_tmp = pl.create_tile(
                        [RMS_PAD_TILE, HEAD_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                    )
                    gather_tmp = pl.create_tile(
                        [RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec
                    )
                    gamma_input = pl.load(norm_w, [0], [NOPE_HEAD_DIM], target_memory=pl.MemorySpace.Vec)
                    gamma = pl.reshape(pl.cast(gamma_input, target_type=pl.FP32), [1, NOPE_HEAD_DIM])
                    gamma_rope_input = pl.load(
                        norm_w, [NOPE_HEAD_DIM], [ROPE_HEAD_DIM], target_memory=pl.MemorySpace.Vec
                    )
                    gamma_rope = pl.reshape(pl.cast(gamma_rope_input, target_type=pl.FP32), [1, ROPE_HEAD_DIM])
                    # 用请求end限制每次GM load，不能将某请求尾部补齐到下一请求。
                    for b0 in pl.range(begin, end, RMS_PAD_TILE):
                        rows = pl.min(RMS_PAD_TILE, end - b0)
                        cos_b, sin_b = load_request_compact_rope_rows(
                            cos, sin, positions, bounds, lengths, offsets, request, b0, rows
                        )
                        kv_rms_low = pl.load(pooled, [b0, 0], [RMS_PAD_TILE, HEAD_TILE], valid_shape=[rows, HEAD_TILE])
                        kv_rms_high = pl.load(
                            pooled, [b0, HEAD_TILE], [RMS_PAD_TILE, HEAD_TILE], valid_shape=[rows, HEAD_TILE]
                        )
                        folded_sq = pl.add(pl.mul(kv_rms_low, kv_rms_low), pl.mul(kv_rms_high, kv_rms_high))
                        square_sum = pl.row_sum(folded_sq, reduce_tmp)
                        variance = pl.add(pl.mul(square_sum, HEAD_DIM_INV), EPS)
                        inv_rms = pl.recip(pl.sqrt(variance))
                        # 两段load即NOPE和RoPE，复用本核Tile，不重读同一GM数据。
                        normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_rms_low, inv_rms), gamma)
                        normed_nope = pl.cast(normed_chunk, target_type=pl.BF16, mode="rint")
                        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rms_high, inv_rms), gamma_rope)
                        swapped = pl.tile.gather(rope_normed, rope_swap_idx, gather_tmp)
                        sin_signed = pl.mul(sin_b, pl.sub(pl.mul(rope_lane, 2.0), 1.0))
                        rope_rot = pl.add(pl.mul(rope_normed, cos_b), pl.mul(swapped, sin_signed))
                        normed_rope = pl.cast(rope_rot, target_type=pl.BF16, mode="rint")
                        for inner in pl.range(rows):
                            token = b0 + inner
                            position = pl.cast(pl.read(positions, [token]), pl.INDEX)
                            expected = history_start + token - begin
                            if position == expected and (position + 1) % COMPRESS_RATIO == 0:
                                compact_row = offset + (position + 1) // COMPRESS_RATIO
                                # 先检查行号再消费紧凑metadata。单位旋转兜底不是
                                # 合法压缩输出标记，容量失配时仅保留初始化的零。
                                if compact_row >= prefix and compact_row < prefix + count:
                                    if compact_row < pl.tensor.dim(cos, 0) and compact_row < pl.tensor.dim(sin, 0):
                                        # inner<rows已由循环保证，明确单行有效域，
                                        # 避免动态父Tile的valid_row推断落到另一表达式。
                                        nope_row = pl.slice(
                                            normed_nope, [1, NOPE_HEAD_DIM], [inner, 0], valid_shape=[1, NOPE_HEAD_DIM]
                                        )
                                        rope_row = pl.slice(
                                            normed_rope, [1, ROPE_HEAD_DIM], [inner, 0], valid_shape=[1, ROPE_HEAD_DIM]
                                        )
                                        pl.store(nope_row, [compact_row, 0], normed)
                                        pl.store(rope_row, [compact_row, NOPE_HEAD_DIM], normed)
    return normed


@pl.jit.inline(auto_scope=False)
def indexer_compressor_hadamard_stage(
    normed: pl.Tensor[[COMPACT_PAD_ROWS_DYN, HEAD_DIM], pl.BF16],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    output: pl.Tensor[[COMPACT_PAD_ROWS_DYN, HEAD_DIM], pl.FP32],
    valid_count: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """AIC消费device实际关闭行数；空C不读输入，尾M16仅load valid_shape行。"""
    rms_blocks = (valid_count + RMS_PAD_TILE - 1) // RMS_PAD_TILE
    padded = rms_blocks * RMS_PAD_TILE
    # 正常入口提前保证容量；此处只防止不一致的device计数造成GM越界，不把
    # 静默截断视为正确数学。不能以normed.shape[0]代替本轮实际C_valid。
    if valid_count > 0 and padded <= pl.tensor.dim(normed, 0) and padded <= pl.tensor.dim(output, 0):
        # 原顺序是N64列块→M16行块。并行分配互不相交的块，不重排块内K128。
        for logical in pl.range(worker_id, (HEAD_DIM // OUT_TILE) * rms_blocks, WORKERS):
            o0 = (logical // rms_blocks) * OUT_TILE
            had_blk = logical % rms_blocks
            hadamard_tile = hadamard[0:HEAD_DIM, o0 : o0 + OUT_TILE]
            had_b0 = had_blk * RMS_PAD_TILE
            had_rows = pl.min(RMS_PAD_TILE, valid_count - had_b0)
            kv_proj_tile = pl.slice(normed, [RMS_PAD_TILE, HEAD_DIM], [had_b0, 0], valid_shape=[had_rows, HEAD_DIM])
            kv_hadamard_acc = pl.matmul(kv_proj_tile, hadamard_tile, out_dtype=pl.FP32)
            output[had_b0 : had_b0 + RMS_PAD_TILE, o0 : o0 + OUT_TILE] = kv_hadamard_acc
    return output
