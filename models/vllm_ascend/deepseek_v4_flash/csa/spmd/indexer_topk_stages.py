# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""动态CSA Indexer的纯Top-512排序/归并：显式范围、原同分规则，无调度。

一、阅读顺序与范围
    indexer_topk_half_leaf把最多4096个连续候选写成一个512项根；
    indexer_topk_segment_pairs处理最多2048候选并返回同核UB根；
    merge2_top512_pairs合并两份GM根，merge_top512_roots按原四路层级
    将任意有效根数合成一份UB根。源码来自当前decode_indexer.py相应四段，
    不导入该整模块、固定S、历史IR或静态entry；不负责QK/请求推导/最终发布。

二、score与pair布局
    score_arena是FP32[SCORE_ROWS,8192]，行容量独立pl.dynamic；每行只有
    调用者给出的有效候选前缀可读。logical_begin将局部列映射为全局压缩KV
    位置，不能拿8192容量代替valid_count。pair_arena是FP32[PAIR_ROWS,1024]，
    每两项为(score,index位编码)，总共512对；index不是浮点数值转换，而是
    sort32接受的UINT32位表示。后继用gather_mask分别提取score和INT32 index。

三、调用者合同与同步
    四个 helper 都使用 Vector 排序/归并指令，必须在 AIV 上内联执行；
    AIC 只生产上游 QK 数据，不能直接调用这些 helper 或消费其 UB 返回值。
    half_leaf要求1<=valid_count<=4096；segment要求1<=valid_count<=2048，
    score_offset是固定小偏移(当前用0或2048)，分支的完整load tile必须位于
    8192列内。merge要求half_count>=1且arena_base+half_count不越容量；所有
    源行都是本代已写的有序512根。零候选由调用者写最小值根/最终-1结果，
    本文件不偷偷扩展原分支语义。所有行号、logical_begin/计数是运行时Scalar。
    helper内部没有spmd/submit/notify；调用者先等生产者、完成业务cache维护，
    再执行本核数学。pair写参数InOut保留其它行，输出槽不得被另一核同时读写。
    merge2可复用自己的左/右输入槽：两份根先load进UB后才store；跨核工作区
    复用仍需外层依赖。返回Tile仅能由同核继续消费，不能当成跨核共享GM。

四、数学与临时区
    sort32先将32个score和index配成有序run；mrgsort的block_len以交错pair
    的FP32元素数计，64/256/1024分别对应32/128/512候选的run。按原调用序列
    归并，不以“TopK集合相同”为理由改变同分优先级。fillpad(min)后再maximum
    模型有限FP32_NEG_INF，屏蔽尾部旧值。每次merge scratch由tile.create独立
    申请；tile.extract把循环携带root明确限制为1024个FP32，避免携带大scratch。
    中文注释说明关键阶段，但运算树、排序分支和输入先后保持canonical原样。
    当前数学AST及CPU编译不等于设备TopK/同分结果或性能已经验证。

B1分数驻留UB
    indexer_topk_half_leaf_resident对应同文件的GM半叶helper，只将输入
    搬运改为UB等宽extract；六条候选长度分支及每次MrgSort输入先后必须
    保持一致。不能把全部长度补到4096统一排序，否则同分Top-K会改变。
    保留GM接口供原路径使用，避免给其它资源区间引入新的搬运或Tile存活期。
"""

import pypto.language as pl

from ..common.config import FLASH as M
from ..common.config import FP32_NEG_INF

IDX_TOPK = M.index_topk
TOPK_PAIR_WIDTH = 2 * IDX_TOPK
TOPK_CANDIDATES_PER_LEAF = 8192
TOPK_MERGE_FAN_IN = 4
SCORE_ROWS_DYN = pl.dynamic("CSA_INDEX_TOPK_SCORE_ROWS")
PAIR_ROWS_DYN = pl.dynamic("CSA_INDEX_TOPK_PAIR_ROWS")


@pl.jit.inline(auto_scope=False)
def merge2_top512_pairs(
    pair_arena: pl.InOut[pl.Tensor[[PAIR_ROWS_DYN, TOPK_PAIR_WIDTH], pl.FP32]],
    left_slot: pl.Scalar[pl.INDEX],
    right_slot: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
) -> None:
    """先读左右512根，再按incoming右根优先合并，写到指定GM行。"""
    left = pl.load(pair_arena, [left_slot, 0], [1, TOPK_PAIR_WIDTH])
    right = pl.load(pair_arena, [right_slot, 0], [1, TOPK_PAIR_WIDTH])
    merge_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
    # 新到的right放在已有left之前；同分时输入位置影响index顺序，不能交换。
    merged_all = pl.tile.mrgsort(right, left, tmp=merge_tmp)
    merged = pl.tile.slice(merged_all, [1, TOPK_PAIR_WIDTH], [0, 0])
    pl.store(merged, [output_slot, 0], pair_arena)


@pl.jit.inline(auto_scope=False)
def merge_top512_roots(
    pair_arena: pl.Tensor[[PAIR_ROWS_DYN, TOPK_PAIR_WIDTH], pl.FP32],
    arena_base: pl.Scalar[pl.INDEX],
    half_count: pl.Scalar[pl.INDEX],
) -> pl.Tile:
    """根0起步，每轮合并至多三个新根，始终只把1024元素root带到下一轮。"""
    root = pl.load(pair_arena, [arena_base, 0], [1, TOPK_PAIR_WIDTH])
    # 一个root已占一路，所以每次跨3个新根。末轮1/2/3个新根分别用2/3/4路。
    # 这里只读取half_count指定的已写根；PAIR_ROWS容量较大不代表其它根有效。
    for child in pl.range(1, half_count, TOPK_MERGE_FAN_IN - 1):
        incoming1 = pl.load(pair_arena, [arena_base + child, 0], [1, TOPK_PAIR_WIDTH])
        if child + 2 < half_count:
            incoming2 = pl.load(pair_arena, [arena_base + child + 1, 0], [1, TOPK_PAIR_WIDTH])
            incoming3 = pl.load(pair_arena, [arena_base + child + 2, 0], [1, TOPK_PAIR_WIDTH])
            merge_tmp4 = pl.tile.create([1, TOPK_MERGE_FAN_IN * TOPK_PAIR_WIDTH], dtype=pl.FP32)
            # 后到的根排前：incoming3→2→1→旧root。exhausted沿原四路调用保留。
            merged4 = pl.tile.mrgsort(incoming3, incoming2, incoming1, root, tmp=merge_tmp4, exhausted=True)
            # 从大merge输出实取1024元素，使下一轮root不携带4096元素存储生存期。
            root = pl.tile.extract(merged4, 0, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec)
        elif child + 1 < half_count:
            incoming2 = pl.load(pair_arena, [arena_base + child + 1, 0], [1, TOPK_PAIR_WIDTH])
            merge_tmp3 = pl.tile.create([1, 3 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
            merged3 = pl.tile.mrgsort(incoming2, incoming1, root, tmp=merge_tmp3, exhausted=True)
            root = pl.tile.extract(merged3, 0, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec)
        else:
            merge_tmp2 = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
            merged2 = pl.tile.mrgsort(incoming1, root, tmp=merge_tmp2)
            root = pl.tile.extract(merged2, 0, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec)
    return root


@pl.jit.inline(auto_scope=False)
def indexer_topk_half_leaf(
    score_arena: pl.Tensor[[SCORE_ROWS_DYN, TOPK_CANDIDATES_PER_LEAF], pl.FP32],
    pair_arena: pl.InOut[pl.Tensor[[PAIR_ROWS_DYN, TOPK_PAIR_WIDTH], pl.FP32]],
    score_row: pl.Scalar[pl.INDEX],
    logical_begin: pl.Scalar[pl.INDEX],
    valid_count: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
) -> None:
    """连续半叶1..4096候选转成512项(score,index)根，保持原分支同分顺序。"""
    logical_begin_i32 = pl.cast(logical_begin, pl.INT32)
    if valid_count <= 512:
        # 即使只有一项，也按512候选的固定tile排序；尾列由valid_shape和fillpad
        # 写成最小score。index先按全局位置生成，再只按位解释为UINT32给sort32。
        short_indices = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), logical_begin_i32)
        short_raw = pl.load(score_arena, [score_row, 0], [1, 512], valid_shape=[1, valid_count])
        short_scores = pl.tile.fillpad(short_raw, pad_value=pl.PadValue.min)
        short_scores = pl.maximum(short_scores, FP32_NEG_INF)
        short_pairs = pl.sort32(short_scores, pl.reinterpret_view(short_indices, pl.UINT32))
        # 32候选局部run→128候选run→512候选根；长度64/256是pair元素数。
        short_pairs = pl.mrgsort(short_pairs, block_len=64)
        short_pairs = pl.mrgsort(short_pairs, block_len=256)
        pl.store(short_pairs, [output_slot, 0], pair_arena)
    elif valid_count <= 1024:
        # 两个512根分开取出后按原left/right顺序合并；不能套用其它分支的
        # “后段优先”，因为1024分支的原输入顺序与2560/4096分支不同。
        small_indices = pl.add(pl.tile.arange(0, [1, 1024], dtype=pl.INT32), logical_begin_i32)
        small_raw = pl.load(score_arena, [score_row, 0], [1, 1024], valid_shape=[1, valid_count])
        small_scores = pl.tile.fillpad(small_raw, pad_value=pl.PadValue.min)
        small_scores = pl.maximum(small_scores, FP32_NEG_INF)
        small_pairs = pl.sort32(small_scores, pl.reinterpret_view(small_indices, pl.UINT32))
        small_pairs = pl.mrgsort(small_pairs, block_len=64)
        small_pairs = pl.mrgsort(small_pairs, block_len=256)
        small_left = pl.tile.slice(small_pairs, [1, TOPK_PAIR_WIDTH], [0, 0])
        small_right = pl.tile.slice(small_pairs, [1, TOPK_PAIR_WIDTH], [0, 1024])
        small_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
        small_merged = pl.tile.mrgsort(small_left, small_right, tmp=small_tmp)
        small_top = pl.tile.slice(small_merged, [1, TOPK_PAIR_WIDTH], [0, 0])
        pl.store(small_top, [output_slot, 0], pair_arena)
    elif valid_count <= 2048:
        # 三轮归并完成2048候选排序，仅前512对写GM；其余排序结果不外传。
        leaf_indices = pl.add(pl.tile.arange(0, [1, 2048], dtype=pl.INT32), logical_begin_i32)
        leaf_scores_raw = pl.load(score_arena, [score_row, 0], [1, 2048], valid_shape=[1, valid_count])
        leaf_scores = pl.tile.fillpad(leaf_scores_raw, pad_value=pl.PadValue.min)
        leaf_scores = pl.maximum(leaf_scores, FP32_NEG_INF)
        pairs = pl.sort32(leaf_scores, pl.reinterpret_view(leaf_indices, pl.UINT32))
        pairs = pl.mrgsort(pairs, block_len=64)
        pairs = pl.mrgsort(pairs, block_len=256)
        pairs = pl.mrgsort(pairs, block_len=1024)
        pl.store(pl.tile.slice(pairs, [1, TOPK_PAIR_WIDTH], [0, 0]), [output_slot, 0], pair_arena)
    elif valid_count <= 2560:
        # 2560恰好五个512候选run：前四组先组成2048，再让最后一组优先于前缀。
        # 不补到4096，既保持原tile容量，也保持同分时后2048段的优先关系。
        partial_indices = pl.add(pl.tile.arange(0, [1, 2560], dtype=pl.INT32), logical_begin_i32)
        partial_raw = pl.load(score_arena, [score_row, 0], [1, 2560], valid_shape=[1, valid_count])
        partial_scores = pl.tile.fillpad(partial_raw, pad_value=pl.PadValue.min)
        partial_scores = pl.maximum(partial_scores, FP32_NEG_INF)
        partial_pairs = pl.sort32(partial_scores, pl.reinterpret_view(partial_indices, pl.UINT32))
        partial_pairs = pl.mrgsort(partial_pairs, block_len=64)
        partial_pairs = pl.mrgsort(partial_pairs, block_len=256)
        partial_prefix = pl.set_validshape(partial_pairs, 1, 4096)
        # 这里4096是2048个(score,index)的元素宽度，不是4096个候选。
        partial_prefix = pl.mrgsort(partial_prefix, block_len=1024)
        partial_left = pl.tile.slice(partial_prefix, [1, TOPK_PAIR_WIDTH], [0, 0])
        partial_right = pl.tile.slice(partial_pairs, [1, TOPK_PAIR_WIDTH], [0, 4096])
        partial_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
        partial_merged = pl.tile.mrgsort(partial_right, partial_left, tmp=partial_tmp)
        pl.store(pl.tile.slice(partial_merged, [1, TOPK_PAIR_WIDTH], [0, 0]), [output_slot, 0], pair_arena)
    elif valid_count <= 3072:
        # 六个512候选run：前四组归并，两份右根按原顺序一起置于前缀之前。
        triple_indices = pl.add(pl.tile.arange(0, [1, 3072], dtype=pl.INT32), logical_begin_i32)
        triple_raw = pl.load(score_arena, [score_row, 0], [1, 3072], valid_shape=[1, valid_count])
        triple_scores = pl.tile.fillpad(triple_raw, pad_value=pl.PadValue.min)
        triple_scores = pl.maximum(triple_scores, FP32_NEG_INF)
        triple_pairs = pl.sort32(triple_scores, pl.reinterpret_view(triple_indices, pl.UINT32))
        triple_pairs = pl.mrgsort(triple_pairs, block_len=64)
        triple_pairs = pl.mrgsort(triple_pairs, block_len=256)
        triple_prefix = pl.set_validshape(triple_pairs, 1, 4096)
        triple_prefix = pl.mrgsort(triple_prefix, block_len=1024)
        triple_left = pl.tile.slice(triple_prefix, [1, TOPK_PAIR_WIDTH], [0, 0])
        triple_right0 = pl.tile.slice(triple_pairs, [1, TOPK_PAIR_WIDTH], [0, 4096])
        triple_right1 = pl.tile.slice(triple_pairs, [1, TOPK_PAIR_WIDTH], [0, 5120])
        triple_tmp = pl.tile.create([1, 3 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
        # right0→right1→left是原三路tie优先序；不要为了按地址排序改成逆序。
        triple_merged = pl.tile.mrgsort(triple_right0, triple_right1, triple_left, tmp=triple_tmp)
        pl.store(pl.tile.slice(triple_merged, [1, TOPK_PAIR_WIDTH], [0, 0]), [output_slot, 0], pair_arena)
    elif valid_count <= 4096:
        # 八个512 run先各合成2048半段，再合成Top512；完整scratch允许容纳
        # 两份1024元素根，但GM只保存截取后的一个根。
        medium_indices = pl.add(pl.tile.arange(0, [1, 4096], dtype=pl.INT32), logical_begin_i32)
        medium_scores_raw = pl.load(score_arena, [score_row, 0], [1, 4096], valid_shape=[1, valid_count])
        medium_scores = pl.tile.fillpad(medium_scores_raw, pad_value=pl.PadValue.min)
        medium_scores = pl.maximum(medium_scores, FP32_NEG_INF)
        medium_pairs = pl.sort32(medium_scores, pl.reinterpret_view(medium_indices, pl.UINT32))
        medium_pairs = pl.mrgsort(medium_pairs, block_len=64)
        medium_pairs = pl.mrgsort(medium_pairs, block_len=256)
        medium_pairs = pl.mrgsort(medium_pairs, block_len=1024)
        medium_left = pl.tile.slice(medium_pairs, [1, TOPK_PAIR_WIDTH], [0, 0])
        medium_right = pl.tile.slice(medium_pairs, [1, TOPK_PAIR_WIDTH], [0, 4096])
        medium_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
        # 新的后2048段在前，因此同分时medium_right优先于medium_left。
        medium_merged = pl.tile.mrgsort(medium_right, medium_left, tmp=medium_tmp)
        pl.store(pl.tile.slice(medium_merged, [1, TOPK_PAIR_WIDTH], [0, 0]), [output_slot, 0], pair_arena)


@pl.jit.inline(auto_scope=False)
def indexer_topk_half_leaf_resident(
    half_scores: pl.Tile[[1, 4096], pl.FP32],
    pair_arena: pl.InOut[pl.Tensor[[PAIR_ROWS_DYN, TOPK_PAIR_WIDTH], pl.FP32]],
    logical_begin: pl.Scalar[pl.INDEX],
    valid_count: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
) -> None:
    """UB内原半叶：六条长度分支、物理排序宽度及同分顺序均复现GM版。"""
    logical_begin_i32 = pl.cast(logical_begin, pl.INT32)
    if valid_count <= 512:
        # 即使只有一项，也按512候选的固定tile排序；尾列由valid_shape和fillpad
        # 写成最小score。index先按全局位置生成，再只按位解释为UINT32给sort32。
        short_indices = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), logical_begin_i32)
        short_raw = pl.tile.extract(
            half_scores, 0, 0, [1, 512], target_memory=pl.MemorySpace.Vec
        )
        short_raw = pl.set_validshape(short_raw, 1, valid_count)
        short_scores = pl.tile.fillpad(short_raw, pad_value=pl.PadValue.min)
        short_scores = pl.maximum(short_scores, FP32_NEG_INF)
        short_pairs = pl.sort32(short_scores, pl.reinterpret_view(short_indices, pl.UINT32))
        # 32候选局部run→128候选run→512候选根；长度64/256是pair元素数。
        short_pairs = pl.mrgsort(short_pairs, block_len=64)
        short_pairs = pl.mrgsort(short_pairs, block_len=256)
        pl.store(short_pairs, [output_slot, 0], pair_arena)
    elif valid_count <= 1024:
        # 两个512根分开取出后按原left/right顺序合并；不能套用其它分支的
        # “后段优先”，因为1024分支的原输入顺序与2560/4096分支不同。
        small_indices = pl.add(pl.tile.arange(0, [1, 1024], dtype=pl.INT32), logical_begin_i32)
        small_raw = pl.tile.extract(
            half_scores, 0, 0, [1, 1024], target_memory=pl.MemorySpace.Vec
        )
        small_raw = pl.set_validshape(small_raw, 1, valid_count)
        small_scores = pl.tile.fillpad(small_raw, pad_value=pl.PadValue.min)
        small_scores = pl.maximum(small_scores, FP32_NEG_INF)
        small_pairs = pl.sort32(small_scores, pl.reinterpret_view(small_indices, pl.UINT32))
        small_pairs = pl.mrgsort(small_pairs, block_len=64)
        small_pairs = pl.mrgsort(small_pairs, block_len=256)
        small_left = pl.tile.slice(small_pairs, [1, TOPK_PAIR_WIDTH], [0, 0])
        small_right = pl.tile.slice(small_pairs, [1, TOPK_PAIR_WIDTH], [0, 1024])
        small_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
        small_merged = pl.tile.mrgsort(small_left, small_right, tmp=small_tmp)
        small_top = pl.tile.slice(small_merged, [1, TOPK_PAIR_WIDTH], [0, 0])
        pl.store(small_top, [output_slot, 0], pair_arena)
    elif valid_count <= 2048:
        # 三轮归并完成2048候选排序，仅前512对写GM；其余排序结果不外传。
        leaf_indices = pl.add(pl.tile.arange(0, [1, 2048], dtype=pl.INT32), logical_begin_i32)
        leaf_scores_raw = pl.tile.extract(
            half_scores, 0, 0, [1, 2048], target_memory=pl.MemorySpace.Vec
        )
        leaf_scores_raw = pl.set_validshape(leaf_scores_raw, 1, valid_count)
        leaf_scores = pl.tile.fillpad(leaf_scores_raw, pad_value=pl.PadValue.min)
        leaf_scores = pl.maximum(leaf_scores, FP32_NEG_INF)
        pairs = pl.sort32(leaf_scores, pl.reinterpret_view(leaf_indices, pl.UINT32))
        pairs = pl.mrgsort(pairs, block_len=64)
        pairs = pl.mrgsort(pairs, block_len=256)
        pairs = pl.mrgsort(pairs, block_len=1024)
        pl.store(pl.tile.slice(pairs, [1, TOPK_PAIR_WIDTH], [0, 0]), [output_slot, 0], pair_arena)
    elif valid_count <= 2560:
        # 2560恰好五个512候选run：前四组先组成2048，再让最后一组优先于前缀。
        # 不补到4096，既保持原tile容量，也保持同分时后2048段的优先关系。
        partial_indices = pl.add(pl.tile.arange(0, [1, 2560], dtype=pl.INT32), logical_begin_i32)
        partial_raw = pl.tile.extract(
            half_scores, 0, 0, [1, 2560], target_memory=pl.MemorySpace.Vec
        )
        partial_raw = pl.set_validshape(partial_raw, 1, valid_count)
        partial_scores = pl.tile.fillpad(partial_raw, pad_value=pl.PadValue.min)
        partial_scores = pl.maximum(partial_scores, FP32_NEG_INF)
        partial_pairs = pl.sort32(partial_scores, pl.reinterpret_view(partial_indices, pl.UINT32))
        partial_pairs = pl.mrgsort(partial_pairs, block_len=64)
        partial_pairs = pl.mrgsort(partial_pairs, block_len=256)
        partial_prefix = pl.set_validshape(partial_pairs, 1, 4096)
        # 这里4096是2048个(score,index)的元素宽度，不是4096个候选。
        partial_prefix = pl.mrgsort(partial_prefix, block_len=1024)
        partial_left = pl.tile.slice(partial_prefix, [1, TOPK_PAIR_WIDTH], [0, 0])
        partial_right = pl.tile.slice(partial_pairs, [1, TOPK_PAIR_WIDTH], [0, 4096])
        partial_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
        partial_merged = pl.tile.mrgsort(partial_right, partial_left, tmp=partial_tmp)
        pl.store(pl.tile.slice(partial_merged, [1, TOPK_PAIR_WIDTH], [0, 0]), [output_slot, 0], pair_arena)
    elif valid_count <= 3072:
        # 六个512候选run：前四组归并，两份右根按原顺序一起置于前缀之前。
        triple_indices = pl.add(pl.tile.arange(0, [1, 3072], dtype=pl.INT32), logical_begin_i32)
        triple_raw = pl.tile.extract(
            half_scores, 0, 0, [1, 3072], target_memory=pl.MemorySpace.Vec
        )
        triple_raw = pl.set_validshape(triple_raw, 1, valid_count)
        triple_scores = pl.tile.fillpad(triple_raw, pad_value=pl.PadValue.min)
        triple_scores = pl.maximum(triple_scores, FP32_NEG_INF)
        triple_pairs = pl.sort32(triple_scores, pl.reinterpret_view(triple_indices, pl.UINT32))
        triple_pairs = pl.mrgsort(triple_pairs, block_len=64)
        triple_pairs = pl.mrgsort(triple_pairs, block_len=256)
        triple_prefix = pl.set_validshape(triple_pairs, 1, 4096)
        triple_prefix = pl.mrgsort(triple_prefix, block_len=1024)
        triple_left = pl.tile.slice(triple_prefix, [1, TOPK_PAIR_WIDTH], [0, 0])
        triple_right0 = pl.tile.slice(triple_pairs, [1, TOPK_PAIR_WIDTH], [0, 4096])
        triple_right1 = pl.tile.slice(triple_pairs, [1, TOPK_PAIR_WIDTH], [0, 5120])
        triple_tmp = pl.tile.create([1, 3 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
        # right0→right1→left是原三路tie优先序；不要为了按地址排序改成逆序。
        triple_merged = pl.tile.mrgsort(triple_right0, triple_right1, triple_left, tmp=triple_tmp)
        pl.store(pl.tile.slice(triple_merged, [1, TOPK_PAIR_WIDTH], [0, 0]), [output_slot, 0], pair_arena)
    elif valid_count <= 4096:
        # 八个512 run先各合成2048半段，再合成Top512；完整scratch允许容纳
        # 两份1024元素根，但GM只保存截取后的一个根。
        medium_indices = pl.add(pl.tile.arange(0, [1, 4096], dtype=pl.INT32), logical_begin_i32)
        medium_scores_raw = pl.tile.extract(
            half_scores, 0, 0, [1, 4096], target_memory=pl.MemorySpace.Vec
        )
        medium_scores_raw = pl.set_validshape(medium_scores_raw, 1, valid_count)
        medium_scores = pl.tile.fillpad(medium_scores_raw, pad_value=pl.PadValue.min)
        medium_scores = pl.maximum(medium_scores, FP32_NEG_INF)
        medium_pairs = pl.sort32(medium_scores, pl.reinterpret_view(medium_indices, pl.UINT32))
        medium_pairs = pl.mrgsort(medium_pairs, block_len=64)
        medium_pairs = pl.mrgsort(medium_pairs, block_len=256)
        medium_pairs = pl.mrgsort(medium_pairs, block_len=1024)
        medium_left = pl.tile.slice(medium_pairs, [1, TOPK_PAIR_WIDTH], [0, 0])
        medium_right = pl.tile.slice(medium_pairs, [1, TOPK_PAIR_WIDTH], [0, 4096])
        medium_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
        # 新的后2048段在前，因此同分时medium_right优先于medium_left。
        medium_merged = pl.tile.mrgsort(medium_right, medium_left, tmp=medium_tmp)
        pl.store(pl.tile.slice(medium_merged, [1, TOPK_PAIR_WIDTH], [0, 0]), [output_slot, 0], pair_arena)


@pl.jit.inline(auto_scope=False)
def indexer_topk_segment_pairs(
    score_arena: pl.Tensor[[SCORE_ROWS_DYN, TOPK_CANDIDATES_PER_LEAF], pl.FP32],
    score_row: pl.Scalar[pl.INDEX],
    logical_begin: pl.Scalar[pl.INDEX],
    valid_count: pl.Scalar[pl.INDEX],
    score_offset: pl.constexpr,
) -> pl.Tile:
    """返回连续1..2048候选的UB根；score_offset仅控制GM起列，不改变全局index。"""
    logical_i32 = pl.cast(logical_begin, pl.INT32)
    if valid_count <= 512:
        # 从score_offset起读实际前缀，尾列补最小值；全局index仍从logical_begin起。
        tiny_indices = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), logical_i32)
        tiny_raw = pl.load(score_arena, [score_row, score_offset], [1, 512], valid_shape=[1, valid_count])
        tiny_scores = pl.maximum(pl.tile.fillpad(tiny_raw, pad_value=pl.PadValue.min), FP32_NEG_INF)
        tiny_pairs = pl.sort32(tiny_scores, pl.reinterpret_view(tiny_indices, pl.UINT32))
        tiny_pairs = pl.mrgsort(tiny_pairs, block_len=64)
        tiny_pairs = pl.mrgsort(tiny_pairs, block_len=256)
        root = pl.tile.extract(tiny_pairs, 0, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec)
    elif valid_count <= 1024:
        # 保留1024分支left/right顺序，不能复用“新段在前”的跨段merge顺序。
        small_indices = pl.add(pl.tile.arange(0, [1, 1024], dtype=pl.INT32), logical_i32)
        small_raw = pl.load(score_arena, [score_row, score_offset], [1, 1024], valid_shape=[1, valid_count])
        small_scores = pl.maximum(pl.tile.fillpad(small_raw, pad_value=pl.PadValue.min), FP32_NEG_INF)
        small_pairs = pl.sort32(small_scores, pl.reinterpret_view(small_indices, pl.UINT32))
        small_pairs = pl.mrgsort(small_pairs, block_len=64)
        small_pairs = pl.mrgsort(small_pairs, block_len=256)
        small_left = pl.tile.slice(small_pairs, [1, TOPK_PAIR_WIDTH], [0, 0])
        small_right = pl.tile.slice(small_pairs, [1, TOPK_PAIR_WIDTH], [0, 1024])
        small_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
        small_merged = pl.tile.mrgsort(small_left, small_right, tmp=small_tmp)
        root = pl.tile.extract(small_merged, 0, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec)
    else:
        # 调用合同限定到2048；这一else不能被误当成任意长度通用排序。
        large_indices = pl.add(pl.tile.arange(0, [1, 2048], dtype=pl.INT32), logical_i32)
        large_raw = pl.load(score_arena, [score_row, score_offset], [1, 2048], valid_shape=[1, valid_count])
        large_scores = pl.maximum(pl.tile.fillpad(large_raw, pad_value=pl.PadValue.min), FP32_NEG_INF)
        large_pairs = pl.sort32(large_scores, pl.reinterpret_view(large_indices, pl.UINT32))
        large_pairs = pl.mrgsort(large_pairs, block_len=64)
        large_pairs = pl.mrgsort(large_pairs, block_len=256)
        large_pairs = pl.mrgsort(large_pairs, block_len=1024)
        root = pl.tile.extract(large_pairs, 0, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec)
    return root
