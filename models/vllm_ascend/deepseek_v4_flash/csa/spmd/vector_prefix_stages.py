# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""动态CSA前段的三个AIV阶段链：按数据就绪重排，不改变每阶段数学。

整体思路与阅读顺序
    indexer_compressor_vector_stage执行Indexer pool→共享prefix读完→state
    commit→norm；main_compressor_vector_stage对主压缩执行相同的生命周期，
    再写compact cache；original_kv_vector_stage完成原KV norm/RoPE及cache。
    数学仍由各原*_stages.py执行，这里只表达真实owner、等待、发布和次序。

动态输入及职责
    所有T/B/页数/行容量来自原runtime_abi的pl.dynamic，Q/L/位置和页表仍为
    device Tensor。main与Indexer state独立；只有完成全部相关请求的pool
    读取后，真实state写者才可覆盖共享prefix。请求数量不得从T/S推断。
    tokens/worker/core/planner由同一次SPMD入口给出，generation原样透传。
    返回的Tensor是原GM别名，不新增输出搬运或第二个incore task。

协同排程
    COOPERATIVE_PREFIX开启时，Indexer压缩在query RoPE前准备；主压缩的
    AIV消费者在AIC主Q_B期间处理已前移的主投影；原KV后处理放到本核Score
    完成之后。无实际请求/行块的核不等待无关输入，但仍按阶段合同发布空完成。
    原KV的32行norm与8行cache写入owner不同；后者必须等实际norm生产者。
    全部发布仍调用原fence/cacheinvalid/notify，不以Scalar缓存一致性为前提。
    开关关闭保留原等待集合，供完整同轮消融；当前默认选择和完整实测边界
    见policy.py。保持空角色发布合同，不能用旧结果代替其它动态场景验证。
"""

import pypto.language as pl

from .cache_stages import original_kv_writeback_stage
from .compressor_output_stages import compressor_norm_rope_cache_stage
from .compressor_pool_stages import (
    compressor_pool_stage,
    compressor_state_commit_stage,
    indexer_compressor_pool_stage,
    indexer_compressor_state_commit_stage,
)
from .indexer_compressor_output_stages import indexer_compressor_norm_rope_stage
from .kv_stages import kv_norm_rope_stage
from .phase_sync import publish_phase, wait_producer, wait_row_owners
from .runtime_abi import (
    CACHE_ROWS,
    CORES,
    GROUPS,
    IC,
    ICPAD,
    IPOOL,
    IPROJ,
    IQ,
    ISTATE_ELEMS,
    ISTATE_PAGES,
    ITABLE_COLS,
    KV_ROWS,
    MC,
    MCACHE_ROWS,
    MNORM,
    MPOOL,
    MPROJ,
    MQ,
    MSLOTS,
    MSTATE_ELEMS,
    MSTATE_PAGES,
    MTABLE_COLS,
    PHASE_IDX_COMMIT,
    PHASE_IDX_NORM,
    PHASE_IDX_POOL,
    PHASE_IDX_PROJECT,
    PHASE_KV_NORM,
    PHASE_KV_PROJECT,
    PHASE_MAIN_COMMIT,
    PHASE_MAIN_OUTPUT,
    PHASE_MAIN_POOL,
    PHASE_MAIN_PROJECT,
    PHASE_META,
    PHASE_ORIGINAL_CACHE,
    PHASE_ROPE,
    VECTORS,
    WORDS,
    B,
    T,
)


@pl.jit.inline(auto_scope=False)
def indexer_compressor_vector_stage(
    positions: pl.Tensor[[T], pl.INT64],
    idx_value: pl.Tensor[[IPROJ, 256], pl.FP32],
    idx_score: pl.Tensor[[IPROJ, 256], pl.FP32],
    idx_state: pl.InOut[pl.Tensor[[ISTATE_PAGES, ISTATE_ELEMS], pl.FP32]],
    idx_state_table: pl.Tensor[[B, ITABLE_COLS], pl.INT32],
    idx_ape: pl.Tensor[[4, 256], pl.FP32],
    idx_bounds: pl.Tensor[[IQ], pl.INT32],
    idx_lengths: pl.Tensor[[B], pl.INT32],
    idx_state_slots: pl.Tensor[[T, 2], pl.INT32],
    idx_pooled: pl.InOut[pl.Tensor[[IPOOL, 128], pl.FP32]],
    idx_gamma: pl.Tensor[[128], pl.BF16],
    idx_cos: pl.Tensor[[IC, 64], pl.FP32],
    idx_sin: pl.Tensor[[IC, 64], pl.FP32],
    idx_offsets: pl.InOut[pl.Tensor[[B], pl.INT32]],
    idx_normed: pl.InOut[pl.Tensor[[ICPAD, 128], pl.BF16]],
    signal: pl.InOut[pl.Tensor[[WORDS], pl.INT32]],
    tokens: pl.Scalar[pl.INDEX],
    worker: pl.Scalar[pl.INDEX],
    core: pl.Scalar[pl.INDEX],
    generation: pl.Scalar[pl.INT32],
    planner: pl.Scalar[pl.INDEX],
    COOPERATIVE_PREFIX: pl.constexpr = False,
):
    # Indexer压缩只依赖自身投影/Native state；不必排在主压缩输出之后。
    # 先取得compact offsets，再做pool→读完集合→commit→norm，和AIC上的
    # 主压缩投影重叠。仍需所有pool读完后才能写共享prefix state。
    signal = wait_producer(signal, PHASE_META, GROUPS + planner, CORES, generation)
    # Indexer压缩与主压缩独立state/metadata；其M16/N32分工也不能照搬主分支。
    idx_requests = pl.tensor.dim(idx_lengths, 0)
    idx_tasks = ((tokens + 15) // 16) * 8
    if not COOPERATIVE_PREFIX or worker < pl.min(idx_requests, VECTORS):
        signal = wait_row_owners(signal, PHASE_IDX_PROJECT, CORES, generation, 0, idx_tasks, 1, GROUPS, 0, 1)
    idx_pooled = indexer_compressor_pool_stage(
        idx_value,
        idx_score,
        idx_state,
        idx_state_table,
        idx_ape,
        positions,
        idx_bounds,
        idx_lengths,
        idx_pooled,
        worker,
        VECTORS,
    )
    signal = publish_phase(signal, PHASE_IDX_POOL, core, CORES, generation)
    if not COOPERATIVE_PREFIX or worker < pl.min(idx_requests, VECTORS):
        signal = wait_row_owners(signal, PHASE_IDX_POOL, CORES, generation, 0, idx_requests, 1, VECTORS, GROUPS, 1)
    idx_state = indexer_compressor_state_commit_stage(
        idx_value,
        idx_score,
        idx_ape,
        positions,
        idx_bounds,
        idx_lengths,
        idx_state_slots,
        idx_state,
        worker,
        VECTORS,
    )
    signal = publish_phase(signal, PHASE_IDX_COMMIT, core, CORES, generation)
    idx_normed = indexer_compressor_norm_rope_stage(
        idx_pooled,
        idx_gamma,
        idx_cos,
        idx_sin,
        positions,
        idx_bounds,
        idx_lengths,
        idx_offsets,
        idx_normed,
        worker,
        VECTORS,
    )
    signal = publish_phase(signal, PHASE_IDX_NORM, core, CORES, generation)
    return idx_pooled, idx_state, idx_normed, signal

@pl.jit.inline(auto_scope=False)
def main_compressor_vector_stage(
    main_value: pl.Tensor[[MPROJ, 1024], pl.FP32],
    main_score: pl.Tensor[[MPROJ, 1024], pl.FP32],
    main_state: pl.InOut[pl.Tensor[[MSTATE_PAGES, MSTATE_ELEMS], pl.FP32]],
    main_state_table: pl.Tensor[[B, MTABLE_COLS], pl.INT32],
    main_ape: pl.Tensor[[4, 1024], pl.FP32],
    positions: pl.Tensor[[T], pl.INT64],
    main_bounds: pl.Tensor[[MQ], pl.INT32],
    main_lengths: pl.Tensor[[B], pl.INT32],
    main_state_slots: pl.Tensor[[T, 2], pl.INT32],
    main_pooled: pl.InOut[pl.Tensor[[MPOOL, 512], pl.FP32]],
    main_gamma: pl.Tensor[[512], pl.BF16],
    main_cos: pl.Tensor[[MC, 64], pl.FP32],
    main_sin: pl.Tensor[[MC, 64], pl.FP32],
    main_offsets: pl.InOut[pl.Tensor[[B], pl.INT32]],
    main_slots: pl.Tensor[[MSLOTS, 2], pl.INT32],
    main_normed: pl.InOut[pl.Tensor[[MNORM, 512], pl.FP32]],
    main_debug: pl.InOut[pl.Tensor[[T, 512], pl.FP32]],
    main_cache: pl.InOut[pl.Tensor[[MCACHE_ROWS, 512], pl.BF16]],
    signal: pl.InOut[pl.Tensor[[WORDS], pl.INT32]],
    tokens: pl.Scalar[pl.INDEX],
    worker: pl.Scalar[pl.INDEX],
    core: pl.Scalar[pl.INDEX],
    generation: pl.Scalar[pl.INT32],
    COOPERATIVE_PREFIX: pl.constexpr = False,
):
    # 主压缩投影M64/N64；请求级pool会读共享prefix历史，必须全部pool读完才能
    # 提交任何state。不能因为本核pool结束就覆盖其他核仍可能读取的Native页。
    main_requests = pl.tensor.dim(main_lengths, 0)
    main_tasks = ((tokens + 63) // 64) * 16
    if not COOPERATIVE_PREFIX or worker < pl.min(main_requests, VECTORS):
        signal = wait_row_owners(signal, PHASE_MAIN_PROJECT, CORES, generation, 0, main_tasks, 1, GROUPS, 0, 1)
    main_pooled = compressor_pool_stage(
        main_value,
        main_score,
        main_state,
        main_state_table,
        main_ape,
        positions,
        main_bounds,
        main_lengths,
        main_pooled,
        worker,
        VECTORS,
    )
    signal = publish_phase(signal, PHASE_MAIN_POOL, core, CORES, generation)
    if not COOPERATIVE_PREFIX or worker < pl.min(main_requests, VECTORS):
        signal = wait_row_owners(signal, PHASE_MAIN_POOL, CORES, generation, 0, main_requests, 1, VECTORS, GROUPS, 1)
    main_state = compressor_state_commit_stage(
        main_value,
        main_score,
        main_ape,
        positions,
        main_bounds,
        main_lengths,
        main_state_slots,
        main_state,
        worker,
        VECTORS,
    )
    signal = publish_phase(signal, PHASE_MAIN_COMMIT, core, CORES, generation)
    main_normed, main_debug, main_cache = compressor_norm_rope_cache_stage(
        main_pooled,
        main_gamma,
        main_cos,
        main_sin,
        main_bounds,
        main_lengths,
        positions,
        main_offsets,
        main_slots,
        main_normed,
        main_debug,
        main_cache,
        worker,
        VECTORS,
    )
    signal = publish_phase(signal, PHASE_MAIN_OUTPUT, core, CORES, generation)

    return main_pooled, main_state, main_normed, main_debug, main_cache, signal

@pl.jit.inline(auto_scope=False)
def original_kv_vector_stage(
    rope_cos: pl.Tensor[[T, 64], pl.FP32],
    rope_signed: pl.InOut[pl.Tensor[[T, 64], pl.FP32]],
    rope_swap: pl.InOut[pl.Tensor[[T, 64], pl.INT32]],
    gamma_ckv: pl.Tensor[[512], pl.BF16],
    kv_fp32: pl.Tensor[[KV_ROWS, 512], pl.FP32],
    kv: pl.InOut[pl.Tensor[[T, 512], pl.BF16]],
    native_slots: pl.Tensor[[T, 2], pl.INT32],
    cache_rows: pl.InOut[pl.Tensor[[CACHE_ROWS, 512], pl.BF16]],
    signal: pl.InOut[pl.Tensor[[WORDS], pl.INT32]],
    tokens: pl.Scalar[pl.INDEX],
    worker: pl.Scalar[pl.INDEX],
    core: pl.Scalar[pl.INDEX],
    generation: pl.Scalar[pl.INT32],
    COOPERATIVE_PREFIX: pl.constexpr = False,
):
    if COOPERATIVE_PREFIX:
        # KV投影逻辑worker已循环置换；实际有归一化工作的核才取得全部AIC。
        # 此时已完成本核Score，不能让独立KV链阻断Score的两槽ACK消费者。
        if worker < pl.min((tokens + 31) // 32, VECTORS):
            signal = wait_row_owners(signal, PHASE_KV_PROJECT, CORES, generation, 0, GROUPS, 1, GROUPS, 0, 1)
            signal = wait_row_owners(signal, PHASE_ROPE, CORES, generation, 0, tokens, 8, VECTORS, GROUPS, 1)
            kv = kv_norm_rope_stage(kv_fp32, gamma_ckv, rope_cos, rope_signed, rope_swap, kv, worker, VECTORS)
        signal = publish_phase(signal, PHASE_KV_NORM, core, CORES, generation)
        # cache每8行一个owner；其余AIV可直接继续Top-K，不替无业务角色等KV。
        if worker < pl.min((tokens + 7) // 8, VECTORS):
            signal = wait_row_owners(signal, PHASE_KV_NORM, CORES, generation, 0, tokens, 32, VECTORS, GROUPS, 1)
            cache_rows = original_kv_writeback_stage(kv, native_slots, cache_rows, worker, VECTORS)
        signal = publish_phase(signal, PHASE_ORIGINAL_CACHE, core, CORES, generation)
    else:
        # 每个最多512行的KV分段有4个N128列块与最多3个M组；第一段的owner
        # 集合覆盖后续各段（完整512行段为12个owner，短尾段只是它的子集）。
        first_segment = pl.min(tokens, 512)
        if first_segment < 128:
            kv_tasks = 4 * pl.min(3, pl.max(1, first_segment // 32))
            signal = wait_row_owners(signal, PHASE_KV_PROJECT, CORES, generation, 0, kv_tasks, 1, GROUPS, 0, 1)
        else:
            kv_tasks = 4 * pl.min(3, pl.max(1, first_segment // 64))
            signal = wait_row_owners(signal, PHASE_KV_PROJECT, CORES, generation, 0, kv_tasks, 1, GROUPS, 0, 1)
        signal = wait_row_owners(
            signal,
            PHASE_ROPE,
            CORES,
            generation,
            0,
            tokens,
            8,
            VECTORS,
            GROUPS,
            1,
        )
        kv = kv_norm_rope_stage(kv_fp32, gamma_ckv, rope_cos, rope_signed, rope_swap, kv, worker, VECTORS)
        signal = publish_phase(signal, PHASE_KV_NORM, core, CORES, generation)
        # KVnorm每32行一个owner，写cache每8行一个owner，不能误用同一分块公式。
        signal = wait_row_owners(
            signal,
            PHASE_KV_NORM,
            CORES,
            generation,
            0,
            tokens,
            32,
            VECTORS,
            GROUPS,
            1,
        )
        cache_rows = original_kv_writeback_stage(kv, native_slots, cache_rows, worker, VECTORS)
        signal = publish_phase(signal, PHASE_ORIGINAL_CACHE, core, CORES, generation)

    return kv, cache_rows, signal
