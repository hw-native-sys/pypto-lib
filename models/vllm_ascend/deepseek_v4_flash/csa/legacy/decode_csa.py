# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TP1 attention-only CSA adapted from the pinned pypto-lib reference."""

import pypto.language as pl

from .compact_metadata import build_compact_row_offsets
from ..common.config import (
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    DECODE_BATCH,
    DECODE_SEQ,
    KV_ORI_BLOCK_NUM,
)
from ..common.config import (
    FLASH as M,
)
from ..common.config import (
    TP as TP_SIZE,
)
from .decode_compressor_ratio4 import compressor_ratio4
from .decode_indexer import LONG_S6_MIN_QUERY_ROWS, TOPK_CANDIDATES_PER_LEAF, indexer, indexer_use_long_s6
from .decode_indexer_compressor import indexer_compressor
from .decode_o_proj import LOCAL_T, LOCAL_T_PAD, decode_o_proj_tp1
from .decode_sparse_attn_csa import T_PAD, sparse_attn_csa_tp1
from .hc_pre import (
    HC_DIM,
    HC_DIM_INV,
    HC_MULT,
    HC_PAD,
    LINEAR_T_TILE,
    MIX_HC,
    NORM_EPS,
    RMS_K_TILE,
    hc_mix_norm,
    hc_pre_gates_from_rms,
)
from .indexer_scale_layout import SCALE_LAYOUT_CAPACITY, prove_scale_layout
from ..common.layout import (
    COMPRESSED_ROWS_DYN,
    COMPRESSED_TABLE_COLUMNS_DYN,
    INDEXER_PAGE_BYTES_DYN,
    INDEXER_ROWS_DYN,
    INDEXER_TABLE_COLUMNS_DYN,
    INNER_STATE_PAGE_ELEMENTS_DYN,
    INNER_STATE_TABLE_COLUMNS_DYN,
    ORIGINAL_TABLE_COLUMNS_DYN,
    QUERY_BOUNDS_DYN,
    STATE_PAGE_ELEMENTS_DYN,
    STATE_TABLE_COLUMNS_DYN,
)
from ..common.nz_mode import BF16_WEIGHT_LAYOUT, QUANT_WEIGHT_LAYOUT, WO_A_WEIGHT_LAYOUT
from .qkv_proj_rope import QR_M_TILE, qkv_proj_rope_policy

# Dynamic shape variables.
B_DYN = pl.dynamic("B_DYN")  # per-request axis
T_DYN = pl.dynamic("T_DYN")  # T = B * S
ORI_BLOCK_NUM_DYN = pl.dynamic("ORI_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")
IDX_NATIVE_CACHE_BLOCK_NUM_DYN = pl.dynamic("IDX_NATIVE_CACHE_BLOCK_NUM_DYN")
MAIN_STATE_BLOCK_NUM_DYN = pl.dynamic("CSA_STATE_BLOCK_NUM_DYN")
INNER_STATE_BLOCK_NUM_DYN = pl.dynamic("INNER_STATE_BLOCK_NUM_DYN")

# model config
B = DECODE_BATCH // TP_SIZE
S = DECODE_SEQ
T = B * S
EPS = M.rms_norm_eps
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
WIN = M.sliding_window
MAX_SEQ_LEN = M.max_position_embeddings
IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
IDX_TOPK = M.index_topk
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = H * HEAD_DIM // O_GROUPS

# kernel constants
COMPRESS_RATIO = 4
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
MAIN_OUT_DIM = COFF * HEAD_DIM
MAIN_STATE_DIM = 2 * MAIN_OUT_DIM
MAIN_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
MAIN_STATE_LEN = COFF * COMPRESS_RATIO
INNER_OUT_DIM = COFF * IDX_HEAD_DIM
INNER_STATE_DIM = 2 * INNER_OUT_DIM
INNER_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
INNER_STATE_LEN = COFF * COMPRESS_RATIO
ORI_MAX_BLOCKS = (MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
ORI_BLOCK_NUM = KV_ORI_BLOCK_NUM
CMP_MAX_ROWS = MAX_SEQ_LEN // COMPRESS_RATIO
CMP_MAX_BLOCKS = (CMP_MAX_ROWS + BLOCK_SIZE - 1) // BLOCK_SIZE
IDX_MAX_BLOCKS = CMP_MAX_BLOCKS

# Indexer dependency slots across conditional scopes.
IDX_QUERY_DEP_SLOT = 0
IDX_HADAMARD_DEP_SLOT = 1
IDX_QUANT_DEP_SLOT = 2
IDX_LEAF_DEP_SLOT = 3
IDX_DEP_COUNT = IDX_LEAF_DEP_SLOT + 1

# tiling
CSA_PROJECTION_PACK_ROW_TILE = 8
CSA_PROJECTION_PACK_WORKERS = 16
CSA_ALL_VISIBLE_WORKERS = 16
CSA_WB_TOKEN_TILE = 8
HC_WIDEN_T_TILE = 8    # hc 残差流 BF16->FP32 的行块
HC_WIDEN_D_TILE = RMS_K_TILE  # 与原RMS的512列归约分段一致。
HC_WIDEN_WORKERS = 48   # 同上，AIV 通道数
CSA_ROPE_SIGN_T_TILE = 4  # RoPE 符号行块，沿用上游 csa_rope_interleave 的 4 行
CSA_ROPE_WORKERS = 16
CSA_WB_WORKERS = 48  # CSA cache-write workers
TP1_CSA_WB_WORKERS = 8  # TP1 CSA cache-write workers

if T != LOCAL_T:
    raise ValueError(f"CSA token capacity {T} must equal TP local token capacity {LOCAL_T}")
if T_PAD != LOCAL_T_PAD:
    raise ValueError(f"CSA token capacity {T_PAD} must equal TP local token capacity {LOCAL_T_PAD}")


@pl.jit.inline(auto_scope=False)
def _read_device_max_seq_len(
    lengths: pl.Tensor[[B_DYN], pl.INT32],
    unused_host_value: pl.Scalar[pl.INT32],
):
    maximum = 0
    for batch in pl.range(pl.tensor.dim(lengths, 0)):
        maximum = pl.max(maximum, pl.read(lengths, [batch]))
    return maximum


@pl.jit.inline(auto_scope=False)
def _use_host_max_seq_len(
    lengths: pl.Tensor[[B_DYN], pl.INT32],
    maximum: pl.Scalar[pl.INT32],
):
    return maximum


def _make_decode_csa_entry(*, host_metadata: bool):
    # A Python factory selects the dependency before IR parsing. An IR-level
    # constant branch would still expose the forbidden Host tensor.read to HBG.
    # Legacy folds the unused default out of its ABI; HBG accepts a runtime int.
    resolve_max_seq_len = _use_host_max_seq_len if host_metadata else _read_device_max_seq_len

    def _decode_csa_tp1_layer(
        x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
        hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
        hc_attn_scale: pl.Tensor[[3], pl.FP32],
        hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
        attn_norm_w: pl.Tensor[[D], pl.BF16],
        wq_a: pl.Tensor[[Q_LORA, D], pl.BF16, BF16_WEIGHT_LAYOUT],
        wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, QUANT_WEIGHT_LAYOUT],
        wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
        wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
        gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
        gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
        freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
        freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
        cmp_freqs_cos: pl.Tensor[[COMPRESSED_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
        cmp_freqs_sin: pl.Tensor[[COMPRESSED_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
        inner_freqs_cos: pl.Tensor[[INDEXER_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
        inner_freqs_sin: pl.Tensor[[INDEXER_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
        cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
        cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
        cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
        cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
        compress_state: pl.InOut[pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, STATE_PAGE_ELEMENTS_DYN], pl.FP32]],
        state_block_table: pl.Tensor[[B_DYN, STATE_TABLE_COLUMNS_DYN], pl.INT32],
        idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
        idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
        weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
        hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
        inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
        inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
        inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
        inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
        inner_compress_state: pl.InOut[pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_PAGE_ELEMENTS_DYN], pl.FP32]],
        inner_state_block_table: pl.Tensor[[B_DYN, INNER_STATE_TABLE_COLUMNS_DYN], pl.INT32],
        kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
        cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
        cmp_block_table: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32],
        idx_native_kv_cache: pl.InOut[pl.Tensor[[IDX_NATIVE_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8]],
        idx_block_table: pl.Tensor[[B_DYN, INDEXER_TABLE_COLUMNS_DYN], pl.INT32],
        ori_slot_mapping: pl.Tensor[[T_DYN, 2], pl.INT32],
        ori_block_table: pl.Tensor[[B_DYN, ORIGINAL_TABLE_COLUMNS_DYN], pl.INT32],
        cmp_slot_mapping: pl.Tensor[[COMPRESSED_ROWS_DYN, 2], pl.INT32],
        idx_slot_mapping: pl.Tensor[[INDEXER_ROWS_DYN, 2], pl.INT32],
        state_slot_mapping: pl.Tensor[[T_DYN, 2], pl.INT32],
        inner_state_slot_mapping: pl.Tensor[[T_DYN, 2], pl.INT32],
        position_ids: pl.Tensor[[T_DYN], pl.INT64],
        kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
        cmp_query_start_loc: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
        cmp_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
        idx_query_start_loc: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
        attn_sink: pl.Tensor[[H], pl.FP32],
        wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
        # 保持 Native 加载后的 K×N 矩阵，分组仅体现为核内 K 偏移。
        wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.INT8, QUANT_WEIGHT_LAYOUT],
        wo_b_scale: pl.Tensor[[D], pl.FP32],
        idx_topk_scores: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32]],
        idx_topk: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32]],
        x_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]],
        host_max_seq_len: pl.Scalar[pl.INT32] = 0,
        SINGLE_M16_QA: pl.constexpr = False,
    ):
        """TP1 整层：mHC 的 pre 门 + input_layernorm + attention + mHC 的 post。

        上游 pypto-lib 的 `_decode_csa_tp1` 本来就是这个形状，我们当初移植时只取了
        中间的 attention 段，把 `hc_pre` / `input_layernorm` / `hc_post` 留在 kernel
        外面，于是它们成了每层 94.9us 的串行头尾（16 卡 aclgraph 实测 57.6 + 16.7 +
        20.6us）。搬进来有两重收益：sinkhorn 的 20 次迭代能整个留在 UB 里做，不必每步
        回 GM；而 `comb` 门与 `hc_post` 里 `comb x residual` 那一项都不依赖 attention
        的输出，可以落进 kernel 开头那段 AIV 空窗（实测 25~100us 有 47~48 核空闲）。

        hc 残差流按 Native 的 BF16 收发：43 层里只有 21 层走 PTO，`x_hc` / `x_out`
        要与 Native 的 `npu_hc_pre_v2` / `npu_hc_post` 在层间互传。
        """
        max_seq_len = resolve_max_seq_len(kv_seq_lens, host_max_seq_len)
        x_hc.bind_dynamic(0, T_DYN)
        x_out.bind_dynamic(0, T_DYN)
        freqs_cos.bind_dynamic(0, T_DYN)
        freqs_sin.bind_dynamic(0, T_DYN)
        cmp_freqs_cos.bind_dynamic(0, COMPRESSED_ROWS_DYN)
        cmp_freqs_sin.bind_dynamic(0, COMPRESSED_ROWS_DYN)
        inner_freqs_cos.bind_dynamic(0, INDEXER_ROWS_DYN)
        inner_freqs_sin.bind_dynamic(0, INDEXER_ROWS_DYN)
        compress_state.bind_dynamic(0, MAIN_STATE_BLOCK_NUM_DYN)
        compress_state.bind_dynamic(1, STATE_PAGE_ELEMENTS_DYN)
        state_block_table.bind_dynamic(0, B_DYN)
        state_block_table.bind_dynamic(1, STATE_TABLE_COLUMNS_DYN)
        inner_compress_state.bind_dynamic(0, INNER_STATE_BLOCK_NUM_DYN)
        inner_compress_state.bind_dynamic(1, INNER_STATE_PAGE_ELEMENTS_DYN)
        inner_state_block_table.bind_dynamic(0, B_DYN)
        inner_state_block_table.bind_dynamic(1, INNER_STATE_TABLE_COLUMNS_DYN)
        kv_cache.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
        cmp_kv.bind_dynamic(0, CMP_BLOCK_NUM_DYN)
        idx_native_kv_cache.bind_dynamic(0, IDX_NATIVE_CACHE_BLOCK_NUM_DYN)
        idx_native_kv_cache.bind_dynamic(1, INDEXER_PAGE_BYTES_DYN)
        ori_slot_mapping.bind_dynamic(0, T_DYN)
        ori_block_table.bind_dynamic(0, B_DYN)
        ori_block_table.bind_dynamic(1, ORIGINAL_TABLE_COLUMNS_DYN)
        cmp_slot_mapping.bind_dynamic(0, COMPRESSED_ROWS_DYN)
        idx_slot_mapping.bind_dynamic(0, INDEXER_ROWS_DYN)
        state_slot_mapping.bind_dynamic(0, T_DYN)
        inner_state_slot_mapping.bind_dynamic(0, T_DYN)
        position_ids.bind_dynamic(0, T_DYN)
        kv_seq_lens.bind_dynamic(0, B_DYN)
        cmp_query_start_loc.bind_dynamic(0, QUERY_BOUNDS_DYN)
        cmp_seq_lens.bind_dynamic(0, B_DYN)
        idx_query_start_loc.bind_dynamic(0, QUERY_BOUNDS_DYN)
        cmp_block_table.bind_dynamic(0, B_DYN)
        cmp_block_table.bind_dynamic(1, COMPRESSED_TABLE_COLUMNS_DYN)
        idx_block_table.bind_dynamic(0, B_DYN)
        idx_block_table.bind_dynamic(1, INDEXER_TABLE_COLUMNS_DYN)
        # attn_out 现在是 kernel 内部张量，不再需要绑定外部动态维。
        idx_topk.bind_dynamic(0, T_DYN)
        idx_topk_scores.bind_dynamic(0, T_DYN)
        t_dim = pl.tensor.dim(x_hc, 0)
        post_t = pl.create_tensor([t_dim, HC_MULT], dtype=pl.FP32)
        comb_t = pl.create_tensor([t_dim, HC_MULT * HC_MULT], dtype=pl.FP32)
        x_normed_t = pl.create_tensor([t_dim, D], dtype=pl.BF16)
        # row_recip=False：走 row_expand_div 的精确路径，与上游 CSA 口径一致
        # （HCA 那条才用 recip 换速度）。
        # 保留已验证的 HC scope 边界以管理任务和临时张量的生命周期。
        # scope 退出释放引用，并不是设备端等待所有任务完成的 barrier；
        # 消费者的执行顺序由张量依赖和显式 TaskId 依赖保证。
        # 加宽后顺手按原512列次序计算RMS，消除独立RMS任务对FP32中间缓冲的再读取。
        # Cube及mix仍消费同一FP32缓冲；不把AIV cast下沉到纯Cube任务。
        x_hc32 = pl.create_tensor([t_dim, HC_MULT, D], dtype=pl.FP32)
        x_hc_flat = pl.reshape(x_hc, [t_dim, HC_MULT * D])
        x_hc32_flat = pl.reshape(x_hc32, [t_dim, HC_MULT * D])
        widen_rows = (t_dim + HC_WIDEN_T_TILE - 1) // HC_WIDEN_T_TILE
        hc_padded_rows = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
        inv_rms = pl.create_tensor([hc_padded_rows, 1], dtype=pl.FP32)
        widen_tail = pl.create_tensor([HC_WIDEN_T_TILE, HC_MULT * D], dtype=pl.FP32)
        # 这是 hc_pre 前段串行链的第一个任务，原本缺 allow_early_resolve，
        # 后继任务要等它整组完成才获得派发资格。理由与下面 csa_row_offsets 的注释相同。
        with pl.spmd(pl.min(widen_rows, HC_WIDEN_WORKERS), name_hint="hc_widen_rms",
                     allow_early_resolve=True) as _widen_tid:
            for widen_blk in pl.range(pl.tile.get_block_idx(), widen_rows,
                                      pl.min(widen_rows, HC_WIDEN_WORKERS)):
                w_t0 = widen_blk * HC_WIDEN_T_TILE
                w_rows = pl.min(HC_WIDEN_T_TILE, t_dim - w_t0)
                w_sq_sum = pl.full([1, HC_WIDEN_T_TILE], dtype=pl.FP32, value=0.0)
                for w_db in pl.pipeline(HC_MULT * D // HC_WIDEN_D_TILE, stage=4):
                    w_d0 = w_db * HC_WIDEN_D_TILE
                    w_src = pl.slice(x_hc_flat, [HC_WIDEN_T_TILE, HC_WIDEN_D_TILE], [w_t0, w_d0],
                                     valid_shape=[w_rows, HC_WIDEN_D_TILE])
                    w_val = pl.cast(w_src, pl.FP32)
                    if w_rows == HC_WIDEN_T_TILE:
                        x_hc32_flat[w_t0:w_t0 + HC_WIDEN_T_TILE, w_d0:w_d0 + HC_WIDEN_D_TILE] = w_val
                        w_sq = pl.mul(w_val, w_val)
                        w_sq_row = pl.reshape(pl.row_sum(w_sq), [1, HC_WIDEN_T_TILE])
                        w_sq_sum = pl.add(w_sq_sum, w_sq_row)
                    else:
                        # cast可能丢失valid_shape，显式恢复并清零无效行后才参与归约。
                        w_valid = pl.set_validshape(w_val, w_rows, HC_WIDEN_D_TILE)
                        w_clean = pl.fillpad(w_valid, pad_value=pl.PadValue.zero)
                        w_sq_tail = pl.mul(w_clean, w_clean)
                        w_sq_row_tail = pl.reshape(pl.row_sum(w_sq_tail), [1, HC_WIDEN_T_TILE])
                        w_sq_sum = pl.add(w_sq_sum, w_sq_row_tail)
                        widen_tail[0:HC_WIDEN_T_TILE, w_d0:w_d0 + HC_WIDEN_D_TILE] = w_val
                        w_out = pl.load(
                            widen_tail, [0, w_d0], [HC_WIDEN_T_TILE, HC_WIDEN_D_TILE],
                            valid_shape=[w_rows, HC_WIDEN_D_TILE], target_memory=pl.MemorySpace.Vec,
                        )
                        pl.store(w_out, [w_t0, w_d0], x_hc32_flat)
                w_mean = pl.add(pl.mul(w_sq_sum, HC_DIM_INV), NORM_EPS)
                w_inv = pl.reshape(pl.rsqrt(w_mean, high_precision=True), [HC_WIDEN_T_TILE, 1])
                inv_rms[w_t0:w_t0 + HC_WIDEN_T_TILE, 0:1] = w_inv

        # 27711336前段证明的多task适配：尺寸在编排侧解析，证明只读页表/长度。
        # 不让Tensor.dim在证明incore中引入对cache payload写回的伪依赖。
        early_scale_pages = pl.tensor.dim(idx_native_kv_cache, 0)
        early_scale_layout = pl.create_tensor([SCALE_LAYOUT_CAPACITY], dtype=pl.INT32)
        early_scale_deps = pl.array.create(1, pl.TASK_ID)
        early_scale_deps[0] = _widen_tid
        if max_seq_len // COMPRESS_RATIO > TOPK_CANDIDATES_PER_LEAF:
            early_s6_group = indexer_use_long_s6(t_dim)
            if early_s6_group == 1:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="indexer_scale_layout",
                           allow_early_resolve=True) as early_scale_tid:
                    early_scale_layout = prove_scale_layout(
                        kv_seq_lens, idx_block_table, early_scale_layout, t_dim, early_scale_pages,
                    )
                early_scale_deps[0] = early_scale_tid

        with pl.scope():
            pre_val_store = pl.create_tensor([hc_padded_rows, HC_PAD], dtype=pl.FP32)
            hc_pre_gates_from_rms(
                x_hc32, hc_attn_fn, hc_attn_scale, hc_attn_base,
                pre_val_store, post_t, comb_t, False, inv_rms,
            )
            hc_mix_norm(x_hc32, pre_val_store, attn_norm_w, x_normed_t)
        wb_blocks = (t_dim + CSA_WB_TOKEN_TILE - 1) // CSA_WB_TOKEN_TILE

        idx_sin_signed = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
        cmp_row_offsets = pl.create_tensor([pl.tensor.dim(cmp_seq_lens, 0)], dtype=pl.INT32)
        idx_row_offsets = pl.create_tensor([pl.tensor.dim(kv_seq_lens, 0)], dtype=pl.INT32)
        # 逐请求 compact 偏移是跨请求前缀和，只能串行；RoPE 符号是逐块独立的，可并行。
        # 原先两件事共用一个 CORE_GROUP 任务，整段被前缀和拖成串行——泳道实测
        # csa_rope_sign count=1、Exec 13.42us 却独占一个串行窗口。拆成两个任务，
        # 符号那段走 SPMD，靠 deps 保证偏移先算好。
        # 元数据也是RMS/cache消费者的实际前置，必须允许其参与预派发资格判定。
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_row_offsets", allow_early_resolve=True) as offsets_tid:
            cmp_row_offsets = build_compact_row_offsets(cmp_query_start_loc, cmp_seq_lens, cmp_row_offsets)
            idx_row_offsets = build_compact_row_offsets(idx_query_start_loc, kv_seq_lens, idx_row_offsets)

        # 向上取整分块并保留 valid_shape：t_dim = batch*6 不保证是 4 的倍数，
        # 上游 csa_rope_interleave 用的 t_dim // 4 会丢掉尾行（batch=1 时 t_dim=6 只覆盖 0~3）。
        rope_sign_blocks = (t_dim + CSA_ROPE_SIGN_T_TILE - 1) // CSA_ROPE_SIGN_T_TILE
        with pl.spmd(pl.min(rope_sign_blocks, CSA_ROPE_WORKERS), name_hint="csa_rope_sign",
                     deps=[offsets_tid], allow_early_resolve=True) as rope_tid:
            for rope_rb in pl.range(pl.tile.get_block_idx(), rope_sign_blocks,
                                    pl.min(rope_sign_blocks, CSA_ROPE_WORKERS)):
                rope_t0 = rope_rb * CSA_ROPE_SIGN_T_TILE
                rope_rows = pl.min(CSA_ROPE_SIGN_T_TILE, t_dim - rope_t0)
                # 符号表按块重算：SPMD 下每个 worker 要有自己的一份。
                il_ones = pl.tile.full([CSA_ROPE_SIGN_T_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
                il_lane_ids = pl.cast(pl.tile.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32)
                il_col = pl.col_expand_mul(il_ones, il_lane_ids)
                il_dup_f = pl.cast(
                    pl.cast(pl.mul(il_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32
                )
                il_lane = pl.sub(il_col, pl.mul(il_dup_f, 2.0))
                il_sign = pl.sub(pl.mul(il_lane, 2.0), 1.0)
                rope_sin_rows = pl.load(freqs_sin, [rope_t0, 0], [CSA_ROPE_SIGN_T_TILE, ROPE_HEAD_DIM],
                                        valid_shape=[rope_rows, ROPE_HEAD_DIM])
                rope_sign_rows = pl.set_validshape(il_sign, rope_rows, ROPE_HEAD_DIM)
                pl.store(pl.mul(rope_sin_rows, rope_sign_rows), [rope_t0, 0], idx_sin_signed)

        q = pl.create_tensor([t_dim, H, HEAD_DIM], dtype=pl.BF16)
        kv = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.BF16)
        qr = pl.create_tensor([t_dim, Q_LORA], dtype=pl.INT8)
        qr_scale = pl.create_tensor([t_dim, 1], dtype=pl.FP32)
        position_ids_t1 = pl.reshape(position_ids, [t_dim, 1])
        with pl.scope():
            # Projection-chain dependency marker.
            late_dep = pl.system.task_dummy(deps=[rope_tid])
            # 来源66882634的独立输出块分工；长历史单M16用N64，原两M16区间保留N128。
            qa_m16_parallel = pl.cast(0, pl.INT32)
            if SINGLE_M16_QA:
                if max_seq_len // COMPRESS_RATIO > TOPK_CANDIDATES_PER_LEAF:
                    if t_dim <= QR_M_TILE:
                        qa_m16_parallel = pl.cast(2, pl.INT32)
                    elif t_dim >= LONG_S6_MIN_QUERY_ROWS:
                        if t_dim < 2 * QR_M_TILE:
                            qa_m16_parallel = pl.cast(1, pl.INT32)
            else:
                if max_seq_len // COMPRESS_RATIO > TOPK_CANDIDATES_PER_LEAF:
                    if t_dim >= LONG_S6_MIN_QUERY_ROWS:
                        if t_dim < 2 * QR_M_TILE:
                            qa_m16_parallel = pl.cast(1, pl.INT32)
            q, qa_tid = qkv_proj_rope_policy(
                x_normed_t,
                wq_a,
                wq_b,
                wq_b_scale,
                wkv,
                freqs_cos,
                freqs_sin,
                gamma_cq,
                gamma_ckv,
                q,
                kv,
                qr,
                qr_scale,
                late_dep,
                qa_m16_parallel,
                SINGLE_M16_QA=SINGLE_M16_QA,
            )

            ori_block_num = pl.tensor.dim(kv_cache, 0)
            kv_cache_flat = pl.reshape(kv_cache, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
            with pl.spmd(TP1_CSA_WB_WORKERS, name_hint="csa_cache_writeback", allow_early_resolve=True):
                wb_worker = pl.tile.get_block_idx()
                for wb_blk in pl.range(wb_worker, wb_blocks, TP1_CSA_WB_WORKERS):
                    wb_t0 = wb_blk * CSA_WB_TOKEN_TILE
                    for write_dt in pl.range(pl.min(CSA_WB_TOKEN_TILE, t_dim - wb_t0)):
                        write_t = wb_t0 + write_dt
                        write_page = pl.read(ori_slot_mapping, [write_t, 0])
                        write_offset = pl.read(ori_slot_mapping, [write_t, 1])
                        if write_page >= 0 and write_offset >= 0:
                            write_row = pl.cast(write_page, pl.INDEX) * BLOCK_SIZE + write_offset
                            kv_cache_flat[write_row : write_row + 1, 0:HEAD_DIM] = kv[write_t : write_t + 1, 0:HEAD_DIM]

            # Keep Q_A ahead of the heavier Attention Compressor projection.
            # The lighter Indexer projection overlaps Q_A; its Hadamard gate stays below.
            compressor_dep = pl.system.task_dummy(deps=[late_dep, qa_tid])
            cmp_out = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.FP32)
            cmp_out, cmp_cache_write_tid, cmp_kv_score_tid = compressor_ratio4(
                x_normed_t,
                cmp_out,
                compress_state,
                state_block_table,
                cmp_wkv,
                cmp_wgate,
                cmp_ape,
                cmp_norm_w,
                cmp_freqs_cos,
                cmp_freqs_sin,
                cmp_row_offsets,
                cmp_kv,
                position_ids,
                cmp_seq_lens,
                cmp_slot_mapping,
                state_slot_mapping,
                compressor_dep,
            )
            idx_kv_unused = pl.create_tensor([t_dim, IDX_HEAD_DIM], dtype=pl.FP32)
            idx_cache_write_tid = indexer_compressor(
                x_normed_t,
                idx_kv_unused,
                inner_compress_state,
                inner_state_block_table,
                inner_wkv,
                inner_wgate,
                inner_ape,
                inner_norm_w,
                inner_freqs_cos,
                inner_freqs_sin,
                idx_row_offsets,
                hadamard_idx,
                idx_native_kv_cache,
                position_ids,
                kv_seq_lens,
                idx_slot_mapping,
                inner_state_slot_mapping,
                late_dep,
                cmp_kv_score_tid,
            )
            # Bound indexer scratch to its own runtime scope.
            with pl.scope():
                _index_scores, index_indices = indexer(
                    x_normed_t,
                    qr,
                    qr_scale,
                    idx_wq_b,
                    idx_wq_b_scale,
                    weights_proj,
                    freqs_cos,
                    idx_sin_signed,
                    hadamard_idx,
                    idx_native_kv_cache,
                    idx_block_table,
                    idx_topk_scores,
                    idx_topk,
                    position_ids,
                    kv_seq_lens,
                    late_dep,
                    idx_cache_write_tid,
                    max_seq_len,
                    early_scale_layout,
                    early_scale_deps[0],
                )

            # sparse_attn_csa folds the compressed-slot masking + valid-block flags in from the
            # raw indexer topk + position.
            o_packed_heads = pl.create_tensor([O_GROUPS * T_PAD, O_GROUP_IN], dtype=pl.BF16)
            o_packed_heads, heads_dep = sparse_attn_csa_tp1(
                q,
                kv_cache,
                ori_block_table,
                cmp_kv,
                cmp_block_table,
                index_indices,
                position_ids_t1,
                kv_seq_lens,
                attn_sink,
                freqs_cos,
                freqs_sin,
                o_packed_heads,
                max_seq_len,
            )
            with pl.scope():
                layer_output = decode_o_proj_tp1(
                    o_packed_heads, wo_a, wo_b, wo_b_scale, x_hc, post_t, comb_t, x_out, heads_dep,
                )
        return layer_output
    if not host_metadata:
        # The specializer strips constexpr parameters before parsing the body.
        # Keep a literal Scalar annotation in source for the HBG parser.
        _decode_csa_tp1_layer.__annotations__["host_max_seq_len"] = pl.constexpr
    else:
        # The new public ABI must never silently assume an empty history.
        _decode_csa_tp1_layer.__defaults__ = (False,)
    return _decode_csa_tp1_layer


_decode_csa_tp1_layer = _make_decode_csa_entry(host_metadata=False)
_decode_csa_tp1_layer_hbg = _make_decode_csa_entry(host_metadata=True)
decode_csa_tp1_layer = pl.jit.inline(auto_scope=False)(_decode_csa_tp1_layer)
decode_csa_tp1_layer_test = pl.jit(auto_scope=False)(_decode_csa_tp1_layer)
decode_csa_tp1_layer_hbg = pl.jit(auto_scope=False)(_decode_csa_tp1_layer_hbg)


def csa_compile_constants(tokens: int) -> dict[str, bool]:
    """只看Tensor逻辑行数，不读取NPU内容；长度/页表/padding仍由设备处理。"""
    return {"SINGLE_M16_QA": 0 < tokens <= QR_M_TILE}
