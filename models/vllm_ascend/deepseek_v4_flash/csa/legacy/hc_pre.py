# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""从上游 pypto-lib `models/deepseek_v4_flash_dspark/hc_pre.py` 移植。

上游的 `_decode_csa_tp1` 本来就是整层入口（x_hc -> hc_pre_norm -> attention
-> hc_post），我们当初移植 CSA 时把 hc 那一段砍掉了，只保留了 attention。
这里把它搬回来，好让 mHC 的 pre 门、20 次 sinkhorn 迭代以及 attention 的
input_layernorm 都落进 PTO kernel。

**x 保持上游的 FP32，不改 BF16**。vllm-ascend 的 hc 残差流是 BF16
（Native 的 npu_hc_pre_v2 / npu_hc_post，实测见
tests/pypto_test/results/release_offline_pd_20260923/native_hc_v1），
转换放在 kernel 入口一次性做（见 decode_csa 的 hc_widen 任务），不要
下沉到本文件每处 tile 读取。试过后者，两个后果都踩了：

  * `pl.cast` 会丢掉 `pl.slice(..., valid_shape=...)` 携带的有效行标记，
    padding 区的陈旧字节被当成有效行参与平方和与加权和——实测 T=60 时
    mix_x 直接产出 64 个非有限值，T=12 / T=24 的 x_normed 偏差从 1 个 ULP
    涨到 0.77，整模型 256 个序列里 42 个输出退化成全 0。
  * cast 与 fillpad 都是 AIV 操作，会把 hc_pre_linear 这个纯 Cube 任务
    编成 Cube+Vector 的 mix kernel，泳道里从上游的单个 `hc_pre_linear`（24 块、
    24.2us）裂成 `hc_pre_linear_aic`(24) + `hc_pre_linear_aiv`(48)、32.9us。

本项目的档位只保证是 DSpark 的 6 的倍数（6/12/18/24/36/42/48/60/66/72/84/90/96），
非 8、非 16 对齐是常态，不像上游恒为整档 T=384，所以 mix_x_rms_norm 的分块
要按向上取整并兜住尾块——这一处仍是本地修改，见该函数内的注释。

golden / build_tensor_specs / __main__ 属于上游 standalone 用例，不搬。
"""

import pypto.language as pl

from ..common.config import DECODE_BATCH, DECODE_SEQ, TP  # noqa: F401 - preserved upstream imports
from ..common.config import FLASH as M
from ..common.rmsnorm import rms_norm_apply, rms_norm_inverse

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # T = B * S
HC_PAD_ROWS_DYN = pl.dynamic("HC_PAD_ROWS_DYN")

# model config
D = M.hidden_size
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
NORM_EPS = M.rms_norm_eps

# tiling
MIX_PAD = 32  # mix_hc (24) padded to a 32-wide cube N / vector row
HC_PAD = 8  # hc (4) padded for 32B-aligned vector ops
T_TILE = 8  # other values miscompare
LINEAR_T_TILE = 16  # cube matmul rows must be a 16-row boxed tile
COMB_T_TILE = 8
PRE_POST_WORKERS = 16
RMS_K_TILE = 512
LINEAR_K_TILE = 256
D_TILE = 256
D_SPMD = 4096
LINEAR_OK = 4
LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK
LINEAR_WORKERS = 24  # hc_pre_linear workers, one per AIC

# pre0..pre3 / row0..row3 are hand-unrolled over the hc lanes.
assert HC_MULT == 4, f"hc_pre is specialized to HC_MULT == 4, got {HC_MULT}"


@pl.jit.inline
def hc_pre_gates_from_rms(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    pre_val_store: pl.Tensor[[HC_PAD_ROWS_DYN, HC_PAD], pl.FP32],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    row_recip: pl.Scalar[pl.BOOL],
    inv_rms: pl.Tensor[[HC_PAD_ROWS_DYN, 1], pl.FP32],
):
    """共享门控与Sinkhorn算术，复用调用方按原规则生成的RMS统计。"""
    t_dim = pl.tensor.dim(x, 0)
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE  # pad t_dim up to whole 16-row cube tiles
    x_flat = pl.reshape(x, [t_dim, HC_DIM])
    hc_base_2d = pl.reshape(hc_base, [1, MIX_HC])  # for per-group comb base loads in comb_sinkhorn

    # linear: split-K matmul -> per-split partials. The t_dim..t_linear pad rows are
    # zero-filled by valid_shape, never materialized.
    mixes_partials = pl.create_tensor([LINEAR_OK * t_linear, MIX_PAD], dtype=pl.FP32)
    linear_units = (t_linear // LINEAR_T_TILE) * LINEAR_OK
    linear_workers = pl.min(linear_units, LINEAR_WORKERS)
    for linear_worker in pl.spmd(linear_workers, name_hint="hc_pre_linear", allow_early_resolve=True):
        # HC权重沿用Native的ND视图及默认缓存策略；缓存策略的调整需单独实测。
        for task in pl.range(linear_worker, linear_units, linear_workers):
            t0 = (task // LINEAR_OK) * LINEAR_T_TILE
            linear_split = task % LINEAR_OK
            k_base = linear_split * LINEAR_K_PER_SPLIT
            t_rows = pl.min(LINEAR_T_TILE, t_dim - t0)  # last row-block spills past t_dim; valid_shape zero-fills the tail
            acc = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
            for kb in pl.pipeline(0, LINEAR_K_PER_SPLIT // LINEAR_K_TILE, stage=2):
                k0 = k_base + kb * LINEAR_K_TILE
                x_linear_chunk = pl.slice(x_flat, [LINEAR_T_TILE, LINEAR_K_TILE], [t0, k0], valid_shape=[t_rows, LINEAR_K_TILE])
                w_chunk = pl.slice(hc_fn, [MIX_PAD, LINEAR_K_TILE], [0, k0], valid_shape=[MIX_HC, LINEAR_K_TILE])
                acc = pl.matmul_acc(acc, x_linear_chunk, w_chunk, b_trans=True, init_cond=(kb == 0))
            partial_row0 = linear_split * t_linear + t0
            mixes_partials[partial_row0 : partial_row0 + LINEAR_T_TILE, 0:MIX_PAD] = acc

    # Partials are reduced in ascending K order.
    mixes_raw = pl.create_tensor([t_linear, MIX_PAD], dtype=pl.FP32)
    for linear_block in pl.spmd(t_linear // LINEAR_T_TILE, name_hint="hc_pre_linear_reduce", allow_early_resolve=True):
        linear_t0 = linear_block * LINEAR_T_TILE
        mixes_total = mixes_partials[linear_t0 : linear_t0 + LINEAR_T_TILE, 0:MIX_PAD]
        for linear_split in pl.range(1, LINEAR_OK):
            partial_t0 = linear_split * t_linear + linear_t0
            partial_tile = mixes_partials[partial_t0 : partial_t0 + LINEAR_T_TILE, 0:MIX_PAD]
            mixes_total = pl.add(mixes_total, partial_tile)
        mixes_raw[linear_t0 : linear_t0 + LINEAR_T_TILE, 0:MIX_PAD] = mixes_total

    # split_pre_post: inv_rms-scaled pre gate -> pre_val_store (for mix_x), post gate -> post.
    # Both compute at HC_PAD width; post narrows to HC_MULT via a valid-shape slice (an 8-wide
    # 32B tile, 4 cols valid -- a bare 4-wide slice allocs a 16B tile ptoas rejects). comb gate
    # lives in comb_sinkhorn.
    # Only the final partial token tile uses these fixed-size staging buffers.
    post_tail_store = pl.create_tensor([T_TILE, HC_PAD], dtype=pl.FP32)
    for ob_worker in pl.spmd(pl.min(token_tiles, PRE_POST_WORKERS), name_hint="split_pre_post", allow_early_resolve=True):
        scale0 = pl.read(hc_scale, [0])
        scale1 = pl.read(hc_scale, [1])
        for ob in pl.range(ob_worker, token_tiles, pl.min(token_tiles, PRE_POST_WORKERS)):
            t0 = ob * T_TILE
            valid_rows = pl.min(T_TILE, t_dim - t0)
            inv_col = inv_rms[t0:t0 + T_TILE, 0:1]

            pre_base = pl.reshape(hc_base[0:HC_PAD], [1, HC_PAD])
            pre_scaled = pl.mul(pl.row_expand_mul(mixes_raw[t0:t0 + T_TILE, 0:HC_PAD], inv_col), scale0)
            pre_logits = pl.add(pre_scaled, pl.col_expand(pre_scaled, pre_base))
            pre_sig = pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0))
            pre_val = pl.add(pre_sig, HC_EPS)
            pre_val_store[t0:t0 + T_TILE, 0:HC_PAD] = pre_val

            post_base = pl.reshape(hc_base[HC_MULT:HC_MULT + HC_PAD], [1, HC_PAD])
            post_scaled = pl.mul(pl.row_expand_mul(mixes_raw[t0:t0 + T_TILE, HC_MULT:HC_MULT + HC_PAD], inv_col), scale1)
            post_logits = pl.add(post_scaled, pl.col_expand(post_scaled, post_base))
            post_sig = pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0))
            post_pad = pl.mul(post_sig, 2.0)
            if valid_rows == T_TILE:
                post[t0:t0 + T_TILE, 0:HC_MULT] = pl.slice(post_pad, [T_TILE, HC_PAD], [0, 0], valid_shape=[T_TILE, HC_MULT])
            else:
                post_tail_store[0:T_TILE, 0:HC_PAD] = post_pad
                post_tile = pl.load(post_tail_store, [0, 0], [T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
                pl.store(post_tile, [t0, 0], post)

    # comb_sinkhorn: comb gate from mixes_raw cols 8/12/16/20, softmax, then a
    # column-first 20-iteration Sinkhorn -> comb.
    comb_tail_store = pl.create_tensor([COMB_T_TILE, HC_PAD * HC_MULT], dtype=pl.FP32)
    for ob in pl.spmd(token_tiles, name_hint="comb_sinkhorn", allow_early_resolve=True):
        scale2 = pl.read(hc_scale, [2])
        t0 = ob * COMB_T_TILE
        valid_rows = pl.min(COMB_T_TILE, t_dim - t0)
        inv_col_t = pl.load(inv_rms, [t0, 0], [COMB_T_TILE, 1], valid_shape=[valid_rows, 1], target_memory=pl.MemorySpace.Vec)
        comb_off = HC_MULT * 2
        mix_g0 = pl.load(mixes_raw, [t0, comb_off + 0 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
        mix_g1 = pl.load(mixes_raw, [t0, comb_off + 1 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
        mix_g2 = pl.load(mixes_raw, [t0, comb_off + 2 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
        mix_g3 = pl.load(mixes_raw, [t0, comb_off + 3 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
        cb0 = pl.load(hc_base_2d, [0, comb_off + 0 * HC_MULT], [1, HC_PAD], valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
        cb1 = pl.load(hc_base_2d, [0, comb_off + 1 * HC_MULT], [1, HC_PAD], valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
        cb2 = pl.load(hc_base_2d, [0, comb_off + 2 * HC_MULT], [1, HC_PAD], valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
        cb3 = pl.load(hc_base_2d, [0, comb_off + 3 * HC_MULT], [1, HC_PAD], valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
        row0 = pl.add(pl.mul(pl.row_expand_mul(mix_g0, inv_col_t), scale2), pl.col_expand(mix_g0, cb0))
        row1 = pl.add(pl.mul(pl.row_expand_mul(mix_g1, inv_col_t), scale2), pl.col_expand(mix_g1, cb1))
        row2 = pl.add(pl.mul(pl.row_expand_mul(mix_g2, inv_col_t), scale2), pl.col_expand(mix_g2, cb2))
        row3 = pl.add(pl.mul(pl.row_expand_mul(mix_g3, inv_col_t), scale2), pl.col_expand(mix_g3, cb3))
        row0_p = pl.fillpad(row0, pad_value=pl.PadValue.min)
        row1_p = pl.fillpad(row1, pad_value=pl.PadValue.min)
        row2_p = pl.fillpad(row2, pad_value=pl.PadValue.min)
        row3_p = pl.fillpad(row3, pad_value=pl.PadValue.min)

        row_max_tmp = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row_sum_tmp = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row0_max = pl.row_max(row0_p, row_max_tmp)
        row1_max = pl.row_max(row1_p, row_max_tmp)
        row2_max = pl.row_max(row2_p, row_max_tmp)
        row3_max = pl.row_max(row3_p, row_max_tmp)
        row0_exp = pl.exp(pl.row_expand_sub(row0_p, row0_max))
        row1_exp = pl.exp(pl.row_expand_sub(row1_p, row1_max))
        row2_exp = pl.exp(pl.row_expand_sub(row2_p, row2_max))
        row3_exp = pl.exp(pl.row_expand_sub(row3_p, row3_max))
        row0_sum = pl.row_sum(row0_exp, row_sum_tmp)
        row1_sum = pl.row_sum(row1_exp, row_sum_tmp)
        row2_sum = pl.row_sum(row2_exp, row_sum_tmp)
        row3_sum = pl.row_sum(row3_exp, row_sum_tmp)
        if row_recip:
            row0_inv = pl.recip(row0_sum)
            row0_prob = pl.row_expand_mul(row0_exp, row0_inv)
            row0_soft = pl.add(row0_prob, HC_EPS)
            row1_inv = pl.recip(row1_sum)
            row1_prob = pl.row_expand_mul(row1_exp, row1_inv)
            row1_soft = pl.add(row1_prob, HC_EPS)
            row2_inv = pl.recip(row2_sum)
            row2_prob = pl.row_expand_mul(row2_exp, row2_inv)
            row2_soft = pl.add(row2_prob, HC_EPS)
            row3_inv = pl.recip(row3_sum)
            row3_prob = pl.row_expand_mul(row3_exp, row3_inv)
            row3_soft = pl.add(row3_prob, HC_EPS)
        else:
            row0_soft = pl.add(pl.row_expand_div(row0_exp, row0_sum), HC_EPS)
            row1_soft = pl.add(pl.row_expand_div(row1_exp, row1_sum), HC_EPS)
            row2_soft = pl.add(pl.row_expand_div(row2_exp, row2_sum), HC_EPS)
            row3_soft = pl.add(pl.row_expand_div(row3_exp, row3_sum), HC_EPS)

        row0_valid = pl.set_validshape(row0_soft, COMB_T_TILE, HC_MULT)
        row1_valid = pl.set_validshape(row1_soft, COMB_T_TILE, HC_MULT)
        row2_valid = pl.set_validshape(row2_soft, COMB_T_TILE, HC_MULT)
        row3_valid = pl.set_validshape(row3_soft, COMB_T_TILE, HC_MULT)
        row0_eff = pl.fillpad(row0_valid, pad_value=pl.PadValue.zero)
        row1_eff = pl.fillpad(row1_valid, pad_value=pl.PadValue.zero)
        row2_eff = pl.fillpad(row2_valid, pad_value=pl.PadValue.zero)
        row3_eff = pl.fillpad(row3_valid, pad_value=pl.PadValue.zero)

        row_sum_tmp_iter = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        col_sum = pl.add(pl.add(row0_eff, row1_eff), pl.add(row2_eff, row3_eff))
        col_sum = pl.add(col_sum, HC_EPS)
        row0_cur = pl.div(row0_eff, col_sum)
        row1_cur = pl.div(row1_eff, col_sum)
        row2_cur = pl.div(row2_eff, col_sum)
        row3_cur = pl.div(row3_eff, col_sum)

        for _sk_it in pl.pipeline(HC_SINKHORN_ITER - 1, stage=2):
            row0_rowsum = pl.add(pl.row_sum(row0_cur, row_sum_tmp_iter), HC_EPS)
            row1_rowsum = pl.add(pl.row_sum(row1_cur, row_sum_tmp_iter), HC_EPS)
            row2_rowsum = pl.add(pl.row_sum(row2_cur, row_sum_tmp_iter), HC_EPS)
            row3_rowsum = pl.add(pl.row_sum(row3_cur, row_sum_tmp_iter), HC_EPS)
            if row_recip:
                row0_inv = pl.recip(row0_rowsum)
                row0_norm = pl.row_expand_mul(row0_cur, row0_inv)
                row1_inv = pl.recip(row1_rowsum)
                row1_norm = pl.row_expand_mul(row1_cur, row1_inv)
                row2_inv = pl.recip(row2_rowsum)
                row2_norm = pl.row_expand_mul(row2_cur, row2_inv)
                row3_inv = pl.recip(row3_rowsum)
                row3_norm = pl.row_expand_mul(row3_cur, row3_inv)
            else:
                row0_norm = pl.row_expand_div(row0_cur, row0_rowsum)
                row1_norm = pl.row_expand_div(row1_cur, row1_rowsum)
                row2_norm = pl.row_expand_div(row2_cur, row2_rowsum)
                row3_norm = pl.row_expand_div(row3_cur, row3_rowsum)
            col_sum = pl.add(pl.add(row0_norm, row1_norm), pl.add(row2_norm, row3_norm))
            col_sum = pl.add(col_sum, HC_EPS)
            row0_cur = pl.div(row0_norm, col_sum)
            row1_cur = pl.div(row1_norm, col_sum)
            row2_cur = pl.div(row2_norm, col_sum)
            row3_cur = pl.div(row3_norm, col_sum)

        if valid_rows == COMB_T_TILE:
            row0_out = pl.set_validshape(row0_cur, COMB_T_TILE, HC_MULT)
            row1_out = pl.set_validshape(row1_cur, COMB_T_TILE, HC_MULT)
            row2_out = pl.set_validshape(row2_cur, COMB_T_TILE, HC_MULT)
            row3_out = pl.set_validshape(row3_cur, COMB_T_TILE, HC_MULT)
            pl.store(row0_out, [t0, 0 * HC_MULT], comb)
            pl.store(row1_out, [t0, 1 * HC_MULT], comb)
            pl.store(row2_out, [t0, 2 * HC_MULT], comb)
            pl.store(row3_out, [t0, 3 * HC_MULT], comb)
        else:
            pl.store(row0_cur, [0, 0 * HC_PAD], comb_tail_store)
            pl.store(row1_cur, [0, 1 * HC_PAD], comb_tail_store)
            pl.store(row2_cur, [0, 2 * HC_PAD], comb_tail_store)
            pl.store(row3_cur, [0, 3 * HC_PAD], comb_tail_store)
            row0_tail = pl.load(comb_tail_store, [0, 0 * HC_PAD], [COMB_T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
            row1_tail = pl.load(comb_tail_store, [0, 1 * HC_PAD], [COMB_T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
            row2_tail = pl.load(comb_tail_store, [0, 2 * HC_PAD], [COMB_T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
            row3_tail = pl.load(comb_tail_store, [0, 3 * HC_PAD], [COMB_T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
            pl.store(row0_tail, [t0, 0 * HC_MULT], comb)
            pl.store(row1_tail, [t0, 1 * HC_MULT], comb)
            pl.store(row2_tail, [t0, 2 * HC_MULT], comb)
            pl.store(row3_tail, [t0, 3 * HC_MULT], comb)

    return pre_val_store


@pl.jit.inline
def hc_pre_gates(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    pre_val_store: pl.Tensor[[T_DYN, HC_PAD], pl.FP32],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    row_recip: pl.Scalar[pl.BOOL],
):
    """保留原独立RMS任务及其归约次序。"""
    t_dim = pl.tensor.dim(x, 0)
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    x_flat = pl.reshape(x, [t_dim, HC_DIM])
    inv_rms = pl.create_tensor([t_linear, 1], dtype=pl.FP32)

    # rms: full-K sum-of-squares per token-tile -> inv_rms.
    for t in pl.spmd(token_tiles, name_hint="hc_pre_rms", allow_early_resolve=True):
        t0 = t * T_TILE
        valid_rows = pl.min(T_TILE, t_dim - t0)
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HC_DIM // RMS_K_TILE, stage=4):
            k0 = kb * RMS_K_TILE
            if valid_rows == T_TILE:
                x_chunk_full = x_flat[t0:t0 + T_TILE, k0:k0 + RMS_K_TILE]
                x_sq_full = pl.mul(x_chunk_full, x_chunk_full)
                x_sq_row_full = pl.reshape(pl.row_sum(x_sq_full), [1, T_TILE])
                sq_sum = pl.add(sq_sum, x_sq_row_full)
            else:
                x_chunk_tail = pl.slice(x_flat, [T_TILE, RMS_K_TILE], [t0, k0], valid_shape=[valid_rows, RMS_K_TILE])
                x_sq_tail = pl.mul(x_chunk_tail, x_chunk_tail)
                x_sq_row_tail = pl.reshape(pl.row_sum(x_sq_tail), [1, T_TILE])
                sq_sum = pl.add(sq_sum, x_sq_row_tail)
        sq_mean = pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS)
        inv = pl.reshape(pl.rsqrt(sq_mean, high_precision=True), [T_TILE, 1])
        inv_rms[t0:t0 + T_TILE, 0:1] = inv

    return hc_pre_gates_from_rms(
        x, hc_fn, hc_scale, hc_base, pre_val_store, post, comb, row_recip, inv_rms,
    )


def _hc_pre(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    x_mixed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    post: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    comb: pl.Out[pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32]],
):
    """Compute HC gates and BF16 pre-mixed activations."""
    x.bind_dynamic(0, T_DYN)
    x_mixed.bind_dynamic(0, T_DYN)
    post.bind_dynamic(0, T_DYN)
    comb.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(x, 0)
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    pre_val_store = pl.create_tensor([t_linear, HC_PAD], dtype=pl.FP32)
    hc_pre_gates(x, hc_fn, hc_scale, hc_base, pre_val_store, post, comb, False)
    x_flat = pl.reshape(x, [t_dim, HC_DIM])

    # mix_x: x_mixed = sum_h pre[:,h]*x[:,h,:], fanned over D/D_SPMD blocks per token tile.
    x_mixed_tail_store = pl.create_tensor([T_TILE, D], dtype=pl.BF16)
    for blk in pl.spmd(token_tiles * (D // D_SPMD), name_hint="mix_x", allow_early_resolve=True):
        t0 = (blk // (D // D_SPMD)) * T_TILE
        d_base = (blk % (D // D_SPMD)) * D_SPMD
        valid_rows = pl.min(T_TILE, t_dim - t0)
        pre_tile_t = pl.transpose(pre_val_store[t0:t0 + T_TILE, 0:HC_PAD], axis1=0, axis2=1)
        pre0 = pl.reshape(pre_tile_t[0:1, 0:T_TILE], [T_TILE, 1])
        pre1 = pl.reshape(pre_tile_t[1:2, 0:T_TILE], [T_TILE, 1])
        pre2 = pl.reshape(pre_tile_t[2:3, 0:T_TILE], [T_TILE, 1])
        pre3 = pl.reshape(pre_tile_t[3:4, 0:T_TILE], [T_TILE, 1])
        for db in pl.pipeline(D_SPMD // D_TILE, stage=2):
            d0 = d_base + db * D_TILE
            x0 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 0 * D + d0], valid_shape=[valid_rows, D_TILE])
            x1 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 1 * D + d0], valid_shape=[valid_rows, D_TILE])
            x2 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 2 * D + d0], valid_shape=[valid_rows, D_TILE])
            x3 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 3 * D + d0], valid_shape=[valid_rows, D_TILE])
            y0 = pl.row_expand_mul(x0, pre0)
            y1 = pl.row_expand_mul(x1, pre1)
            y2 = pl.row_expand_mul(x2, pre2)
            y3 = pl.row_expand_mul(x3, pre3)
            y_tile = pl.add(pl.add(y0, y1), pl.add(y2, y3))
            y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
            if valid_rows == T_TILE:
                x_mixed[t0:t0 + T_TILE, d0:d0 + D_TILE] = y_bf16
            else:
                x_mixed_tail_store[0:T_TILE, d0:d0 + D_TILE] = y_bf16
                y_out = pl.load(x_mixed_tail_store, [0, d0], [T_TILE, D_TILE], valid_shape=[valid_rows, D_TILE], target_memory=pl.MemorySpace.Vec)
                pl.store(y_out, [t0, d0], x_mixed)
    return x_mixed


hc_pre = pl.jit.inline(_hc_pre)
hc_pre_test = pl.jit(_hc_pre)


@pl.jit.inline
def hc_mix_norm(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_val_store: pl.Tensor[[HC_PAD_ROWS_DYN, HC_PAD], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    x_normed: pl.Tensor[[T_DYN, D], pl.BF16],
):
    """共享HC混合、BF16边界及后续RMSNorm算术。"""
    t_dim = pl.tensor.dim(x, 0)
    x_flat = pl.reshape(x, [t_dim, HC_DIM])
    # 上游这段按整 8 行块走（它的 T 恒为 384）。本包的档位只保证是 DSpark 的 6 的
    # 倍数，t_dim=12/18/30/... 都不是 8 的倍数，向下取整会整块丢掉尾部 token。
    # 这里按向上取整分块：读侧靠 pl.slice 的 valid_shape 补零（x 是 FP32，切片直接
    # 带有效行标记，不像 BF16 那样要先 cast 而丢掉它），写侧尾块走 tail_store。
    # x_mixed 是内部缓冲，直接按补齐行数分配，无效行的统计量只落在它自己那几行上。
    t_pad = ((t_dim + T_TILE - 1) // T_TILE) * T_TILE
    x_mixed = pl.create_tensor([t_pad, D], dtype=pl.BF16)
    x_normed_tail_store = pl.create_tensor([T_TILE, D], dtype=pl.BF16)

    # The RMS statistic includes the intermediate BF16 rounding.
    with pl.spmd(t_pad // T_TILE, name_hint="mix_x_rms_norm", allow_early_resolve=True) as mixed_tid:
        t0 = pl.tile.get_block_idx() * T_TILE
        valid_rows = pl.min(T_TILE, t_dim - t0)
        pre_tile = pre_val_store[t0:t0 + T_TILE, 0:HC_PAD]
        pre_tile_t = pl.transpose(pre_tile, axis1=0, axis2=1)
        pre0 = pl.reshape(pre_tile_t[0:1, 0:T_TILE], [T_TILE, 1])
        pre1 = pl.reshape(pre_tile_t[1:2, 0:T_TILE], [T_TILE, 1])
        pre2 = pl.reshape(pre_tile_t[2:3, 0:T_TILE], [T_TILE, 1])
        pre3 = pl.reshape(pre_tile_t[3:4, 0:T_TILE], [T_TILE, 1])
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for mix_db in pl.pipeline(D // D_TILE, stage=2):
            d0 = mix_db * D_TILE
            x0 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, d0], valid_shape=[valid_rows, D_TILE])
            x1 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, D + d0], valid_shape=[valid_rows, D_TILE])
            x2 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 2 * D + d0], valid_shape=[valid_rows, D_TILE])
            x3 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 3 * D + d0], valid_shape=[valid_rows, D_TILE])
            y0 = pl.row_expand_mul(x0, pre0)
            y1 = pl.row_expand_mul(x1, pre1)
            y2 = pl.row_expand_mul(x2, pre2)
            y3 = pl.row_expand_mul(x3, pre3)
            y01 = pl.add(y0, y1)
            y23 = pl.add(y2, y3)
            y = pl.add(y01, y23)
            y_bf16 = pl.cast(y, pl.BF16, mode="rint")
            x_mixed[t0:t0 + T_TILE, d0:d0 + D_TILE] = y_bf16
            y_rounded = pl.cast(y_bf16, pl.FP32)
            y_sq = pl.mul(y_rounded, y_rounded)
            y_sq_sum = pl.row_sum(y_sq)
            y_sq_row = pl.reshape(y_sq_sum, [1, T_TILE])
            sq_sum = pl.add(sq_sum, y_sq_row)
        norm_sq_sum = pl.create_tensor([1, T_TILE], dtype=pl.FP32)
        norm_sq_sum[:, :] = sq_sum
        y_inv = pl.create_tensor([1, T_TILE], dtype=pl.FP32)
        inverse_result = rms_norm_inverse(norm_sq_sum)
        y_inv[:, :] = inverse_result
        for norm_db in pl.pipeline(D // D_TILE, stage=2):
            d0 = norm_db * D_TILE
            mixed_input = x_mixed[t0:t0 + T_TILE, d0:d0 + D_TILE]
            norm_w_input = norm_w[d0:d0 + D_TILE]
            norm_w_row = pl.reshape(norm_w_input, [1, D_TILE])
            normed_bf16 = rms_norm_apply(mixed_input, norm_w_row, y_inv)
            if valid_rows == T_TILE:
                x_normed[t0:t0 + T_TILE, d0:d0 + D_TILE] = normed_bf16
            else:
                x_normed_tail_store[0:T_TILE, d0:d0 + D_TILE] = normed_bf16
                normed_valid = pl.load(
                    x_normed_tail_store, [0, d0], [T_TILE, D_TILE],
                    valid_shape=[valid_rows, D_TILE], target_memory=pl.MemorySpace.Vec,
                )
                pl.store(normed_valid, [t0, d0], x_normed)
    return mixed_tid


@pl.jit.inline
def hc_pre_norm(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    x_normed: pl.Tensor[[T_DYN, D], pl.BF16],
    row_recip: pl.Scalar[pl.BOOL],
):
    """Normalize pre-mixed activations for complete eight-token decode tiles."""
    t_dim = pl.tensor.dim(x, 0)
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    pre_val_store = pl.create_tensor([t_linear, HC_PAD], dtype=pl.FP32)
    hc_pre_gates(x, hc_fn, hc_scale, hc_base, pre_val_store, post, comb, row_recip)
    return hc_mix_norm(x, pre_val_store, norm_w, x_normed)


def _hc_pre_norm_entry(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    post: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    comb: pl.Out[pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32]],
    x_normed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    """独立入口，只为把 hc_pre_norm（含 mix_x_rms_norm）单独拿出来对数值。

    hc_pre_test 走的是 _hc_pre 的 mix_x 段，覆盖不到这里的归一化收尾。
    """
    x.bind_dynamic(0, T_DYN)
    post.bind_dynamic(0, T_DYN)
    comb.bind_dynamic(0, T_DYN)
    x_normed.bind_dynamic(0, T_DYN)
    hc_pre_norm(x, hc_fn, hc_scale, hc_base, norm_w, post, comb, x_normed, False)
    return x_normed


hc_pre_norm_test = pl.jit(_hc_pre_norm_entry)
