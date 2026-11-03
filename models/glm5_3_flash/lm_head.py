# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""GLM-5.3-Flash vocabulary projections, greedy selection and TP reduction."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __name__ == "__main__" and not any(
    tok in ("--tp", "--ep") or tok.startswith(("--tp=", "--ep=")) for tok in sys.argv
):
    sys.argv += ["--tp", "2", "--ep", "2"]


import pypto.language as pl
import pypto.language.distributed as pld
import torch

from models.glm5_3_flash.config import EP_SIZE, FLASH, LOCAL_VOCAB, TP_SIZE, D
from models.glm5_3_flash.rmsnorm import rms_norm, rmsnorm
from models.glm5_3_flash.utils import Checkpoint, component_hidden

LOGIT_ROWS_DYN = pl.dynamic('GLM53_LOGIT_ROWS_DYN')
T_DYN = pl.dynamic('GLM53_T_DYN')

FP32_HEAD_EPS = FLASH.rms_norm_eps
FP32_NEG_INF_BITS = -8388608
SELECT_D_TILE = 512
LM_ROW_TILE = 16
LM_VOCAB_TILE = 128
LM_K_TILE = 512
MAX_M_TILE = 8
MAX_N_TILE = 256
TP_CANDIDATE_TILE = 8
FP32_HEAD_N_TILE = 64
FP32_HEAD_K_TILE = 128
FP32_HEAD_REDUCE_TILE = 8
FP32_HEAD_VOCAB_PAD = ((LOCAL_VOCAB + FP32_HEAD_N_TILE - 1) // FP32_HEAD_N_TILE) * FP32_HEAD_N_TILE
TP_TEST_EPOCHS = 3
TP_TEST_ROWS = TP_SIZE + 17


@pl.jit.inline
def lm_head(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    row_indices: pl.Tensor[[LOGIT_ROWS_DYN], pl.INT32],
    norm_weight: pl.Tensor[[D], pl.BF16],
    weight: pl.Tensor[[LOCAL_VOCAB, D], pl.BF16],
    local_max_value: pl.Tensor[[LOGIT_ROWS_DYN], pl.FP32],
    local_argmax: pl.Tensor[[LOGIT_ROWS_DYN], pl.INT32],
):
    """Return a shard-local maximum; row -1 emits (-inf, -1).

    Other row indices must be in [0, T). The caller adds the vocabulary offset
    and combines the local candidates across TP ranks.
    """
    rows = pl.tensor.dim(row_indices, 0)
    hidden_rows = pl.tensor.dim(x, 0)
    selected = pl.create_tensor([rows, D], dtype=pl.BF16)
    for block in pl.spmd(rows * (D // SELECT_D_TILE), name_hint="lm_head_select"):
        row = block // (D // SELECT_D_TILE)
        d0 = block % (D // SELECT_D_TILE) * SELECT_D_TILE
        source_row = pl.cast(pl.read(row_indices, [row]), pl.INDEX)
        selected[row:row + 1, d0:d0 + SELECT_D_TILE] = pl.full([1, SELECT_D_TILE], dtype=pl.BF16, value=0.0)
        if source_row >= 0:
            if source_row < hidden_rows:
                source = x[source_row : source_row + 1, d0 : d0 + SELECT_D_TILE]
                selected[row : row + 1, d0 : d0 + SELECT_D_TILE] = source

    normalized = pl.create_tensor([rows, D], dtype=pl.BF16)
    rmsnorm(selected, norm_weight, normalized)

    logits = pl.create_tensor([rows, LOCAL_VOCAB], dtype=pl.FP32)
    for row0 in pl.parallel(0, rows, LM_ROW_TILE):
        for block in pl.spmd((LOCAL_VOCAB + LM_VOCAB_TILE - 1) // LM_VOCAB_TILE, name_hint="lm_head_projection"):
            vocab0 = block * LM_VOCAB_TILE
            valid_rows = pl.min(LM_ROW_TILE, rows - row0)
            valid_vocab = pl.min(LM_VOCAB_TILE, LOCAL_VOCAB - vocab0)
            accum = pl.create_tensor([LM_ROW_TILE, LM_VOCAB_TILE], dtype=pl.FP32)
            for k0 in pl.pipeline(0, D, LM_K_TILE, stage=2):
                hidden = pl.slice(normalized, [LM_ROW_TILE, LM_K_TILE], [row0, k0], valid_shape=[valid_rows, LM_K_TILE])
                vocab = pl.slice(weight, [LM_VOCAB_TILE, LM_K_TILE], [vocab0, k0], valid_shape=[valid_vocab, LM_K_TILE])
                accum = pl.matmul_acc(accum, hidden, vocab, b_trans=True, init_cond=(k0 == 0))
            accum_valid = pl.set_validshape(accum, valid_rows, valid_vocab)
            logits[row0 : row0 + LM_ROW_TILE, vocab0 : vocab0 + LM_VOCAB_TILE] = accum_valid

    # One writer owns the compact output arrays and their shared cache lines.
    for _block in pl.spmd(1, name_hint="lm_head_local_argmax"):
        negative_inf_bits = pl.full([1, 8], dtype=pl.INT32, value=FP32_NEG_INF_BITS)
        negative_inf = pl.read(pl.reinterpret_view(negative_inf_bits, pl.FP32), [0, 0])
        for row in pl.range(rows):
            best_value = negative_inf
            best_index = pl.cast(-1, pl.INT32)
            source_row = pl.cast(pl.read(row_indices, [row]), pl.INDEX)
            if source_row >= 0:
                if source_row < hidden_rows:
                    for vocab0 in pl.range(0, LOCAL_VOCAB, MAX_N_TILE):
                        valid_vocab = pl.min(MAX_N_TILE, LOCAL_VOCAB - vocab0)
                        scores = pl.slice(logits, [MAX_M_TILE, MAX_N_TILE], [row, vocab0], valid_shape=[1, valid_vocab])
                        scores = pl.fillpad(scores, pad_value=pl.PadValue.min)
                        scores = pl.set_validshape(scores, MAX_M_TILE, MAX_N_TILE)
                        value = pl.read(pl.row_max(scores), [0, 0])
                        if value > best_value:
                            index_ramp = pl.arange(0, [1, MAX_N_TILE], dtype=pl.INT32)
                            index_zeros = pl.full([MAX_M_TILE, MAX_N_TILE], dtype=pl.INT32, value=0)
                            indices = pl.col_expand(index_zeros, index_ramp)
                            matches = pl.cast(pl.cmp(scores, value, cmp_type=0), pl.INT32)
                            candidates = pl.add(pl.mul(matches, pl.add(indices, -1073741824)), 1073741824)
                            offset = pl.read(pl.row_min(candidates), [0, 0])
                            best_value = value
                            best_index = pl.cast(vocab0, pl.INT32) + offset
            pl.write(local_max_value, [row], best_value)
            pl.write(local_argmax, [row], best_index)
    return local_max_value, local_argmax


@pl.jit
def lm_head_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    row_indices: pl.Tensor[[LOGIT_ROWS_DYN], pl.INT32],
    norm_weight: pl.Tensor[[D], pl.BF16],
    weight: pl.Tensor[[LOCAL_VOCAB, D], pl.BF16],
    local_max_value: pl.Out[pl.Tensor[[LOGIT_ROWS_DYN], pl.FP32]],
    local_argmax: pl.Out[pl.Tensor[[LOGIT_ROWS_DYN], pl.INT32]],
):
    x.bind_dynamic(0, T_DYN)
    row_indices.bind_dynamic(0, LOGIT_ROWS_DYN)
    local_max_value.bind_dynamic(0, LOGIT_ROWS_DYN)
    local_argmax.bind_dynamic(0, LOGIT_ROWS_DYN)
    lm_head(x, row_indices, norm_weight, weight, local_max_value, local_argmax)
    return local_max_value, local_argmax


@pl.jit.inline
def lm_head_project_fp32(
    x: pl.Tensor[[T_DYN, D], pl.FP32],
    weight: pl.Tensor[[FP32_HEAD_VOCAB_PAD, D], pl.FP32],
    logits: pl.Tensor[[T_DYN, FP32_HEAD_VOCAB_PAD], pl.FP32],
):
    """Project FP32 activations without imposing another normalization boundary."""
    rows = pl.tensor.dim(x, 0)
    for row0 in pl.parallel(0, rows, LM_ROW_TILE):
        for block in pl.spmd(FP32_HEAD_VOCAB_PAD // FP32_HEAD_N_TILE, name_hint="lm_head_fp32_projection"):
            vocab0 = block * FP32_HEAD_N_TILE
            valid_rows = pl.min(LM_ROW_TILE, rows - row0)
            accum = pl.create_tensor([LM_ROW_TILE, FP32_HEAD_N_TILE], dtype=pl.FP32)
            for k0 in pl.pipeline(0, D, FP32_HEAD_K_TILE, stage=2):
                projection_hidden = pl.slice(x, [LM_ROW_TILE, FP32_HEAD_K_TILE], [row0, k0], valid_shape=[valid_rows, FP32_HEAD_K_TILE])
                projection_vocab = weight[vocab0 : vocab0 + FP32_HEAD_N_TILE, k0 : k0 + FP32_HEAD_K_TILE]
                accum = pl.matmul_acc(accum, projection_hidden, projection_vocab, b_trans=True, init_cond=k0 == 0)
            accum = pl.set_validshape(accum, valid_rows, FP32_HEAD_N_TILE)
            logits[row0 : row0 + LM_ROW_TILE, vocab0 : vocab0 + FP32_HEAD_N_TILE] = accum
    return logits


@pl.jit.inline
def lm_head_logits(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    norm_weight: pl.Tensor[[D], pl.BF16],
    weight: pl.Tensor[[FP32_HEAD_VOCAB_PAD, D], pl.FP32],
    logits: pl.Tensor[[T_DYN, FP32_HEAD_VOCAB_PAD], pl.FP32],
):
    """Normalize BF16 rows in FP32 and project the complete local vocabulary.

    The vocabulary shard contains exactly widened BF16 checkpoint weights and
    zero padding to a complete projection tile. Normalized activations and the
    output stay FP32. The caller strips padding before concatenating TP shards.
    """
    rows = pl.tensor.dim(x, 0)
    normalized = pl.create_tensor([rows, D], dtype=pl.FP32)
    for row in pl.spmd(rows, name_hint="lm_head_fp32_norm"):
        hidden = pl.cast(x[row : row + 1, 0:D], pl.FP32)
        squared = pl.mul(hidden, hidden)
        squared_rows = pl.reshape(squared, [FP32_HEAD_REDUCE_TILE, D // FP32_HEAD_REDUCE_TILE])
        partial_sum = pl.row_sum(squared_rows)
        partial_row = pl.reshape(partial_sum, [1, FP32_HEAD_REDUCE_TILE])
        reduction = pl.full([FP32_HEAD_REDUCE_TILE, FP32_HEAD_REDUCE_TILE], dtype=pl.FP32, value=0.0)
        reduction[0:1, 0:FP32_HEAD_REDUCE_TILE] = partial_row
        reduction = pl.set_validshape(reduction, 1, FP32_HEAD_REDUCE_TILE)
        squared_sum = pl.row_sum(reduction)
        mean_squared = pl.mul(squared_sum, 1.0 / D)
        variance = pl.add(mean_squared, FP32_HEAD_EPS)
        inverse_rms = pl.rsqrt(variance, high_precision=True)
        inverse_rms_value = pl.read(inverse_rms, [0, 0])
        scaled = pl.mul(hidden, inverse_rms_value)
        gamma_values = pl.cast(norm_weight[0:D], pl.FP32)
        gamma = pl.reshape(gamma_values, [1, D])
        normalized_row = pl.mul(scaled, gamma)
        normalized[row : row + 1, 0:D] = normalized_row

    lm_head_project_fp32(normalized, weight, logits)
    return logits


@pl.jit.inline
def lm_head_mtp_logits(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    norm_weight: pl.Tensor[[D], pl.BF16],
    weight: pl.Tensor[[FP32_HEAD_VOCAB_PAD, D], pl.FP32],
    logits: pl.Tensor[[T_DYN, FP32_HEAD_VOCAB_PAD], pl.FP32],
):
    """Apply the MTP shared head's BF16 norm once, then project in FP32."""
    rows = pl.tensor.dim(x, 0)
    normalized = pl.create_tensor([rows, D], dtype=pl.BF16)
    rmsnorm(x, norm_weight, normalized)
    widened = pl.create_tensor([rows, D], dtype=pl.FP32)
    for block in pl.spmd(((rows + 7) // 8) * (D // 128), name_hint="mtp_head_widen"):
        row0 = block // (D // 128) * 8
        col0 = block % (D // 128) * 128
        valid = pl.min(8, rows - row0)
        source = pl.slice(normalized, [8, 128], [row0, col0], valid_shape=[valid, 128])
        widened[row0 : row0 + 8, col0 : col0 + 128] = pl.cast(source, pl.FP32)
    lm_head_project_fp32(widened, weight, logits)
    return logits


@pl.jit
def lm_head_mtp_logits_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    norm_weight: pl.Tensor[[D], pl.BF16],
    weight: pl.Tensor[[FP32_HEAD_VOCAB_PAD, D], pl.FP32],
    logits: pl.Out[pl.Tensor[[T_DYN, FP32_HEAD_VOCAB_PAD], pl.FP32]],
):
    x.bind_dynamic(0, T_DYN)
    logits.bind_dynamic(0, T_DYN)
    lm_head_mtp_logits(x, norm_weight, weight, logits)
    return logits


@pl.jit
def lm_head_logits_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    norm_weight: pl.Tensor[[D], pl.BF16],
    weight: pl.Tensor[[FP32_HEAD_VOCAB_PAD, D], pl.FP32],
    logits: pl.Out[pl.Tensor[[T_DYN, FP32_HEAD_VOCAB_PAD], pl.FP32]],
):
    x.bind_dynamic(0, T_DYN)
    logits.bind_dynamic(0, T_DYN)
    lm_head_logits(x, norm_weight, weight, logits)
    return logits


@pl.jit.inline(auto_scope=False)
def lm_head_tp_reduce(
    local_max_value: pl.Tensor[[LOGIT_ROWS_DYN], pl.FP32],
    local_argmax: pl.Tensor[[LOGIT_ROWS_DYN], pl.INT32],
    value_exchange: pld.DistributedTensor[[LOGIT_ROWS_DYN, TP_CANDIDATE_TILE], pl.FP32],
    index_exchange: pld.DistributedTensor[[LOGIT_ROWS_DYN, TP_CANDIDATE_TILE], pl.INT32],
    arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    global_max_value: pl.Tensor[[LOGIT_ROWS_DYN], pl.FP32],
    global_argmax: pl.Tensor[[LOGIT_ROWS_DYN], pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    """Reduce TP candidates, resolving equal scores to the lowest global ID.

    Local IDs outside [0, LOCAL_VOCAB), NaNs, and -inf scores are absent
    candidates. An all-absent row returns (-inf, -1). Epochs start at one and
    increase whenever the caller reuses these windows and notification counts.
    """
    rows = pl.tensor.dim(local_argmax, 0)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_tp_reuse", allow_early_resolve=False) as reuse_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(arrived, offsets=[peer, 0], expected=(epoch - 1) * 2, cmp=pld.WaitCmp.Ge)

    with pl.spmd(1, name_hint="lm_tp_publish", deps=[reuse_tid], allow_early_resolve=False) as publish_tid:
        publish_writer = pl.tile.get_block_idx()
        inf_bits = pl.full([1, TP_CANDIDATE_TILE], dtype=pl.INT32, value=FP32_NEG_INF_BITS)
        negative_inf = pl.read(pl.reinterpret_view(inf_bits, pl.FP32), [0, 0])
        for row in pl.range(publish_writer, rows, 1):
            value = pl.read(local_max_value, [row])
            index = pl.read(local_argmax, [row])
            candidate_value = negative_inf
            candidate_index = pl.cast(-1, pl.INT32)
            if index >= 0:
                if index < LOCAL_VOCAB:
                    if value > negative_inf:
                        candidate_value = value
                        candidate_index = pl.cast(index + tp_rank * LOCAL_VOCAB, pl.INT32)
            value_row = pl.tile.full([1, TP_CANDIDATE_TILE], dtype=pl.FP32, value=0.0)
            index_row = pl.tile.full([1, TP_CANDIDATE_TILE], dtype=pl.INT32, value=-1)
            pl.tile.write(value_row, [0, 0], candidate_value)
            pl.tile.write(index_row, [0, 0], candidate_index)
            pl.store(value_row, [row, 0], value_exchange)
            pl.store(index_row, [row, 0], index_exchange)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_tp_ready", deps=[publish_tid]) as ready_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(arrived, peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_tp_wait", deps=[ready_tid], allow_early_resolve=False) as wait_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(arrived, offsets=[peer, 0], expected=epoch * 2 - 1, cmp=pld.WaitCmp.Ge)

    # One writer owns both compact arrays and their shared cache lines.
    with pl.spmd(1, name_hint="lm_tp_reduce", deps=[wait_tid]) as reduce_tid:
        reduce_writer = pl.tile.get_block_idx()
        inf_bits = pl.full([1, TP_CANDIDATE_TILE], dtype=pl.INT32, value=FP32_NEG_INF_BITS)
        negative_inf = pl.read(pl.reinterpret_view(inf_bits, pl.FP32), [0, 0])
        for row in pl.range(reduce_writer, rows, 1):
            best_value = negative_inf
            best_index = pl.cast(-1, pl.INT32)
            for peer in pl.range(TP_SIZE):
                peer_value = pld.tile.remote_load(
                    value_exchange, peer=group_base + peer, offsets=[row, 0], shape=[1, TP_CANDIDATE_TILE],
                )
                peer_index = pld.tile.remote_load(
                    index_exchange, peer=group_base + peer, offsets=[row, 0], shape=[1, TP_CANDIDATE_TILE],
                )
                value = pl.tile.read(peer_value, [0, 0])
                index = pl.tile.read(peer_index, [0, 0])
                if index >= 0:
                    if value > best_value:
                        best_value = value
                        best_index = pl.cast(index, pl.INT32)
                    elif value == best_value:
                        if best_index < 0:
                            best_index = pl.cast(index, pl.INT32)
                        elif index < best_index:
                            best_index = pl.cast(index, pl.INT32)
            pl.write(global_max_value, [row], best_value)
            pl.write(global_argmax, [row], best_index)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_tp_release", deps=[reduce_tid]) as release_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(arrived, peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_tp_consumed", deps=[release_tid], allow_early_resolve=False):
        for peer in pl.range(TP_SIZE):
            pld.system.wait(arrived, offsets=[peer, 0], expected=epoch * 2, cmp=pld.WaitCmp.Ge)
    return global_max_value, global_argmax


@pl.jit(auto_scope=False)
def lm_head_tp_reduce_test(
    local_values: pl.Tensor[[TP_TEST_ROWS], pl.FP32],
    local_indices: pl.Tensor[[TP_TEST_ROWS], pl.INT32],
    value_exchange: pld.DistributedTensor[[TP_TEST_ROWS, TP_CANDIDATE_TILE], pl.FP32],
    index_exchange: pld.DistributedTensor[[TP_TEST_ROWS, TP_CANDIDATE_TILE], pl.INT32],
    arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    global_values: pl.Out[pl.Tensor[[TP_TEST_ROWS], pl.FP32]],
    global_indices: pl.Out[pl.Tensor[[TP_TEST_ROWS], pl.INT32]],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    """Reduce within the rank's TP group in the complete device world."""
    group_base = (rank // TP_SIZE) * TP_SIZE
    tp_rank = rank % TP_SIZE
    lm_head_tp_reduce(
        local_values, local_indices, value_exchange, index_exchange, arrived, global_values, global_indices,
        tp_rank, group_base, epoch,
    )
    return global_values, global_indices


@pl.jit.host
def l3_lm_head_tp_reduce_test(
    local_values: pl.Tensor[[TP_TEST_EPOCHS, EP_SIZE, TP_TEST_ROWS], pl.FP32],
    local_indices: pl.Tensor[[TP_TEST_EPOCHS, EP_SIZE, TP_TEST_ROWS], pl.INT32],
    global_values: pl.Out[pl.Tensor[[TP_TEST_EPOCHS, EP_SIZE, TP_TEST_ROWS], pl.FP32]],
    global_indices: pl.Out[pl.Tensor[[TP_TEST_EPOCHS, EP_SIZE, TP_TEST_ROWS], pl.INT32]],
):
    """Keep exchange windows and notification counts across three changed inputs."""
    value_buffer = pld.alloc_window_buffer([TP_TEST_ROWS, TP_CANDIDATE_TILE], dtype=pl.FP32)
    index_buffer = pld.alloc_window_buffer([TP_TEST_ROWS, TP_CANDIDATE_TILE], dtype=pl.INT32)
    arrived_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for epoch in pl.range(TP_TEST_EPOCHS):
        for rank in pl.range(pld.world_size()):
            value_exchange = pld.window(value_buffer, [TP_TEST_ROWS, TP_CANDIDATE_TILE], dtype=pl.FP32)
            index_exchange = pld.window(index_buffer, [TP_TEST_ROWS, TP_CANDIDATE_TILE], dtype=pl.INT32)
            arrived = pld.window(arrived_buffer, [TP_SIZE, 1], dtype=pl.INT32)
            lm_head_tp_reduce_test(
                local_values[epoch, rank], local_indices[epoch, rank], value_exchange, index_exchange,
                arrived, global_values[epoch, rank], global_indices[epoch, rank], rank, epoch + 1,
                device=rank,
            )


def reference_lm_head_logits(x, norm_weight, weight, *, vocab_block=4096):
    """Keep FP32 normalization boundaries and round independent FP64 dots to FP32.

    CPU FP32 GEMM can itself exceed the all-element 1e-5 acceptance gate near
    cancellation. The wide dot product avoids using its reduction error as an
    expected value. This does not change the gate or the device calculation.
    """
    hidden = x.float()
    inverse_rms = torch.rsqrt(hidden.square().mean(dim=-1, keepdim=True) + FP32_HEAD_EPS)
    normalized = (hidden * inverse_rms * norm_weight.float()).double()
    output = torch.empty((hidden.shape[0], weight.shape[0]), dtype=torch.float32)
    for begin in range(0, weight.shape[0], vocab_block):
        end = min(begin + vocab_block, weight.shape[0])
        output[:, begin:end] = (normalized @ weight[begin:end].double().t()).float()
    return output


def golden_lm_head_logits(tensors):
    tensors["logits"].copy_(reference_lm_head_logits(tensors["x"], tensors["norm_weight"], tensors["weight"]))


def run_fp32_case(options, device_id):
    from functools import lru_cache

    from golden import TensorSpec, ratio_allclose, run

    if not 0 <= options.shard < TP_SIZE:
        raise ValueError(f"--shard must be in [0, {TP_SIZE})")

    @lru_cache(maxsize=1)
    def inputs():
        generator = torch.Generator().manual_seed(5357)
        x = component_hidden(options, D, generator)
        weight = torch.zeros(FP32_HEAD_VOCAB_PAD, D, dtype=torch.float32)
        if options.checkpoint is None:
            weight[:LOCAL_VOCAB] = (torch.randn(LOCAL_VOCAB, D, generator=generator) * 0.02).bfloat16().float()
            norm = torch.ones(D, dtype=torch.bfloat16)
        else:
            checkpoint = Checkpoint(options.checkpoint)
            rows = slice(options.shard * LOCAL_VOCAB, (options.shard + 1) * LOCAL_VOCAB)
            head_name = "lm_head.weight"
            norm_names = ("model.language_model.norm.weight", "model.norm.weight")
            if options.case == "mtp_fp32":
                layer = FLASH.mtp_layer_id
                norm_names = (
                    f"model.language_model.layers.{layer}.shared_head.norm.weight",
                    f"model.layers.{layer}.shared_head.norm.weight",
                )
            names = [name for name in norm_names if name in checkpoint]
            if len(names) != 1:
                raise ValueError(f"checkpoint norm must resolve exactly once: expected one of {norm_names}")
            if options.case == "mtp_fp32":
                head_name = names[0].replace("norm.weight", "head.weight")
            weight[:LOCAL_VOCAB] = checkpoint.read(head_name, (rows, slice(None))).float()
            norm = checkpoint.read(names[0])
        return {"x": x, "norm_weight": norm, "weight": weight}

    specs = [
        TensorSpec("x", (options.tokens, D), torch.bfloat16, init_value=lambda: inputs()["x"]),
        TensorSpec("norm_weight", (D,), torch.bfloat16, init_value=lambda: inputs()["norm_weight"]),
        TensorSpec("weight", (FP32_HEAD_VOCAB_PAD, D), torch.float32, init_value=lambda: inputs()["weight"]),
        TensorSpec("logits", (options.tokens, FP32_HEAD_VOCAB_PAD), torch.float32),
    ]

    def golden(values):
        if options.case == "mtp_fp32":
            normalized = rms_norm(values["x"], values["norm_weight"]).double()
            values["logits"].copy_((normalized @ values["weight"].double().t()).float())
        else:
            golden_lm_head_logits(values)

    fn = lm_head_mtp_logits_test if options.case == "mtp_fp32" else lm_head_logits_test
    result = run(
        fn=fn,
        specs=specs,
        golden_fn=golden,
        config=dict(backend_type=options.backend_type, execution_mode=options.execution_mode, device_id=device_id),
        compare_fn={"logits": ratio_allclose(atol=1e-05, rtol=1e-05, max_error_ratio=0)},
        compile_only=options.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)


def golden_lm_head(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Return full local FP32 logits after the BF16 RMSNorm boundary."""
    normalized = rms_norm(x, norm_weight).to(torch.bfloat16)
    return torch.nn.functional.linear(normalized.float(), weight.float())


def golden_lm_head_tp_reduce(local_max_value: torch.Tensor, local_argmax: torch.Tensor):
    """Merge one [TP_SIZE, R] stack of local candidates into global candidates."""
    if local_max_value.shape != local_argmax.shape or local_max_value.ndim != 2:
        raise ValueError("local candidate values and IDs must have equal [TP_SIZE, R] shapes")
    if local_max_value.shape[0] != TP_SIZE:
        raise ValueError(f"expected {TP_SIZE} vocabulary shards")
    valid = (local_argmax >= 0) & (local_argmax < LOCAL_VOCAB)
    valid &= ~torch.isnan(local_max_value) & ~torch.isneginf(local_max_value)
    values = local_max_value.float().masked_fill(~valid, float("-inf"))
    best_value = values.max(dim=0).values
    ranks = torch.arange(TP_SIZE, device=local_argmax.device, dtype=torch.int64).unsqueeze(1)
    global_ids = local_argmax.to(torch.int64) + ranks * LOCAL_VOCAB
    winners = valid & (values == best_value.unsqueeze(0))
    candidates = global_ids.masked_fill(~winners, torch.iinfo(torch.int32).max)
    best_index = candidates.min(dim=0).values.to(torch.int32)
    best_index[~valid.any(dim=0)] = -1
    return best_value, best_index


def golden_lm_head_tp_reduce_test(tensors):
    """Fill every group's replicas using the candidate-reduction reference."""
    for epoch in range(TP_TEST_EPOCHS):
        for start in range(0, EP_SIZE, TP_SIZE):
            group = slice(start, start + TP_SIZE)
            (values, indices) = golden_lm_head_tp_reduce(
                tensors["local_values"][epoch, group], tensors["local_indices"][epoch, group],
            )
            tensors["global_values"][epoch, group] = values.unsqueeze(0)
            tensors["global_indices"][epoch, group] = indices.unsqueeze(0)


def golden_lm_head_case(tensors):
    """Fill local candidates for selected rows, including -1 padding."""
    rows = tensors["row_indices"].to(torch.int64)
    if bool(((rows < -1) | (rows >= tensors["x"].shape[0])).any()):
        raise ValueError("logit row indices must be -1 or valid hidden-state row indices")
    tensors["local_max_value"].fill_(float("-inf"))
    tensors["local_argmax"].fill_(-1)
    active = rows >= 0
    if bool(active.any()):
        selected = tensors["x"].index_select(0, rows[active])
        logits = golden_lm_head(selected, tensors["norm_weight"], tensors["weight"])
        values, indices = logits.max(dim=-1)
        tensors["local_max_value"][active] = values
        tensors["local_argmax"][active] = indices.to(torch.int32)


def compare_local_max(actual, expected, **kwargs):
    """Check -inf padding exactly and compare finite candidates with the harness."""
    from golden import ratio_allclose

    active = torch.isfinite(expected)
    padding = torch.isneginf(expected)
    if not bool((active | padding).all()):
        return False, "expected maxima must be finite or -inf padding"
    if not bool(torch.isneginf(actual[padding]).all()):
        return False, "padded logit rows must emit -inf maxima"
    if not bool(active.any()):
        return True, ""
    compare = ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0)
    return compare(actual[active], expected[active], **kwargs)


def compare_local_argmax(actual, expected, **_kwargs):
    """Require exact local token indices and -1 padding."""
    return bool(torch.equal(actual, expected)), "local argmax indices differ"


def build_lm_head_tensor_specs(case: str):
    """Cover selected rows, ties, negative vocabulary tails and padding."""
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(53)
    tokens = 19 if case == "selection" else 3 if case == "ties" else 5 if case == "negative_tail" else 1
    x = torch.randn(tokens, D, generator=generator, dtype=torch.bfloat16)
    norm_weight = (1.0 + 0.05 * torch.randn(D, generator=generator)).bfloat16()
    weight = torch.randn(LOCAL_VOCAB, D, generator=generator, dtype=torch.bfloat16).mul_(0.01)
    if case in ("decode", "selection"):
        labels = (torch.arange(tokens) * 347 + 53).remainder(LOCAL_VOCAB)
        if case == "decode":
            labels[0] = LOCAL_VOCAB - 1
            row_indices = torch.tensor([0], dtype=torch.int32)
        else:
            labels[:6] = torch.tensor([0, 127, 128, 1023, 1024, LOCAL_VOCAB - 1])
            row_indices = torch.tensor([18, 0, 7, 7, 16, 1, 15, 2, -1, 14, 3, 13, 4, 12, 5, 11, 6], dtype=torch.int32)
        weight[labels] = rms_norm(x, norm_weight).mul(2).bfloat16()
    else:
        x.fill_(1)
        norm_weight.fill_(1)
        weight.fill_(-1)
        if case == "ties":
            weight[torch.tensor([7, 42, 1024, LOCAL_VOCAB - 1])] = 1
            row_indices = torch.tensor([2, 0, 2], dtype=torch.int32)
        elif case == "negative_tail":
            weight[-1] = -0.5
            row_indices = torch.tensor([4], dtype=torch.int32)
        elif case == "padding":
            row_indices = torch.tensor([-1], dtype=torch.int32)
        else:
            raise ValueError(f"unknown LM-head case: {case}")
    rows = row_indices.numel()
    return [
        TensorSpec("x", [tokens, D], torch.bfloat16, init_value=x),
        TensorSpec("row_indices", [rows], torch.int32, init_value=row_indices),
        TensorSpec("norm_weight", [D], torch.bfloat16, init_value=norm_weight),
        TensorSpec("weight", [LOCAL_VOCAB, D], torch.bfloat16, init_value=weight),
        TensorSpec("local_max_value", [rows], torch.float32),
        TensorSpec("local_argmax", [rows], torch.int32),
    ]


def build_tp_reduce_tensor_specs():
    """Cover every winning peer, absent/invalid candidates, ties and stale epochs."""
    from golden import TensorSpec

    shape = [TP_TEST_EPOCHS, EP_SIZE, TP_TEST_ROWS]
    values = torch.full(shape, float("-inf"), dtype=torch.float32)
    indices = torch.full(shape, -1, dtype=torch.int32)
    for start in range(0, EP_SIZE, TP_SIZE):
        group = slice(start, start + TP_SIZE)
        group_offset = start * 100
        for rank in range(TP_SIZE):
            peer = start + rank
            values[0, peer] = -rank - 1 - group_offset
            indices[0, peer] = torch.arange(TP_TEST_ROWS, dtype=torch.int32)
            values[0, peer, rank] = 10 + rank + group_offset
            values[2, peer] = -TP_SIZE + rank - group_offset
            indices[2, peer] = LOCAL_VOCAB - 1 - rank

        # Equal scores choose the first global ID, even with different local IDs.
        values[0, group, TP_SIZE] = -0.5 - group_offset
        indices[0, group, TP_SIZE] = torch.arange(TP_SIZE, 0, -1, dtype=torch.int32)
        indices[0, group, TP_SIZE + 1] = -1
        values[0, start, TP_SIZE + 2] = 1e6
        indices[0, start, TP_SIZE + 2] = LOCAL_VOCAB
        values[0, group, TP_SIZE + 3] = float("nan")
        values[0, group, TP_SIZE + 4] = float("-inf")
        values[0, start, TP_SIZE + 5] = 1e6
        indices[0, start, TP_SIZE + 5] = -2
        indices[0, group, TP_SIZE + 6] = LOCAL_VOCAB - 1
        values[0, group, TP_SIZE + 7] = float("nan")
        values[0, start, TP_SIZE + 7] = -7 - group_offset
        indices[0, start, TP_SIZE + 7] = 0

        # An entirely absent epoch must overwrite every previous valid result.
        values[1, group] = 1e6
        indices[1, group] = LOCAL_VOCAB
        values[2, group, TP_SIZE] = -0.25 - group_offset
        indices[2, group, TP_SIZE] = 0
    return [
        TensorSpec("local_values", shape, torch.float32, init_value=values),
        TensorSpec("local_indices", shape, torch.int32, init_value=indices),
        TensorSpec("global_values", shape, torch.float32),
        TensorSpec("global_indices", shape, torch.int32),
    ]


def main():
    import argparse
    import subprocess

    from pypto.ir import DistributedConfig

    from golden import run

    cases = ("decode", "selection", "ties", "negative_tail", "padding")
    fp32_cases = ("fp32", "mtp_fp32")
    parser = argparse.ArgumentParser(description="GLM-5.3-Flash local LM-head and TP reduction validation")
    parser.add_argument("-p", "--platform", dest="backend_type", default="a2a3", choices=["a2a3"])
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", default=",".join(str(i) for i in range(EP_SIZE)))
    parser.add_argument("--tp", type=int, default=TP_SIZE)
    parser.add_argument("--ep", type=int, default=EP_SIZE)
    parser.add_argument("--case", choices=("all", *cases, "tp_reduce", *fp32_cases), default="all")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--hidden-input", type=Path)
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()
    device_ids = [int(device) for device in args.device.split(",")]
    if args.checkpoint is not None or args.hidden_input is not None:
        if args.case not in fp32_cases:
            parser.error("--checkpoint and --hidden-input require --case fp32 or mtp_fp32")
    if args.tokens < 1:
        parser.error("--tokens must be positive")
    if args.case in fp32_cases:
        run_fp32_case(args, device_ids[0])
        return
    if args.case in ("all", "tp_reduce") and len(device_ids) != args.ep:
        parser.error(f"TP validation needs exactly {args.ep} devices")
    if args.case == "all":
        # Separate processes own the single-device and distributed runtime lifetimes.
        for case in (*cases, *fp32_cases, "tp_reduce"):
            devices = args.device if case == "tp_reduce" else str(device_ids[0])
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "-p",
                args.backend_type,
                "-d",
                devices,
                "--tp",
                str(args.tp),
                "--ep",
                str(args.ep),
                "--case",
                case,
            ]
            if case in fp32_cases:
                command.extend(("--tokens", str(args.tokens), "--shard", str(args.shard)))
            if args.execution_mode:
                command.append("--sim")
            if args.compile_only:
                command.append("--compile-only")
            result = subprocess.run(command, check=False)
            if result.returncode:
                raise SystemExit(result.returncode)
        return
    if args.case == "tp_reduce":
        result = run(
            fn=l3_lm_head_tp_reduce_test,
            specs=build_tp_reduce_tensor_specs(),
            golden_fn=golden_lm_head_tp_reduce_test,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0)),
            compare_fn={"global_values": compare_local_max, "global_indices": compare_local_argmax},
            compile_only=args.compile_only,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
        print(f"[GOLDEN] PASS lm_head tp_reduce TP{args.tp}/EP{args.ep} epochs={TP_TEST_EPOCHS}")
        return
    for case in cases:
        if args.case not in ("all", case):
            continue
        result = run(
            fn=lm_head_test,
            specs=build_lm_head_tensor_specs(case),
            golden_fn=golden_lm_head_case,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=device_ids[0]),
            compare_fn={"local_max_value": compare_local_max, "local_argmax": compare_local_argmax},
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
        print(f"[GOLDEN] PASS lm_head {case}")


if __name__ == "__main__":
    main()
