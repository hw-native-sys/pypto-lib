# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fused cross-entropy forward + backward — loss = mean(logsumexp(logits) - logits[row, label]); dlogits = (softmax(logits) - onehot) / M. bf16 input, fp32 sensitive arithmetic."""
import pypto.language as pl

DEFAULT_SHAPE = {"M": 512, "N": 1024, "dtype": "bf16"}


WORKERS = 48
STRIPES = 8
REDUCTION_LANES = 1024
LOSS_SLOT_LANES = 128
MEAN_ROWS = 128
PRIVATE_LOSS = True
LOSS_LANES = 32


@pl.jit
def cross_entropy_bf16(
    logits: pl.Tensor,
    labels: pl.Tensor,
    colidx: pl.Tensor,
    loss: pl.Out[pl.Tensor],
    dlogits: pl.Out[pl.Tensor],
):
    rows, cols = logits.shape
    padded_cols = ((cols + REDUCTION_LANES - 1) // REDUCTION_LANES) * REDUCTION_LANES
    workspace = pl.create_tensor([WORKERS, LOSS_SLOT_LANES], dtype=pl.FP32)
    with pl.spmd(WORKERS, name_hint="resident_rows", allow_early_resolve=True) as producer:
        pl.set_cache_policy(logits, pl.CachePolicy.BYPASS)
        worker = pl.tile.get_block_idx()
        loss_zero = pl.full([1, LOSS_LANES], dtype=pl.FP32, value=0.0)
        for row_index, (loss_state, partial_state, gradient_state) in pl.pipeline(
            worker, rows, WORKERS, stage=3, init_values=(loss_zero, workspace, dlogits)
        ):
            raw = pl.slice(logits, [1, padded_cols], [row_index, 0], valid_shape=[1, cols])
            values = pl.cast(raw, target_type=pl.FP32)
            masked = pl.fillpad(values, pad_value=pl.PadValue.min)
            parts = pl.reshape(masked, [STRIPES, padded_cols // STRIPES])
            compact_max = pl.col_max(pl.reshape(masked, [padded_cols // REDUCTION_LANES, REDUCTION_LANES]))
            maxima = pl.reshape(pl.row_max(pl.reshape(compact_max, [STRIPES, REDUCTION_LANES // STRIPES])), [1, STRIPES])
            first_max = pl.read(maxima, [0, 0])
            for stripe, (maximum_state,) in pl.range(1, STRIPES, init_values=(first_max,)):
                candidate_max = pl.read(maxima, [0, stripe])
                if candidate_max > maximum_state:
                    next_maximum = candidate_max
                else:
                    next_maximum = maximum_state
                row_max = pl.yield_(next_maximum)
            maximum_zero = pl.full([1, STRIPES], dtype=pl.FP32, value=0.0)
            maximum_column = pl.reshape(pl.add(maximum_zero, row_max), [STRIPES, 1])
            exponentials = pl.reshape(pl.row_expand_expdif(parts, maximum_column), [1, padded_cols])
            exponentials = pl.fillpad(pl.set_validshape(exponentials, 1, cols))
            sums = pl.reshape(pl.row_sum(pl.reshape(exponentials, [STRIPES, padded_cols // STRIPES])), [1, STRIPES])
            first_sum = pl.read(sums, [0, 0])
            for stripe, (sum_state,) in pl.range(1, STRIPES, init_values=(first_sum,)):
                row_sum = pl.yield_(sum_state + pl.read(sums, [0, stripe]))
            label_value = pl.read(labels, [0, row_index])
            label_index = pl.cast(label_value, pl.INDEX)
            target = pl.read(values, [0, label_index])
            target_exponential = pl.read(exponentials, [0, label_index])
            corrected = pl.write(exponentials, [0, label_index], target_exponential - row_sum)
            gradient_scale = (1.0 / rows) / row_sum
            gradient = pl.mul(corrected, gradient_scale)
            gradient = pl.set_validshape(gradient, 1, cols)
            gradient_next = pl.assemble(gradient_state, gradient, [row_index, 0])
            log_zero = pl.full([1, LOSS_LANES], dtype=pl.FP32, value=0.0)
            log_values = pl.log(pl.add(log_zero, row_sum))
            target_values = pl.add(log_zero, target)
            centered_target = pl.sub(target_values, row_max)
            loss_values = pl.sub(log_values, centered_target)
            partial_next = partial_state
            loss_next = pl.add(loss_state, loss_values)
            accumulated_loss, row_losses, gradient_output = pl.yield_(loss_next, partial_next, gradient_next)
        final_loss = pl.read(accumulated_loss, [0, 0])
        slot_zero = pl.full([1, LOSS_SLOT_LANES], dtype=pl.FP32, value=0.0)
        slot = pl.add(slot_zero, final_loss)
        workspace_output = pl.assemble(row_losses, slot, [worker, 0])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="resident_mean", deps=[producer]):
        block = pl.slice(workspace_output, [WORKERS, 8], [0, 0])
        total_lanes = pl.col_sum(block)
        total = pl.read(total_lanes, [0, 0])
        pl.write(loss, [0, 0], total * (1.0 / rows))
    return loss, gradient_output


@pl.jit
def cross_entropy_fp32(
    logits: pl.Tensor,
    labels: pl.Tensor,
    colidx: pl.Tensor,
    loss: pl.Out[pl.Tensor],
    dlogits: pl.Out[pl.Tensor],
):
    rows, cols = logits.shape
    padded_cols = ((cols + REDUCTION_LANES - 1) // REDUCTION_LANES) * REDUCTION_LANES
    workspace = pl.create_tensor([WORKERS, LOSS_SLOT_LANES], dtype=pl.FP32)
    with pl.spmd(WORKERS, name_hint="resident_rows", allow_early_resolve=True) as producer:
        pl.set_cache_policy(logits, pl.CachePolicy.BYPASS)
        worker = pl.tile.get_block_idx()
        loss_zero = pl.full([1, LOSS_LANES], dtype=pl.FP32, value=0.0)
        for row_index, (loss_state, partial_state, gradient_state) in pl.pipeline(
            worker, rows, WORKERS, stage=3, init_values=(loss_zero, workspace, dlogits)
        ):
            raw = pl.slice(logits, [1, padded_cols], [row_index, 0], valid_shape=[1, cols])
            values = raw
            masked = pl.fillpad(values, pad_value=pl.PadValue.min)
            parts = pl.reshape(masked, [STRIPES, padded_cols // STRIPES])
            compact_max = pl.col_max(pl.reshape(masked, [padded_cols // REDUCTION_LANES, REDUCTION_LANES]))
            maxima = pl.reshape(pl.row_max(pl.reshape(compact_max, [STRIPES, REDUCTION_LANES // STRIPES])), [1, STRIPES])
            first_max = pl.read(maxima, [0, 0])
            for stripe, (maximum_state,) in pl.range(1, STRIPES, init_values=(first_max,)):
                candidate_max = pl.read(maxima, [0, stripe])
                if candidate_max > maximum_state:
                    next_maximum = candidate_max
                else:
                    next_maximum = maximum_state
                row_max = pl.yield_(next_maximum)
            maximum_zero = pl.full([1, STRIPES], dtype=pl.FP32, value=0.0)
            maximum_column = pl.reshape(pl.add(maximum_zero, row_max), [STRIPES, 1])
            exponentials = pl.reshape(pl.row_expand_expdif(parts, maximum_column), [1, padded_cols])
            exponentials = pl.fillpad(pl.set_validshape(exponentials, 1, cols))
            sums = pl.reshape(pl.row_sum(pl.reshape(exponentials, [STRIPES, padded_cols // STRIPES])), [1, STRIPES])
            first_sum = pl.read(sums, [0, 0])
            for stripe, (sum_state,) in pl.range(1, STRIPES, init_values=(first_sum,)):
                row_sum = pl.yield_(sum_state + pl.read(sums, [0, stripe]))
            label_value = pl.read(labels, [0, row_index])
            label_index = pl.cast(label_value, pl.INDEX)
            target = pl.read(values, [0, label_index])
            target_exponential = pl.read(exponentials, [0, label_index])
            corrected = pl.write(exponentials, [0, label_index], target_exponential - row_sum)
            gradient_scale = (1.0 / rows) / row_sum
            gradient = pl.mul(corrected, gradient_scale)
            gradient = pl.set_validshape(gradient, 1, cols)
            gradient_next = pl.assemble(gradient_state, gradient, [row_index, 0])
            log_zero = pl.full([1, LOSS_LANES], dtype=pl.FP32, value=0.0)
            log_values = pl.log(pl.add(log_zero, row_sum))
            target_values = pl.add(log_zero, target)
            centered_target = pl.sub(target_values, row_max)
            loss_values = pl.sub(log_values, centered_target)
            partial_next = partial_state
            loss_next = pl.add(loss_state, loss_values)
            accumulated_loss, row_losses, gradient_output = pl.yield_(loss_next, partial_next, gradient_next)
        final_loss = pl.read(accumulated_loss, [0, 0])
        slot_zero = pl.full([1, LOSS_SLOT_LANES], dtype=pl.FP32, value=0.0)
        slot = pl.add(slot_zero, final_loss)
        workspace_output = pl.assemble(row_losses, slot, [worker, 0])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="resident_mean", deps=[producer]):
        block = pl.slice(workspace_output, [WORKERS, 8], [0, 0])
        total_lanes = pl.col_sum(block)
        total = pl.read(total_lanes, [0, 0])
        pl.write(loss, [0, 0], total * (1.0 / rows))
    return loss, gradient_output


@pl.jit
def cross_entropy_one_class(
    logits: pl.Tensor,
    labels: pl.Tensor,
    colidx: pl.Tensor,
    loss: pl.Out[pl.Tensor],
    dlogits: pl.Out[pl.Tensor],
):
    rows, cols = logits.shape
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="one_class"):
        for row_index in pl.range(rows):
            pl.write(dlogits, [row_index, 0], pl.cast(0.0, pl.FP32))
        pl.write(loss, [0, 0], pl.cast(0.0, pl.FP32))
    return loss, dlogits


@pl.jit
def cross_entropy_bf16_exact(
    logits: pl.Tensor,
    labels: pl.Tensor,
    colidx: pl.Tensor,
    loss: pl.Out[pl.Tensor],
    dlogits: pl.Out[pl.Tensor],
):
    rows, cols = logits.shape
    workspace = pl.create_tensor([WORKERS, LOSS_SLOT_LANES], dtype=pl.FP32)
    with pl.spmd(WORKERS, name_hint="resident_rows", allow_early_resolve=True) as producer:
        pl.set_cache_policy(logits, pl.CachePolicy.BYPASS)
        worker = pl.tile.get_block_idx()
        loss_zero = pl.full([1, LOSS_LANES], dtype=pl.FP32, value=0.0)
        for row_index, (loss_state, partial_state, gradient_state) in pl.pipeline(
            worker, rows, WORKERS, stage=3, init_values=(loss_zero, workspace, dlogits)
        ):
            raw = pl.slice(logits, [1, cols], [row_index, 0])
            values = pl.cast(raw, target_type=pl.FP32)
            masked = values
            parts = pl.reshape(masked, [STRIPES, cols // STRIPES])
            compact_max = pl.col_max(pl.reshape(masked, [cols // REDUCTION_LANES, REDUCTION_LANES]))
            maxima = pl.reshape(pl.row_max(pl.reshape(compact_max, [STRIPES, REDUCTION_LANES // STRIPES])), [1, STRIPES])
            first_max = pl.read(maxima, [0, 0])
            for stripe, (maximum_state,) in pl.range(1, STRIPES, init_values=(first_max,)):
                candidate_max = pl.read(maxima, [0, stripe])
                if candidate_max > maximum_state:
                    next_maximum = candidate_max
                else:
                    next_maximum = maximum_state
                row_max = pl.yield_(next_maximum)
            maximum_zero = pl.full([1, STRIPES], dtype=pl.FP32, value=0.0)
            maximum_column = pl.reshape(pl.add(maximum_zero, row_max), [STRIPES, 1])
            exponentials = pl.reshape(pl.row_expand_expdif(parts, maximum_column), [1, cols])
            exponentials = pl.set_validshape(exponentials, 1, cols)
            sums = pl.reshape(pl.row_sum(pl.reshape(exponentials, [STRIPES, cols // STRIPES])), [1, STRIPES])
            first_sum = pl.read(sums, [0, 0])
            for stripe, (sum_state,) in pl.range(1, STRIPES, init_values=(first_sum,)):
                row_sum = pl.yield_(sum_state + pl.read(sums, [0, stripe]))
            label_value = pl.read(labels, [0, row_index])
            label_index = pl.cast(label_value, pl.INDEX)
            target = pl.read(values, [0, label_index])
            target_exponential = pl.read(exponentials, [0, label_index])
            corrected = pl.write(exponentials, [0, label_index], target_exponential - row_sum)
            gradient_scale = (1.0 / rows) / row_sum
            gradient = pl.mul(corrected, gradient_scale)
            gradient = pl.set_validshape(gradient, 1, cols)
            gradient_next = pl.assemble(gradient_state, gradient, [row_index, 0])
            log_zero = pl.full([1, LOSS_LANES], dtype=pl.FP32, value=0.0)
            log_values = pl.log(pl.add(log_zero, row_sum))
            target_values = pl.add(log_zero, target)
            centered_target = pl.sub(target_values, row_max)
            loss_values = pl.sub(log_values, centered_target)
            partial_next = partial_state
            loss_next = pl.add(loss_state, loss_values)
            accumulated_loss, row_losses, gradient_output = pl.yield_(loss_next, partial_next, gradient_next)
        final_loss = pl.read(accumulated_loss, [0, 0])
        slot_zero = pl.full([1, LOSS_SLOT_LANES], dtype=pl.FP32, value=0.0)
        slot = pl.add(slot_zero, final_loss)
        workspace_output = pl.assemble(row_losses, slot, [worker, 0])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="resident_mean", deps=[producer]):
        block = pl.slice(workspace_output, [WORKERS, 8], [0, 0])
        total_lanes = pl.col_sum(block)
        total = pl.read(total_lanes, [0, 0])
        pl.write(loss, [0, 0], total * (1.0 / rows))
    return loss, gradient_output



@pl.jit
def cross_entropy_fp32_exact(
    logits: pl.Tensor,
    labels: pl.Tensor,
    colidx: pl.Tensor,
    loss: pl.Out[pl.Tensor],
    dlogits: pl.Out[pl.Tensor],
):
    rows, cols = logits.shape
    workspace = pl.create_tensor([WORKERS, LOSS_SLOT_LANES], dtype=pl.FP32)
    with pl.spmd(WORKERS, name_hint="resident_rows", allow_early_resolve=True) as producer:
        pl.set_cache_policy(logits, pl.CachePolicy.BYPASS)
        worker = pl.tile.get_block_idx()
        loss_zero = pl.full([1, LOSS_LANES], dtype=pl.FP32, value=0.0)
        for row_index, (loss_state, partial_state, gradient_state) in pl.pipeline(
            worker, rows, WORKERS, stage=3, init_values=(loss_zero, workspace, dlogits)
        ):
            raw = pl.slice(logits, [1, cols], [row_index, 0])
            values = raw
            masked = values
            parts = pl.reshape(masked, [STRIPES, cols // STRIPES])
            compact_max = pl.col_max(pl.reshape(masked, [cols // REDUCTION_LANES, REDUCTION_LANES]))
            maxima = pl.reshape(pl.row_max(pl.reshape(compact_max, [STRIPES, REDUCTION_LANES // STRIPES])), [1, STRIPES])
            first_max = pl.read(maxima, [0, 0])
            for stripe, (maximum_state,) in pl.range(1, STRIPES, init_values=(first_max,)):
                candidate_max = pl.read(maxima, [0, stripe])
                if candidate_max > maximum_state:
                    next_maximum = candidate_max
                else:
                    next_maximum = maximum_state
                row_max = pl.yield_(next_maximum)
            maximum_zero = pl.full([1, STRIPES], dtype=pl.FP32, value=0.0)
            maximum_column = pl.reshape(pl.add(maximum_zero, row_max), [STRIPES, 1])
            exponentials = pl.reshape(pl.row_expand_expdif(parts, maximum_column), [1, cols])
            exponentials = pl.set_validshape(exponentials, 1, cols)
            sums = pl.reshape(pl.row_sum(pl.reshape(exponentials, [STRIPES, cols // STRIPES])), [1, STRIPES])
            first_sum = pl.read(sums, [0, 0])
            for stripe, (sum_state,) in pl.range(1, STRIPES, init_values=(first_sum,)):
                row_sum = pl.yield_(sum_state + pl.read(sums, [0, stripe]))
            label_value = pl.read(labels, [0, row_index])
            label_index = pl.cast(label_value, pl.INDEX)
            target = pl.read(values, [0, label_index])
            target_exponential = pl.read(exponentials, [0, label_index])
            corrected = pl.write(exponentials, [0, label_index], target_exponential - row_sum)
            gradient_scale = (1.0 / rows) / row_sum
            gradient = pl.mul(corrected, gradient_scale)
            gradient = pl.set_validshape(gradient, 1, cols)
            gradient_next = pl.assemble(gradient_state, gradient, [row_index, 0])
            log_zero = pl.full([1, LOSS_LANES], dtype=pl.FP32, value=0.0)
            log_values = pl.log(pl.add(log_zero, row_sum))
            target_values = pl.add(log_zero, target)
            centered_target = pl.sub(target_values, row_max)
            loss_values = pl.sub(log_values, centered_target)
            partial_next = partial_state
            loss_next = pl.add(loss_state, loss_values)
            accumulated_loss, row_losses, gradient_output = pl.yield_(loss_next, partial_next, gradient_next)
        final_loss = pl.read(accumulated_loss, [0, 0])
        slot_zero = pl.full([1, LOSS_SLOT_LANES], dtype=pl.FP32, value=0.0)
        slot = pl.add(slot_zero, final_loss)
        workspace_output = pl.assemble(row_losses, slot, [worker, 0])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="resident_mean", deps=[producer]):
        block = pl.slice(workspace_output, [WORKERS, 8], [0, 0])
        total_lanes = pl.col_sum(block)
        total = pl.read(total_lanes, [0, 0])
        pl.write(loss, [0, 0], total * (1.0 / rows))
    return loss, gradient_output


@pl.jit
def cross_entropy_bf16_exact_wide(
    logits: pl.Tensor,
    labels: pl.Tensor,
    colidx: pl.Tensor,
    loss: pl.Out[pl.Tensor],
    dlogits: pl.Out[pl.Tensor],
):
    rows, cols = logits.shape
    workspace = pl.create_tensor([WORKERS, LOSS_SLOT_LANES], dtype=pl.FP32)
    with pl.spmd(WORKERS, name_hint="resident_rows", allow_early_resolve=True) as producer:
        pl.set_cache_policy(logits, pl.CachePolicy.BYPASS)
        worker = pl.tile.get_block_idx()
        loss_zero = pl.full([1, LOSS_LANES], dtype=pl.FP32, value=0.0)
        for row_index, (loss_state, partial_state, gradient_state) in pl.pipeline(
            worker, rows, WORKERS, stage=3, init_values=(loss_zero, workspace, dlogits)
        ):
            raw = pl.slice(logits, [1, cols], [row_index, 0])
            values = pl.cast(raw, target_type=pl.FP32)
            masked = values
            parts = pl.reshape(masked, [STRIPES, cols // STRIPES])
            compact_max = pl.col_max(pl.reshape(masked, [cols // REDUCTION_LANES, REDUCTION_LANES]))
            maxima = pl.reshape(pl.row_max(pl.reshape(compact_max, [STRIPES, REDUCTION_LANES // STRIPES])), [1, STRIPES])
            first_max = pl.read(maxima, [0, 0])
            for stripe, (maximum_state,) in pl.range(1, STRIPES, init_values=(first_max,)):
                candidate_max = pl.read(maxima, [0, stripe])
                if candidate_max > maximum_state:
                    next_maximum = candidate_max
                else:
                    next_maximum = maximum_state
                row_max = pl.yield_(next_maximum)
            maximum_zero = pl.full([1, STRIPES], dtype=pl.FP32, value=0.0)
            maximum_column = pl.reshape(pl.add(maximum_zero, row_max), [STRIPES, 1])
            exponentials = pl.reshape(pl.row_expand_expdif(parts, maximum_column), [1, cols])
            exponentials = pl.fillpad(pl.set_validshape(exponentials, 1, cols))
            sums = pl.reshape(pl.row_sum(pl.reshape(exponentials, [STRIPES, cols // STRIPES])), [1, STRIPES])
            first_sum = pl.read(sums, [0, 0])
            for stripe, (sum_state,) in pl.range(1, STRIPES, init_values=(first_sum,)):
                row_sum = pl.yield_(sum_state + pl.read(sums, [0, stripe]))
            label_value = pl.read(labels, [0, row_index])
            label_index = pl.cast(label_value, pl.INDEX)
            target = pl.read(values, [0, label_index])
            target_exponential = pl.read(exponentials, [0, label_index])
            corrected = pl.write(exponentials, [0, label_index], target_exponential - row_sum)
            gradient_scale = (1.0 / rows) / row_sum
            gradient = pl.mul(corrected, gradient_scale)
            gradient = pl.set_validshape(gradient, 1, cols)
            gradient_next = pl.assemble(gradient_state, gradient, [row_index, 0])
            log_zero = pl.full([1, LOSS_LANES], dtype=pl.FP32, value=0.0)
            log_values = pl.log(pl.add(log_zero, row_sum))
            target_values = pl.add(log_zero, target)
            centered_target = pl.sub(target_values, row_max)
            loss_values = pl.sub(log_values, centered_target)
            partial_next = partial_state
            loss_next = pl.add(loss_state, loss_values)
            accumulated_loss, row_losses, gradient_output = pl.yield_(loss_next, partial_next, gradient_next)
        final_loss = pl.read(accumulated_loss, [0, 0])
        slot_zero = pl.full([1, LOSS_SLOT_LANES], dtype=pl.FP32, value=0.0)
        slot = pl.add(slot_zero, final_loss)
        workspace_output = pl.assemble(row_losses, slot, [worker, 0])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="resident_mean", deps=[producer]):
        block = pl.slice(workspace_output, [WORKERS, 8], [0, 0])
        total_lanes = pl.col_sum(block)
        total = pl.read(total_lanes, [0, 0])
        pl.write(loss, [0, 0], total * (1.0 / rows))
    return loss, gradient_output



@pl.jit
def cross_entropy_fp32_exact_wide(
    logits: pl.Tensor,
    labels: pl.Tensor,
    colidx: pl.Tensor,
    loss: pl.Out[pl.Tensor],
    dlogits: pl.Out[pl.Tensor],
):
    rows, cols = logits.shape
    workspace = pl.create_tensor([WORKERS, LOSS_SLOT_LANES], dtype=pl.FP32)
    with pl.spmd(WORKERS, name_hint="resident_rows", allow_early_resolve=True) as producer:
        pl.set_cache_policy(logits, pl.CachePolicy.BYPASS)
        worker = pl.tile.get_block_idx()
        loss_zero = pl.full([1, LOSS_LANES], dtype=pl.FP32, value=0.0)
        for row_index, (loss_state, partial_state, gradient_state) in pl.pipeline(
            worker, rows, WORKERS, stage=3, init_values=(loss_zero, workspace, dlogits)
        ):
            raw = pl.slice(logits, [1, cols], [row_index, 0])
            values = raw
            masked = values
            parts = pl.reshape(masked, [STRIPES, cols // STRIPES])
            compact_max = pl.col_max(pl.reshape(masked, [cols // REDUCTION_LANES, REDUCTION_LANES]))
            maxima = pl.reshape(pl.row_max(pl.reshape(compact_max, [STRIPES, REDUCTION_LANES // STRIPES])), [1, STRIPES])
            first_max = pl.read(maxima, [0, 0])
            for stripe, (maximum_state,) in pl.range(1, STRIPES, init_values=(first_max,)):
                candidate_max = pl.read(maxima, [0, stripe])
                if candidate_max > maximum_state:
                    next_maximum = candidate_max
                else:
                    next_maximum = maximum_state
                row_max = pl.yield_(next_maximum)
            maximum_zero = pl.full([1, STRIPES], dtype=pl.FP32, value=0.0)
            maximum_column = pl.reshape(pl.add(maximum_zero, row_max), [STRIPES, 1])
            exponentials = pl.reshape(pl.row_expand_expdif(parts, maximum_column), [1, cols])
            exponentials = pl.fillpad(pl.set_validshape(exponentials, 1, cols))
            sums = pl.reshape(pl.row_sum(pl.reshape(exponentials, [STRIPES, cols // STRIPES])), [1, STRIPES])
            first_sum = pl.read(sums, [0, 0])
            for stripe, (sum_state,) in pl.range(1, STRIPES, init_values=(first_sum,)):
                row_sum = pl.yield_(sum_state + pl.read(sums, [0, stripe]))
            label_value = pl.read(labels, [0, row_index])
            label_index = pl.cast(label_value, pl.INDEX)
            target = pl.read(values, [0, label_index])
            target_exponential = pl.read(exponentials, [0, label_index])
            corrected = pl.write(exponentials, [0, label_index], target_exponential - row_sum)
            gradient_scale = (1.0 / rows) / row_sum
            gradient = pl.mul(corrected, gradient_scale)
            gradient = pl.set_validshape(gradient, 1, cols)
            gradient_next = pl.assemble(gradient_state, gradient, [row_index, 0])
            log_zero = pl.full([1, LOSS_LANES], dtype=pl.FP32, value=0.0)
            log_values = pl.log(pl.add(log_zero, row_sum))
            target_values = pl.add(log_zero, target)
            centered_target = pl.sub(target_values, row_max)
            loss_values = pl.sub(log_values, centered_target)
            partial_next = partial_state
            loss_next = pl.add(loss_state, loss_values)
            accumulated_loss, row_losses, gradient_output = pl.yield_(loss_next, partial_next, gradient_next)
        final_loss = pl.read(accumulated_loss, [0, 0])
        slot_zero = pl.full([1, LOSS_SLOT_LANES], dtype=pl.FP32, value=0.0)
        slot = pl.add(slot_zero, final_loss)
        workspace_output = pl.assemble(row_losses, slot, [worker, 0])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="resident_mean", deps=[producer]):
        block = pl.slice(workspace_output, [WORKERS, 8], [0, 0])
        total_lanes = pl.col_sum(block)
        total = pl.read(total_lanes, [0, 0])
        pl.write(loss, [0, 0], total * (1.0 / rows))
    return loss, gradient_output


def build(shape):
    import torch
    from golden import TensorSpec

    rows, cols = int(shape["M"]), int(shape["N"])
    if rows < 1 or cols < 1:
        raise ValueError("M and N must be positive")
    specs = [
        TensorSpec("logits", [rows, cols], torch.bfloat16 if shape["dtype"] == "bf16" else torch.float32, init_value=torch.randn),
        TensorSpec("labels", [1, rows], torch.int32,
                   init_value=lambda: torch.randint(0, cols, (1, rows), dtype=torch.int32)),
        TensorSpec("colidx", [1, cols], torch.float32,
                   init_value=lambda: torch.arange(cols, dtype=torch.float32).reshape(1, cols)),
        TensorSpec("loss", [1, 1], torch.float32),
        TensorSpec("dlogits", [rows, cols], torch.float32),
    ]

    def golden_fn(tensors):
        logits = tensors["logits"].float()
        labels = tensors["labels"].long().reshape(-1)
        tensors["loss"][:] = torch.nn.functional.cross_entropy(logits, labels, reduction="mean")
        gradient = torch.softmax(logits, dim=-1)
        gradient.scatter_add_(1, labels.unsqueeze(1), -torch.ones((rows, 1), dtype=torch.float32))
        tensors["dlogits"][:] = gradient / rows

    fn = cross_entropy_one_class if cols == 1 else (cross_entropy_bf16 if shape["dtype"] == "bf16" else cross_entropy_fp32)
    exact_extent = cols % REDUCTION_LANES == 0 and cols % STRIPES == 0 and (cols // STRIPES) % 8 == 0
    if cols > 1 and exact_extent:
        fn = cross_entropy_bf16_exact if shape["dtype"] == "bf16" else cross_entropy_fp32_exact
        if cols >= 6 * REDUCTION_LANES:
            fn = cross_entropy_bf16_exact_wide if shape["dtype"] == "bf16" else cross_entropy_fp32_exact_wide
    return fn, specs, golden_fn, 1e-4, min(1e-8, 1e-3 / (rows * cols))

if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    args = parser.parse_args()

    fn, specs, golden_fn, rtol, atol = build(DEFAULT_SHAPE)
    result = run(
        fn=fn,
        specs=specs,
        golden_fn=golden_fn,
        config=dict(
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=rtol,
        atol=atol,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
