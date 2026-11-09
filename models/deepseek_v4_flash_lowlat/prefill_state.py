# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Retain prefill target taps and initialize the batch-one decode state."""

import torch

from .prefill_metadata import prefill_tap_positions


CP_SIZE = 4
RANKS = 8
HISTORY_ROWS = 128
HISTORY_ALLOC_ROWS = 136
FEATURE_WIDTH = 12288
VOCAB_SIZE = 129280


def update_prefill_tail(history, history_positions, features, positions):
    """Store one CP group's actual prefill taps at absolute-position ring slots.

    ``features`` and ``positions`` are the four rank-local prefill outputs and
    their corresponding position IDs. Negative position IDs mark padding.
    """
    if tuple(history.shape) != (HISTORY_ROWS, FEATURE_WIDTH) or history.dtype != torch.bfloat16:
        raise ValueError("history must be a BF16 [128, 12288] tensor")
    if tuple(history_positions.shape) != (HISTORY_ROWS,) or history_positions.dtype != torch.int64:
        raise ValueError("history_positions must be an INT64 [128] tensor")
    if features.ndim != 3 or features.shape[0] != CP_SIZE or features.shape[2] != FEATURE_WIDTH:
        raise ValueError("features must have shape [4, local_tokens, 12288]")
    if features.dtype != torch.bfloat16 or tuple(positions.shape) != tuple(features.shape[:2]):
        raise ValueError("Prefill feature dtype or rank-local positions do not match")
    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError("Prefill positions must be integer values")
    if history.device != features.device or history_positions.device != positions.device or history.device != positions.device:
        raise ValueError("History, features, and positions must be on the same device")
    flat_positions = positions.reshape(-1).to(torch.int64)
    if bool((flat_positions < -1).any()):
        raise ValueError("Only -1 may mark padding")
    valid = flat_positions >= 0
    current = flat_positions[valid]
    if current.numel() != torch.unique(current).numel():
        raise ValueError("A prefill position appears in more than one CP shard")
    if current.numel() == 0:
        return
    order = torch.argsort(current)
    current = current[order]
    if current.numel() > 1 and not bool(torch.all(current[1:] == current[:-1] + 1)):
        raise ValueError("The prefill chunk must cover consecutive positions")
    previous = history_positions[history_positions >= 0]
    if previous.numel() and int(current[0]) != int(previous.max()) + 1:
        raise ValueError("The prefill chunk must follow the retained history")
    current = current[-HISTORY_ROWS:]
    selected = features.reshape(-1, FEATURE_WIDTH)[valid][order][-HISTORY_ROWS:]
    physical = current.remainder(HISTORY_ROWS)
    history.index_copy_(0, physical, selected)
    history_positions.index_copy_(0, physical, current)


def capture_prefill_outputs(worker, feature_root, sampled_root, history, history_positions, *,
                            start_pos, token_count, terminal):
    """Capture one completed prefill chunk from its actual CP4 device outputs."""
    positions = prefill_tap_positions(start_pos, token_count)
    local = positions.shape[1]
    if (tuple(feature_root.worker_ids) != tuple(range(RANKS))
            or tuple(feature_root.shape) != (RANKS, local, FEATURE_WIDTH)
            or feature_root.dtype != torch.bfloat16):
        raise ValueError("The prefill feature root does not match the CP4 output")
    if (tuple(sampled_root.worker_ids) != tuple(range(RANKS)) or sampled_root.dtype != torch.int32
            or len(sampled_root.shape) != 3 or sampled_root.shape[0] != RANKS or sampled_root.shape[2] != 8):
        raise ValueError("The sampled-id root does not match the prefill output")
    if type(terminal) is not bool:
        raise TypeError("terminal must be a bool")
    features = torch.empty((CP_SIZE, local, FEATURE_WIDTH), dtype=torch.bfloat16)
    for rank in range(CP_SIZE):
        output = features[rank]
        worker.copy_from(output.data_ptr(), feature_root.shards[rank].data_ptr,
                         output.numel() * output.element_size(), worker_id=rank)
    update_prefill_tail(history, history_positions, features, positions)
    if not terminal:
        return None
    sampled = torch.empty((8,), dtype=torch.int32)
    worker.copy_from(sampled.data_ptr(), sampled_root.shards[0].data_ptr,
                     sampled.numel() * sampled.element_size(), worker_id=0)
    token = int(sampled[0])
    if not 0 <= token < VOCAB_SIZE:
        raise ValueError("The prefill producer returned an invalid sampled token")
    return token
