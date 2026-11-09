# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Initialize the batch-one decode state from completed prefill."""

import torch

from .prefill_state import FEATURE_WIDTH, HISTORY_ALLOC_ROWS, HISTORY_ROWS, RANKS, VOCAB_SIZE


def prepare_bootstrap(history, history_positions, *, prompt_length, first_token, max_new_tokens, epoch):
    """Return owner-zero history and replicated state for the first D/M cycle."""
    if tuple(history.shape) != (HISTORY_ROWS, FEATURE_WIDTH) or history.dtype != torch.bfloat16:
        raise ValueError("history must be a BF16 [128, 12288] tensor")
    if tuple(history_positions.shape) != (HISTORY_ROWS,) or history_positions.dtype != torch.int64:
        raise ValueError("history_positions must be an INT64 [128] tensor")
    if history.device.type != "cpu" or history_positions.device.type != "cpu":
        raise ValueError("The host bootstrap requires CPU history tensors")
    if type(prompt_length) is not int or prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    if type(first_token) is not int or not 0 <= first_token < VOCAB_SIZE:
        raise ValueError("first_token must be an actual vocabulary ID")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 2048:
        raise ValueError("max_new_tokens must be in [1, 2048]")
    if type(epoch) is not int or not 1 <= epoch < 2**31:
        raise ValueError("epoch must be a positive INT32 communication epoch")
    first_position = max(0, prompt_length - HISTORY_ROWS)
    required = torch.arange(first_position, prompt_length, dtype=torch.int64)
    if not torch.equal(history_positions[required.remainder(HISTORY_ROWS)], required):
        raise ValueError("The last prompt positions are not present in the current prefill tail")
    ring = torch.zeros((HISTORY_ALLOC_ROWS, FEATURE_WIDTH), dtype=torch.bfloat16)
    ring[:HISTORY_ROWS].copy_(history)
    state = torch.tensor([epoch, prompt_length - 1, max_new_tokens - 1, first_token], dtype=torch.int64)
    return ring, state.unsqueeze(0).expand(RANKS, -1).contiguous()


def publish_bootstrap(worker, history_root, state_root, history, state):
    """Upload the completed owner-zero history and the same request state to all ranks.

    The worker must be idle and both roots must belong to it. If a copy fails,
    the caller must discard this request state instead of starting decode.
    """
    if tuple(history.shape) != (HISTORY_ALLOC_ROWS, FEATURE_WIDTH) or history.dtype != torch.bfloat16:
        raise ValueError("history must be a BF16 [136, 12288] tensor")
    if tuple(state.shape) != (RANKS, 4) or state.dtype != torch.int64:
        raise ValueError("state must be an INT64 [8, 4] tensor")
    if history.device.type != "cpu" or state.device.type != "cpu":
        raise ValueError("Bootstrap transfers require CPU tensors")
    if tuple(history_root.worker_ids) != tuple(range(RANKS)) or tuple(state_root.worker_ids) != tuple(range(RANKS)):
        raise ValueError("Bootstrap roots must use the eight target workers")
    if tuple(history_root.shape) != (RANKS, HISTORY_ALLOC_ROWS, FEATURE_WIDTH) or history_root.dtype != history.dtype:
        raise ValueError("The history root does not match native decode")
    if tuple(state_root.shape) != (RANKS, 4) or state_root.dtype != state.dtype:
        raise ValueError("The request-state root does not match native decode")
    history = history.contiguous()
    state = state.contiguous()
    worker.copy_to(history_root.shards[0].data_ptr, history.data_ptr(), history.numel() * history.element_size(), worker_id=0)
    for rank, shard in enumerate(state_root.shards):
        worker.copy_to(shard.data_ptr, state[rank].data_ptr(), 4 * state.element_size(), worker_id=rank)
