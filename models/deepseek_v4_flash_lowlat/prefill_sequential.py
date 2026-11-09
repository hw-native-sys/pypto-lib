# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cold-prompt prefill using the already resident batch-one decode weights."""

import torch

from .prefill_bootstrap import prepare_bootstrap
from .prefill_fwd import prefill_arguments
from .prefill_state import FEATURE_WIDTH, HISTORY_ROWS, RANKS, VOCAB_SIZE


_STEP_INPUTS = {
    "position_ids": ((RANKS, 1), torch.int32),
    "kv_seq_lens": ((RANKS, 1), torch.int32),
    "input_ids": ((RANKS, 1), torch.int64),
    "sampling_positions": ((RANKS, 1), torch.int32),
    "logit_row_indices": ((RANKS, 1), torch.int32),
    "num_tokens_per_owner": ((RANKS,), torch.int32),
}


def prefill_with_single(worker, program, metadata, bindings, tokens, *, max_new_tokens, epoch, config=None):
    """Prefill a cold B1 request with the public single-token target program.

    The caller registers this program and the normal speculative decode
    programs on the same Worker and supplies their shared owning weights,
    initialized caches, page tables, scratch, and pre-fork shared HOST inputs.
    No EP-layout weight copy is needed. This fallback performs one target
    forward per prompt token; it trades prompt latency for weight residency.

    Return ``(first_token, history, state)`` after all submissions finish. The
    caller can publish the returned history and state with
    :func:`prefill_bootstrap.publish_bootstrap` before speculative decode.
    """
    if (not isinstance(tokens, torch.Tensor) or tokens.device.type != "cpu" or tokens.ndim != 1
            or tokens.dtype != torch.int64 or tokens.numel() == 0):
        raise ValueError("tokens must be a nonempty one-dimensional CPU INT64 tensor")
    if tokens.numel() > 16384 or bool(((tokens < 0) | (tokens >= VOCAB_SIZE)).any()):
        raise ValueError("The prompt exceeds the compiled sequence or vocabulary range")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 2048:
        raise ValueError("max_new_tokens must be in [1, 2048]")
    if type(epoch) is not int or not 1 <= epoch < 2**31:
        raise ValueError("epoch must be a positive INT32 communication epoch")
    arguments = prefill_arguments(metadata, bindings)
    for name, (shape, dtype) in _STEP_INPUTS.items():
        value = bindings[name]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != dtype
                or value.device.type != "cpu" or not value.is_shared()):
            raise ValueError(f"{name} must be a pre-fork shared CPU tensor with shape {shape} and dtype {dtype}")
    features = bindings["dspark_target_hidden_out"]
    samples = bindings["sampled_ids"]
    if (tuple(features.shape) != (RANKS, 1, FEATURE_WIDTH) or features.dtype != torch.bfloat16
            or tuple(features.worker_ids) != tuple(range(RANKS))):
        raise ValueError("The single-token target feature output has an unexpected device layout")
    if (tuple(samples.shape) != (RANKS, 1, 8) or samples.dtype != torch.int32
            or tuple(samples.worker_ids) != tuple(range(RANKS))):
        raise ValueError("The single-token sampled-id output has an unexpected device layout")

    history = torch.zeros((HISTORY_ROWS, FEATURE_WIDTH), dtype=torch.bfloat16)
    positions = torch.full((HISTORY_ROWS,), -1, dtype=torch.int64)
    keep_from = max(0, tokens.numel() - HISTORY_ROWS)
    bindings["num_tokens_per_owner"].fill_(1)
    bindings["logit_row_indices"].zero_()
    for position, token in enumerate(tokens.tolist()):
        bindings["position_ids"].fill_(position)
        bindings["kv_seq_lens"].fill_(position + 1)
        bindings["input_ids"].fill_(token)
        bindings["sampling_positions"].fill_(position)
        worker.submit(program, *arguments, config=config).result()
        if position >= keep_from:
            row = position % HISTORY_ROWS
            worker.copy_from(history[row].data_ptr(), features.shards[0].data_ptr,
                             FEATURE_WIDTH * history.element_size(), worker_id=0)
            positions[row] = position
    sampled = torch.empty(1, dtype=torch.int32)
    worker.copy_from(sampled.data_ptr(), samples.shards[0].data_ptr, sampled.element_size(), worker_id=0)
    first = int(sampled[0])
    ring, state = prepare_bootstrap(history, positions, prompt_length=tokens.numel(), first_token=first,
                                    max_new_tokens=max_new_tokens, epoch=epoch)
    return first, ring, state
