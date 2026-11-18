# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""B1 K5 streaming to 2048 total output tokens with explicit EOS successor drain.

The existing public submit/result/HOST capture and original packet validators
remain caller callbacks. No device state/copy/reset is performed in this policy.
"""

from collections import deque
from dataclasses import dataclass
from time import perf_counter_ns


@dataclass(frozen=True)
class Pending:
    index: int
    frame: int
    cycles: int
    handle: object


def choose_cycles(remaining):
    if type(remaining) is not int or not 1 <= remaining <= 2047:
        raise ValueError("Budget excludes the already counted prefill prediction; expected 1..2047")
    return 2 if remaining >= 7 else 1


def run_request(
    *,
    token_budget,
    eos_token_ids,
    submit,
    capture_and_validate,
    release_frame,
    deliver,
    discard_after_eos,
    events=None,
):
    """Return delivered IDs and raw-drain bookkeeping after every accepted handle.

    token_budget is max_new_tokens minus the single actual prefill prediction.
    Caller handles an EOS prefill prediction without invoking this controller.
    capture_and_validate persists actual raw packets and checks the original
    physical commit chain, including any accepted successor after EOS.
    discard_after_eos consumes a validated frame without delivering its tokens.
    On exceptions the caller drains/closes the Worker; no replacement is submitted.
    """
    if type(token_budget) is not int or not 0 <= token_budget <= 2047:
        raise ValueError("At most 2048 total generated tokens, including the prefill prediction")
    eos = set(eos_token_ids)
    if any(type(t) is not int or not 0 <= t < 129280 for t in eos):
        raise ValueError("Actual tokenizer EOS IDs required")
    trace = [] if events is None else events
    pending = deque()
    free = deque([0, 1])
    remaining = token_budget
    emitted = []
    raw_outputs = []
    discarded = []
    submitted = 0
    stopped = False

    def enqueue(cycles):
        nonlocal submitted
        if stopped or not free or len(pending) >= 2:
            raise RuntimeError("Invalid new parent or owned frame")
        frame = free.popleft()
        index = submitted
        trace.append(
            dict(event="submit_begin", index=index, frame=frame, cycles=cycles, time_ns=perf_counter_ns())
        )
        handle = submit(frame, cycles)
        pending.append(Pending(index, frame, cycles, handle))
        submitted += 1
        trace.append(dict(event="submit_return", index=index, frame=frame, time_ns=perf_counter_ns()))

    while pending or (remaining and not stopped):
        if not stopped:
            if not pending:
                enqueue(choose_cycles(remaining))
            if len(pending) == 1 and free and remaining >= 24 and pending[0].cycles == 2:
                enqueue(2)
        current = pending[0]
        current.handle.result()
        trace.append(
            dict(
                event="whole_parent_result",
                index=current.index,
                frame=current.frame,
                time_ns=perf_counter_ns(),
            )
        )
        tokens = capture_and_validate(current.frame, current.cycles)
        if not (
            isinstance(tokens, list) and current.cycles <= len(tokens) <= min(remaining, 6 * current.cycles)
        ):
            raise RuntimeError("Actual physical commit emission violates reserved K5 budget")
        if any(type(t) is not int or not 0 <= t < 129280 for t in tokens):
            raise RuntimeError("Invalid emitted token")
        remaining -= len(tokens)
        raw_outputs.extend(tokens)
        if stopped:
            discard_after_eos(tokens)
            discarded.extend(tokens)
            trace.append(
                dict(
                    event="drained_after_eos",
                    index=current.index,
                    frame=current.frame,
                    raw_count=len(tokens),
                    time_ns=perf_counter_ns(),
                )
            )
        else:
            cut = next((i + 1 for i, t in enumerate(tokens) if t in eos), len(tokens))
            visible = tokens[:cut]
            emitted.extend(visible)
            deliver(visible)
            stopped = any(t in eos for t in visible)
            discarded.extend(tokens[cut:])
            trace.append(
                dict(
                    event="delivered",
                    index=current.index,
                    frame=current.frame,
                    count=len(visible),
                    raw_count=len(tokens),
                    discarded_same_packet=len(tokens) - cut,
                    raw_remaining=remaining,
                    EOS=stopped,
                    time_ns=perf_counter_ns(),
                )
            )
        release_frame(current.frame)
        pending.popleft()
        free.append(current.frame)
        if remaining == 0 and pending:
            raise RuntimeError("Successor admitted beyond the real token budget")
    if not stopped and len(emitted) != token_budget:
        raise RuntimeError("Incomplete length-limited generation")
    return dict(
        output_ids=emitted,
        raw_output_ids=raw_outputs,
        discarded_after_eos=discarded,
        finish_reason="stop" if stopped else "length",
        raw_remaining=remaining,
        actual_parents=submitted,
        pending=0,
        output_count_excludes_prefill_prediction=True,
    )
