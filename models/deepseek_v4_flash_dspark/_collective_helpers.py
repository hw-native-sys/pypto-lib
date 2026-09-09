# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared InCore helpers for the hand-rolled split-phase collective idiom.

DSpark publish → wait → consume → complete sites had already drifted on wait
semantics (``defer_wait`` vs blocking ``wait``) and on how credits are retired.
These helpers make that choice explicit at the call site without forcing every
rail onto one answer. First consumers: prefill CP token allgather (deferred)
and decode CP allgather (blocking).

``n_peers`` must be a Python ``int`` closed over by the factories below so
``pl.range(n_peers)`` has a statically known trip count (required for deferred
waiter budget proofs). Call sites typically do::

    wait_deferred = make_wait_for_peer_credits_deferred(TP_SIZE)
    clear = make_clear_peer_credits(TP_SIZE)

Signal windows stay typed at each caller; helpers return the signal so
``@pl.jit.inline`` has a value to bind.
"""

from __future__ import annotations

from typing import Callable

import pypto.language as pl
import pypto.language.distributed as pld

_WaitFn = Callable[
    [pld.DistributedTensor, pl.Scalar[pl.INT32], pl.Scalar[pl.INT32]],
    pld.DistributedTensor,
]
_ClearFn = Callable[
    [
        pld.DistributedTensor,
        pl.Scalar[pl.INT32],
        pl.Scalar[pl.INT32],
        pl.Scalar[pl.INT32],
    ],
    pld.DistributedTensor,
]


def make_wait_for_peer_credits_deferred(n_peers: int) -> _WaitFn:
    """Build a deferred peer-credit waiter with static ``pl.range(n_peers)``."""
    if n_peers <= 0:
        raise ValueError(f"n_peers must be positive, got {n_peers}")

    @pl.jit.inline
    def wait_for_peer_credits_deferred(
        signal: pld.DistributedTensor,
        my_peer_idx: pl.Scalar[pl.INT32],
        expected: pl.Scalar[pl.INT32],
    ) -> pld.DistributedTensor:
        """Register per-peer ``defer_wait`` conditions (non-blocking completion).

        Used by prefill CP token all-gather. Leaves the physical core free while
        the counter catches up; pair with ``deps=[wait_tid]`` on consumers.
        """
        for source in pl.range(n_peers):
            if source != my_peer_idx:
                pld.system.defer_wait(
                    signal=signal,
                    offsets=[source, 0],
                    expected=expected,
                    cmp=pld.WaitCmp.Ge,
                )
        return signal

    return wait_for_peer_credits_deferred


def make_wait_for_peer_credits_blocking(n_peers: int) -> _WaitFn:
    """Build a blocking peer-credit waiter with static ``pl.range(n_peers)``."""
    if n_peers <= 0:
        raise ValueError(f"n_peers must be positive, got {n_peers}")

    @pl.jit.inline
    def wait_for_peer_credits_blocking(
        signal: pld.DistributedTensor,
        my_peer_idx: pl.Scalar[pl.INT32],
        expected: pl.Scalar[pl.INT32],
    ) -> pld.DistributedTensor:
        """Spin on per-peer ``wait`` until ``expected`` credits arrive (blocking).

        Used by decode CP all-gather. Same credit arithmetic as the deferred form;
        different scheduling. Prefer the deferred helper when overlap matters and
        the enclosing graph can express ``deps=`` edges.
        """
        for source in pl.range(n_peers):
            if source != my_peer_idx:
                pld.system.wait(
                    signal=signal,
                    offsets=[source, 0],
                    expected=expected,
                    cmp=pld.WaitCmp.Ge,
                )
        return signal

    return wait_for_peer_credits_blocking


def make_clear_peer_credits(n_peers: int) -> _ClearFn:
    """Build a self-clear helper with static ``pl.range(n_peers)``."""
    if n_peers <= 0:
        raise ValueError(f"n_peers must be positive, got {n_peers}")

    @pl.jit.inline
    def clear_peer_credits(
        signal: pld.DistributedTensor,
        my_peer_idx: pl.Scalar[pl.INT32],
        self_comm_rank: pl.Scalar[pl.INT32],
        reset_value: pl.Scalar[pl.INT32],
    ) -> pld.DistributedTensor:
        """Self-clear peer cells with ``AtomicAdd(reset_value)`` (typically ``-N``).

        ``self_comm_rank`` is the absolute notify peer (e.g. ``group_base + tp_rank``).
        ``my_peer_idx`` / ``source`` index the signal's peer axis (0..n_peers-1).
        """
        for source in pl.range(n_peers):
            if source != my_peer_idx:
                pld.system.notify(
                    target=signal,
                    peer=self_comm_rank,
                    offsets=[source, 0],
                    value=reset_value,
                    op=pld.NotifyOp.AtomicAdd,
                )
        return signal

    return clear_peer_credits
