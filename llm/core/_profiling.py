# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import time


class StageTimer:
    """Lightweight stage-timing helper for L3 trace breakdowns.

    Use ``mark()`` after each labelled stage, then ``report()`` at the end.
    All operations are no-ops when ``enabled`` is False.
    """

    __slots__ = ("_enabled", "_prefix", "_title", "_t0", "_stages")

    def __init__(self, *, enabled: bool, prefix: str, title: str) -> None:
        self._enabled = enabled
        self._prefix = prefix
        self._title = title
        self._t0 = time.perf_counter() if enabled else 0.0
        self._stages: list[tuple[str, float]] = []

    def mark(self, label: str) -> None:
        if self._enabled:
            self._stages.append((label, time.perf_counter()))

    def report(self) -> None:
        if not self._enabled or not self._stages:
            return
        prev = self._t0
        total_ms = (self._stages[-1][1] - self._t0) * 1000.0
        print(f"[{self._prefix}] {self._title}:", flush=True)
        for label, t in self._stages:
            dt_ms = (t - prev) * 1000.0
            print(f"[{self._prefix}]   {label:30s} : {dt_ms:9.1f} ms", flush=True)
            prev = t
        print(f"[{self._prefix}]   {'TOTAL':30s} : {total_ms:9.1f} ms", flush=True)
