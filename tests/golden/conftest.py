# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pytest conftest for golden tests — adds repo root to sys.path.

The stub pypto these tests fall back on when the real one is not installed
lives in tests/conftest.py, which more than one subdirectory shares.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))




# Bench knobs golden.runner reads from the environment. An exported PYPTO_BENCH
# sends every run() down the benchmark path — which the stub pypto cannot serve —
# and fails a couple of dozen unrelated tests, so clear them for the whole suite;
# the tests that exercise these knobs set them explicitly via monkeypatch.
_BENCH_ENV = ("PYPTO_BENCH", "PYPTO_BENCH_RAW", "PYPTO_BENCH_ROUNDS", "PYPTO_BENCH_WARMUP")


@pytest.fixture(autouse=True)
def _isolate_bench_env(monkeypatch):
    for name in _BENCH_ENV:
        monkeypatch.delenv(name, raising=False)
