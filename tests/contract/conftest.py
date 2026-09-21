# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pytest conftest for contract tests — adds the repo root to sys.path.

The per-model contract tests import model sources (``models.*``) and the
``contract`` package as plain modules; ``models/`` has no ``__init__.py``, so
a standalone ``pytest tests/contract/...`` run needs the repo root on the
path (a combined run previously borrowed tests/golden/conftest.py's insert).

Nothing here needs a working pypto. ``models/qwen3_14b/contract.py`` is the
only contract module that imports ``pypto.language``, and it touches one
symbol at import time: the ``@pl.jit.host`` decorator on its three serving
wrappers — its ``from __future__ import annotations`` keeps every
``pl.Tensor`` / ``pl.Out[...]`` annotation an unevaluated string. When pypto
is absent, tests/conftest.py stands in a stub carrying that decorator. Tests
that need real kernel objects guard on ``HAS_PYPTO``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
