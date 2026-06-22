# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Golden testing infrastructure for PyPTO-Lib.

Provides tensor and scalar specifications, result validation, and a runner
that compiles and executes PyPTO programs with golden reference comparison.
"""

from .runner import RunResult, run, run_jit
from .spec import ScalarSpec, TensorSpec
from .validation import (
    error_distribution,
    ratio_allclose,
    ratio_reldiff,
    topk_pair_compare,
    validate_golden,
)

__all__ = [
    "TensorSpec",
    "ScalarSpec",
    "validate_golden",
    "ratio_allclose",
    "ratio_reldiff",
    "error_distribution",
    "topk_pair_compare",
    "RunResult",
    "run",
    "run_jit",
]
