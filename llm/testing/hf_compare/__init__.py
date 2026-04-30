# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Hugging Face comparison framework.

This package provides a pluggable harness for verifying that PyPTO kernels /
scopes / end-to-end models produce numerically equivalent outputs to a
reference implementation (typically a Hugging Face module on CPU).

Top-level entry points:

    from llm.testing.hf_compare import (
        ComparisonCase, register_case, get_case, list_cases,
    )

Cases live under ``llm/testing/hf_compare/cases/`` and are registered with
``@register_case("model.scope")``. The CLI in ``__main__.py`` discovers them
by name.
"""
from .base import (
    ChainStep,
    ChainedComparisonCase,
    CompareReport,
    ComparisonCase,
    InputSpec,
    OutputSelector,
    ReferenceModel,
    SelectorResult,
    TargetModel,
    TensorSpec,
    Tolerance,
    WeightAdapter,
    get_case,
    list_cases,
    register_case,
)

__all__ = [
    "ChainStep",
    "ChainedComparisonCase",
    "CompareReport",
    "ComparisonCase",
    "InputSpec",
    "OutputSelector",
    "ReferenceModel",
    "SelectorResult",
    "TargetModel",
    "TensorSpec",
    "Tolerance",
    "WeightAdapter",
    "get_case",
    "list_cases",
    "register_case",
]
