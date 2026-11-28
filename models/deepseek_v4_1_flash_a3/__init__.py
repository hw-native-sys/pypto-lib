# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4.1 Flash A3 Prefill and Decode product operators."""

from importlib import import_module


_EXPORTS = {
    "l2_prefill_fwd": ("prefill_fwd", "l2_prefill_fwd"),
    "l3_prefill_fwd": ("prefill_fwd", "l3_prefill_fwd"),
    "l2_decode_fwd": ("decode_fwd", "l2_decode_fwd"),
    "l3_decode_fwd": ("decode_fwd", "l3_decode_fwd"),
}


__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Load a public operator only when a caller requests it."""

    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return eager package names and lazy public operator names."""

    return sorted(set(globals()) | set(__all__))
