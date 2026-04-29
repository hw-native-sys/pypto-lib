# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Specifications for runtime inputs of the golden testing harness.

Two kinds of specifications are supported:

- :class:`TensorSpec` — describes a ``torch.Tensor`` argument (shape, dtype,
  initialization strategy, output flag).
- :class:`ScalarSpec` — describes a scalar argument matching a ``pl.Scalar``
  parameter on the orchestration function.  Encodes Python values into the
  ``ctypes._SimpleCData`` form expected by ``pypto.runtime.execute_compiled``.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class TensorSpec:
    """Specification for a runtime tensor.

    Attributes:
        name: Tensor name, matching the orchestration function's parameter name.
        shape: Tensor shape as a list of integers.
        dtype: PyTorch dtype (e.g. ``torch.float32``, ``torch.bfloat16``).
        init_value: Initial value strategy.  Can be:

            - ``None`` — zero-initialised.
            - ``int`` or ``float`` — every element set to this constant.
            - ``torch.Tensor`` — use this tensor directly (must have matching shape/dtype).
            - ``Callable`` — a no-argument callable that returns a ``torch.Tensor``, or
              one of the supported ``torch`` factory functions
              (``torch.randn``, ``torch.rand``, ``torch.zeros``, ``torch.ones``)
              that will be called with ``(shape, dtype=dtype)``.
        is_output: If ``True``, the tensor is an output to be validated against the
            golden reference.

    Example:
        >>> import torch
        >>> TensorSpec("query", [32, 128], torch.bfloat16, init_value=torch.randn)
        >>> TensorSpec("out", [32, 128], torch.float32, is_output=True)
    """

    name: str
    shape: list[int]
    dtype: torch.dtype
    init_value: int | float | torch.Tensor | Callable | None = field(default=None)
    is_output: bool = False

    def create_tensor(self) -> torch.Tensor:
        """Create and return a ``torch.Tensor`` based on this specification.

        Returns:
            Initialised tensor with the requested shape and dtype.
        """
        if self.init_value is None:
            return torch.zeros(self.shape, dtype=self.dtype)
        if isinstance(self.init_value, (int, float)):
            return torch.full(self.shape, self.init_value, dtype=self.dtype)
        if isinstance(self.init_value, torch.Tensor):
            return self.init_value.to(dtype=self.dtype)
        if callable(self.init_value):
            # Support the standard torch factory functions used as callables
            fn = self.init_value
            if fn in (torch.randn, torch.rand, torch.zeros, torch.ones):
                return fn(self.shape, dtype=self.dtype)
            # Generic callable: call with no arguments, then cast
            result: Any = fn()
            return torch.as_tensor(result, dtype=self.dtype)
        raise TypeError(f"Unsupported init_value type {type(self.init_value)!r} for tensor {self.name!r}")


# ---------------------------------------------------------------------------
# ScalarSpec
# ---------------------------------------------------------------------------

# Per-int torch.dtype: (ctypes type, lo, hi).  fp16/bf16 are handled separately
# since Python ctypes has no native half-precision type.
_INT_TABLE: dict[torch.dtype, tuple[type, int, int]] = {
    torch.int8:  (ctypes.c_int8,  -(1 << 7),  (1 << 7) - 1),
    torch.int32: (ctypes.c_int32, -(1 << 31), (1 << 31) - 1),
    torch.int64: (ctypes.c_int64, -(1 << 63), (1 << 63) - 1),
    torch.uint8: (ctypes.c_uint8, 0,          (1 << 8) - 1),
}

_HALF_DTYPES: tuple[torch.dtype, ...] = (torch.float16, torch.bfloat16)

SUPPORTED_SCALAR_DTYPES: tuple[torch.dtype, ...] = (
    *_INT_TABLE.keys(),
    torch.bool,
    torch.float32,
    *_HALF_DTYPES,
)


def _validate_primitive(name: str, dtype: torch.dtype, value: Any) -> None:
    """Type/range validation for a python-primitive *value* against *dtype*.

    Called from :meth:`ScalarSpec.__post_init__` when the user passes an int /
    float / bool; for ``torch.Tensor`` inputs the dtype is already canonical
    and only ndim/dtype consistency is checked.
    """
    if dtype in _INT_TABLE:
        # bool is an int subclass — reject explicitly so dtype is unambiguous
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"ScalarSpec {name!r} dtype={dtype} requires int value, got {type(value).__name__}"
            )
        _, lo, hi = _INT_TABLE[dtype]
        if not lo <= value <= hi:
            raise ValueError(
                f"ScalarSpec {name!r} value {value} out of range for {dtype} [{lo}, {hi}]"
            )
        return
    if dtype is torch.bool:
        if not isinstance(value, bool):
            raise ValueError(
                f"ScalarSpec {name!r} dtype=torch.bool requires bool value, got {type(value).__name__}"
            )
        return
    if dtype is torch.float32 or dtype in _HALF_DTYPES:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"ScalarSpec {name!r} dtype={dtype} requires int or float value, got {type(value).__name__}"
            )
        return
    raise ValueError(
        f"ScalarSpec {name!r}: unsupported dtype {dtype!r}; "
        f"expected one of {list(SUPPORTED_SCALAR_DTYPES)}"
    )


@dataclass
class ScalarSpec:
    """Specification for a scalar runtime argument.

    Attributes:
        name: Scalar name, matching the orchestration function's ``pl.Scalar``
            parameter name.
        dtype: PyTorch dtype, mirroring :class:`TensorSpec`.  Must be one of
            :data:`SUPPORTED_SCALAR_DTYPES`: ``torch.int8``, ``torch.int32``,
            ``torch.int64``, ``torch.uint8``, ``torch.bool``, ``torch.float32``,
            ``torch.float16``, ``torch.bfloat16``.
        value: Constructor accepts ``int`` / ``float`` / ``bool`` (validated
            and coerced) or a 0-dim ``torch.Tensor`` whose dtype matches
            *dtype* (used directly).  After ``__post_init__`` runs, ``value``
            is **always** a 0-dim ``torch.Tensor`` carrying the dtype-precise
            representation, so cache I/O is just ``torch.save`` / ``torch.load``.

    Example:
        >>> ScalarSpec("alpha", torch.float32, 0.125)
        >>> ScalarSpec("seq_len", torch.int32, 4096)
    """

    name: str
    dtype: torch.dtype
    value: int | float | bool | torch.Tensor

    def __post_init__(self) -> None:
        if self.dtype not in SUPPORTED_SCALAR_DTYPES:
            raise ValueError(
                f"ScalarSpec {self.name!r}: unsupported dtype {self.dtype!r}; "
                f"expected one of {list(SUPPORTED_SCALAR_DTYPES)}"
            )
        if isinstance(self.value, torch.Tensor):
            if self.value.ndim != 0:
                raise ValueError(
                    f"ScalarSpec {self.name!r}: value tensor must be 0-dim, "
                    f"got shape {tuple(self.value.shape)}"
                )
            if self.value.dtype != self.dtype:
                raise ValueError(
                    f"ScalarSpec {self.name!r}: value tensor dtype {self.value.dtype} "
                    f"does not match dtype field {self.dtype}"
                )
            return
        _validate_primitive(self.name, self.dtype, self.value)
        # Coerce to dtype-precise 0-dim tensor so storage matches dtype.
        self.value = torch.tensor(self.value, dtype=self.dtype)

    def to_ctypes(self) -> ctypes._SimpleCData:
        """Encode to ``ctypes._SimpleCData`` for :func:`pypto.runtime.execute_compiled`.

        Half-precision values are wrapped in ``c_uint16`` carrying the
        bit pattern (Python ctypes has no native half).
        """
        if self.dtype in _INT_TABLE:
            return _INT_TABLE[self.dtype][0](int(self.value.item()))
        if self.dtype is torch.bool:
            return ctypes.c_bool(bool(self.value.item()))
        if self.dtype is torch.float32:
            return ctypes.c_float(float(self.value.item()))
        # fp16 / bf16: read raw 16-bit pattern.
        bits = self.value.view(torch.int16).item()
        return ctypes.c_uint16(int(bits) & 0xFFFF)

    def to_python(self) -> int | float | bool:
        """Return a Python value for ``golden_fn``, matched to device precision.

        Because :attr:`value` is already a dtype-precise 0-dim tensor, the
        round-trip is just ``.item()``.
        """
        return self.value.item()
