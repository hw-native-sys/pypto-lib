# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2 Engram host-table staging contract.

The V4.1 Engram table is stored as E4M3 rows with E8M0 scales.  A2 runs the
PyPTO lookup and projection in BF16, so the host path decodes each requested
row once and stages the resulting BF16 tile to the device.  This module keeps
the physical resident range separate from the declared production row count:
an audit fixture containing rows ``0:4096`` must never be mistaken for the
384,006,168-row table in the checkpoint metadata.

The class is intentionally usable with a complete raw table when one is
available, but it does not allocate the complete table.  ``read_rows`` uses a
memory-mapped byte tensor and decodes only the requested rows.  The local
fixture has no complete file, so an out-of-window request raises an explicit
``LookupOutsideResidentWindow`` exception from ``engram_hash``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch

from .engram_hash import LookupOutsideResidentWindow, build_compressed_token_map


@dataclass(frozen=True)
class EngramTableGeometry:
    """Physical and declared dimensions of one Engram table."""

    full_rows: int
    resident_rows: int
    row_width: int = 256
    scale_width: int = 8
    scale_block: int = 32

    def __post_init__(self) -> None:
        if self.full_rows <= 0 or self.resident_rows <= 0:
            raise ValueError("Engram table dimensions must be positive")
        if self.resident_rows > self.full_rows:
            raise ValueError("resident table cannot exceed declared full table")
        if self.row_width != self.scale_width * self.scale_block:
            raise ValueError("scale geometry does not cover one row")


class A2EngramHostTable:
    """Decode and stage a bounded E4M3/E8M0 Engram table window.

    ``weight_path`` and ``scale_path`` contain raw bytes, one row after
    another.  The byte tensors are memory mapped by ``torch.from_file`` and
    therefore do not duplicate the resident table in host memory.  A complete
    production table can use the same API if its files are supplied; callers
    still receive only requested rows.
    """

    def __init__(
        self,
        weight_path: str | Path,
        scale_path: str | Path,
        *,
        full_rows: int,
        row_width: int = 256,
        scale_width: int = 8,
        scale_block: int = 32,
    ) -> None:
        self.weight_path = Path(weight_path)
        self.scale_path = Path(scale_path)
        if not self.weight_path.is_file() or not self.scale_path.is_file():
            raise FileNotFoundError("Engram raw weight or scale file is missing")
        if row_width != scale_width * scale_block:
            raise ValueError("scale geometry does not cover one row")
        weight_bytes = self.weight_path.stat().st_size
        scale_bytes = self.scale_path.stat().st_size
        if weight_bytes % row_width:
            raise ValueError("raw E4M3 file is not an integral number of rows")
        if scale_bytes % scale_width:
            raise ValueError("raw E8M0 file is not an integral number of rows")
        resident_rows = weight_bytes // row_width
        scale_rows = scale_bytes // scale_width
        if resident_rows != scale_rows:
            raise ValueError("weight and scale files have different row counts")
        self.geometry = EngramTableGeometry(
            full_rows=int(full_rows),
            resident_rows=int(resident_rows),
            row_width=int(row_width),
            scale_width=int(scale_width),
            scale_block=int(scale_block),
        )
        # ``size`` is required because torch.from_file defaults to an empty
        # mapping when the size is omitted.
        self._weight_bytes = torch.from_file(
            str(self.weight_path), dtype=torch.uint8, size=weight_bytes
        ).reshape(self.geometry.resident_rows, self.geometry.row_width)
        self._scale_bytes = torch.from_file(
            str(self.scale_path), dtype=torch.uint8, size=scale_bytes
        ).reshape(self.geometry.resident_rows, self.geometry.scale_width)

    @property
    def full_rows(self) -> int:
        return self.geometry.full_rows

    @property
    def resident_rows(self) -> int:
        return self.geometry.resident_rows

    def coverage(self) -> dict[str, int | str]:
        """Return an audit-friendly declaration of the available range."""

        return {
            "full_rows": self.geometry.full_rows,
            "resident_rows": self.geometry.resident_rows,
            "resident_start": 0,
            "resident_end_exclusive": self.geometry.resident_rows,
            "row_width": self.geometry.row_width,
            "scale_width": self.geometry.scale_width,
            "scale_block": self.geometry.scale_block,
            "coverage": (
                "full"
                if self.geometry.resident_rows == self.geometry.full_rows
                else "bounded_resident_window"
            ),
        }

    def _validate_ids(self, row_ids: torch.Tensor) -> torch.Tensor:
        ids = torch.as_tensor(row_ids, dtype=torch.int64).reshape(-1)
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= self.full_rows):
            raise ValueError(
                f"row id outside declared table range [0, {self.full_rows})"
            )
        if ids.numel() and int(ids.max()) >= self.resident_rows:
            raise LookupOutsideResidentWindow(
                int(ids.max()), self.resident_rows, self.full_rows
            )
        return ids

    def read_rows(self, row_ids: torch.Tensor) -> torch.Tensor:
        """Decode requested rows to BF16 while preserving row-id order."""

        original = torch.as_tensor(row_ids, dtype=torch.int64)
        ids = self._validate_ids(original)
        if ids.numel() == 0:
            return torch.empty((*original.shape, self.geometry.row_width), dtype=torch.bfloat16)
        raw = self._weight_bytes.index_select(0, ids)
        raw_fp32 = raw.view(torch.float8_e4m3fn).float()
        scale_raw = self._scale_bytes.index_select(0, ids)
        # vLLM's A2 path materializes E8M0 as an FP32 exponent bit pattern.
        # Byte 0 consequently maps to +0.0, matching the native bitcast.
        scale_fp32 = (scale_raw.to(torch.int32) << 23).view(torch.float32)
        decoded = raw_fp32.reshape(-1, self.geometry.scale_width, self.geometry.scale_block)
        decoded = (decoded * scale_fp32[:, :, None]).reshape(-1, self.geometry.row_width)
        return decoded.to(torch.bfloat16).reshape(*original.shape, self.geometry.row_width)

    def iter_rows(
        self, row_ids: Sequence[int] | torch.Tensor, *, chunk_rows: int = 32
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield ``(ids, decoded_rows)`` in bounded chunks."""

        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        ids = torch.as_tensor(row_ids, dtype=torch.int64).reshape(-1)
        for start in range(0, ids.numel(), chunk_rows):
            chunk = ids[start : start + chunk_rows]
            yield chunk, self.read_rows(chunk)

    def stage_rows(self, row_ids: torch.Tensor, device: torch.device | str) -> torch.Tensor:
        """Decode on host and copy rows to an A2 device, when available."""

        rows = self.read_rows(row_ids)
        staged = rows.to(device)
        if getattr(staged.device, "type", None) == "npu":
            torch.npu.synchronize()
        return staged


def token_map_from_tokenizer(tokenizer, *, expected_size: int = 99092) -> torch.Tensor:
    """Build the native normalized token map and enforce its hash contract."""

    values, compressed_size = build_compressed_token_map(tokenizer)
    if compressed_size != expected_size:
        raise ValueError(
            f"tokenizer compressed vocab mismatch: {compressed_size} != {expected_size}"
        )
    return torch.tensor(values, dtype=torch.int32)


__all__ = [
    "A2EngramHostTable",
    "EngramTableGeometry",
    "LookupOutsideResidentWindow",
    "token_map_from_tokenizer",
]
