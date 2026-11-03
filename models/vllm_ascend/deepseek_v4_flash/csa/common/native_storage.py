# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Checked, zero-copy descriptors for Native cache allocations."""

import torch

from .layout import INDEXER_KEY_BYTES, INDEXER_MIN_PAGE_BYTES


def physical_pages(view: torch.Tensor) -> torch.Tensor:
    """Expose the complete padded page without changing its owner or offset."""
    if view.ndim != 4 or view.shape[2] != 1 or view.stride(-1) != 1:
        raise ValueError(f"Unsupported native cache view: {view.shape}, {view.stride()}")
    pages, tokens, _, width = view.shape
    stride = view.stride(0)
    if view.stride(1) != width or stride < tokens * width:
        raise ValueError("Native cache must have contiguous token rows within each physical page")
    end = view.storage_offset() + pages * stride
    if end * view.element_size() > view.untyped_storage().nbytes():
        raise ValueError("Native cache allocation does not contain the full final page")
    return view.as_strided((pages, stride), (stride, 1), view.storage_offset())


def indexer_storage(key: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Use one INT8 carrier; kernels reinterpret only the small scale region."""
    if key.dtype != torch.int8 or tuple(key.shape[1:]) != (32, 1, 128):
        raise ValueError("CSA requires Native INT8 indexer keys with 32 tokens per page")
    if scale.dtype != torch.float16 or tuple(scale.shape) != (key.shape[0], 32, 1, 1):
        raise ValueError("CSA requires one Native FP16 indexer scale per token")
    if key.untyped_storage().data_ptr() != scale.untyped_storage().data_ptr():
        raise ValueError("Indexer keys and scales must share their Native allocation")
    if scale.storage_offset() * 2 != key.storage_offset() + INDEXER_KEY_BYTES:
        raise ValueError("Unexpected Native indexer scale offset")
    page_bytes = key.stride(0)
    if page_bytes < INDEXER_MIN_PAGE_BYTES or page_bytes % 64 or scale.stride(0) * 2 != page_bytes:
        raise ValueError("Unexpected Native indexer page stride")
    return physical_pages(key)


def table_storage(table: torch.Tensor) -> torch.Tensor:
    """Include allocated row padding in a contiguous descriptor; do not copy."""
    if table.dtype != torch.int32 or table.ndim != 2 or table.stride(1) != 1:
        raise ValueError("Native block tables must be INT32 with contiguous columns")
    if table.is_contiguous():
        return table
    rows, columns = table.shape
    stride = table.stride(0)
    if stride < columns or (table.storage_offset() + rows * stride) * 4 > table.untyped_storage().nbytes():
        raise ValueError("Native table allocation does not contain full padded rows")
    return table.as_strided((rows, stride), (stride, 1), table.storage_offset())
