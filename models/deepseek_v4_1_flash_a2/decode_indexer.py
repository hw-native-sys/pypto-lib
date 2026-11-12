# Copyright (c) PyPTO Contributors.
# This program is free software; you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.
"""Unified A2/A3 Decode index query, selection and candidate entry points."""

from .candidate_pool import make_candidate_block_pool
from .index_cache import dequantize_index_cache, make_index_cache_quantize, quantize_index_cache
from .indexer import (
    INDEX_HEADS,
    INDEX_TOPK,
    INDEX_WIDTH,
    QUANT_ROWS_TILE,
    WEIGHT_ROWS_TILE,
    WINDOW_SIZE,
    make_empty_selection,
    make_indexer_projection,
    make_indexer_query_quantize,
    make_sparse_controls,
)
from .indexer_prepare import make_indexer_prepare, prepare_indexer_indices_reference
from .indexer_query import (
    indexer_query_reference,
    make_l3_indexer_query,
    quantize_indexer_query_reference,
)
from .indexer_select import (
    indexer_select_reference,
    make_indexer_select,
    make_indexer_select_restricted,
)


__all__ = [
    "INDEX_HEADS",
    "INDEX_TOPK",
    "INDEX_WIDTH",
    "QUANT_ROWS_TILE",
    "WEIGHT_ROWS_TILE",
    "WINDOW_SIZE",
    "dequantize_index_cache",
    "indexer_query_reference",
    "indexer_select_reference",
    "make_candidate_block_pool",
    "make_empty_selection",
    "make_index_cache_quantize",
    "make_indexer_prepare",
    "make_indexer_projection",
    "make_indexer_query_quantize",
    "make_indexer_select",
    "make_indexer_select_restricted",
    "make_l3_indexer_query",
    "make_sparse_controls",
    "prepare_indexer_indices_reference",
    "quantize_index_cache",
    "quantize_indexer_query_reference",
]
