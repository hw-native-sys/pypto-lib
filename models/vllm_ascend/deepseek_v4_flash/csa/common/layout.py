# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native page geometry used by the CSA integration's tensor descriptors."""

import pypto.language as pl

INDEXER_PAGE_BYTES_DYN = pl.dynamic("CSA_INDEXER_PAGE_BYTES")
INDEXER_TABLE_COLUMNS_DYN = pl.dynamic("CSA_INDEXER_TABLE_COLUMNS")
ORIGINAL_TABLE_COLUMNS_DYN = pl.dynamic("CSA_ORIGINAL_TABLE_COLUMNS")
COMPRESSED_TABLE_COLUMNS_DYN = pl.dynamic("CSA_COMPRESSED_TABLE_COLUMNS")
COMPRESSED_ROWS_DYN = pl.dynamic("CSA_COMPRESSED_METADATA_ROWS")
INDEXER_ROWS_DYN = pl.dynamic("CSA_INDEXER_METADATA_ROWS")
QUERY_BOUNDS_DYN = pl.dynamic("CSA_QUERY_BOUNDS")
STATE_PAGE_ELEMENTS_DYN = pl.dynamic("CSA_STATE_PAGE_ELEMENTS")
STATE_TABLE_COLUMNS_DYN = pl.dynamic("CSA_STATE_TABLE_COLUMNS")
INNER_STATE_PAGE_ELEMENTS_DYN = pl.dynamic("CSA_INNER_STATE_PAGE_ELEMENTS")
INNER_STATE_TABLE_COLUMNS_DYN = pl.dynamic("CSA_INNER_STATE_TABLE_COLUMNS")
INDEXER_PAGE_TOKENS = 32
INDEXER_KEY_DIM = 128
INDEXER_KEY_BYTES = INDEXER_PAGE_TOKENS * INDEXER_KEY_DIM
INDEXER_MIN_PAGE_BYTES = INDEXER_KEY_BYTES + INDEXER_PAGE_TOKENS * 2
