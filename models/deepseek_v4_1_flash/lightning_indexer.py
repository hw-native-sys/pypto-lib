# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared bindings for the native paged lightning indexer."""
import os
from pathlib import Path

import pypto.language as pl
from pypto.runtime import pto_isa_include_dir


KERNEL_DIR = Path(__file__).parent / "kernels"
LI_SOURCE = KERNEL_DIR / "lightning_indexer.cpp"
CANN = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
WORKERS = 32


LI_INCLUDES = (pto_isa_include_dir(), CANN / "asc", CANN / "asc/include",
               CANN / "asc/include/basic_api", CANN / "include")
# Stage tags in the lightning_indexer.cpp task ABI.
METADATA, ATTENTION, SCORE, POOL, SCORE_CANDIDATES = range(5)


@pl.jit.extern(core_type="aiv", aiv_source=LI_SOURCE, include_dirs=LI_INCLUDES)
def _score_prepare(
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    metadata: pl.Out[pl.Tensor],
    mode: pl.Scalar[pl.INT64],
    active: pl.Scalar[pl.INT64],
) -> pl.Tensor:
    ...


@pl.jit.extern(core_type="mixed", aic_source=LI_SOURCE, aiv_source=LI_SOURCE,
               dual_aiv_dispatch=True, include_dirs=LI_INCLUDES)
def _score(
    query: pl.Tensor,
    query_scale: pl.Tensor,
    weights: pl.Tensor,
    key: pl.Tensor,
    key_scale: pl.Tensor,
    table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    metadata: pl.Tensor,
    scores: pl.Out[pl.Tensor],
    active: pl.Scalar[pl.INT64],
    mode: pl.Scalar[pl.INT64],
) -> pl.Tensor:
    ...


@pl.jit.extern(core_type="mixed", aic_source=LI_SOURCE, aiv_source=KERNEL_DIR / "prefill_lightning_indexer.cpp",
               dual_aiv_dispatch=True, include_dirs=(*LI_INCLUDES, KERNEL_DIR))
def _prefill_score(
    query: pl.Tensor,
    query_scale: pl.Tensor,
    weights: pl.Tensor,
    key: pl.Tensor,
    key_scale: pl.Tensor,
    table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    metadata: pl.Tensor,
    scores: pl.Out[pl.Tensor],
    active: pl.Scalar[pl.INT64],
    mode: pl.Scalar[pl.INT64],
) -> pl.Tensor:
    ...


@pl.jit.extern(core_type="mixed", aic_source=LI_SOURCE, aiv_source=LI_SOURCE,
               dual_aiv_dispatch=True, include_dirs=LI_INCLUDES)
def _score_candidates(
    query: pl.Tensor,
    query_scale: pl.Tensor,
    weights: pl.Tensor,
    key: pl.Tensor,
    key_scale: pl.Tensor,
    table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    metadata: pl.Tensor,
    scores: pl.Out[pl.Tensor],
    block_scores: pl.Out[pl.Tensor],
    mode: pl.Scalar[pl.INT64],
    active: pl.Scalar[pl.INT64],
) -> pl.Tensor:
    ...


@pl.jit.inline(auto_scope=False)
def prepare_score_metadata(
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    active_rows = pl.cast(num_tokens, pl.INT64)
    metadata = pl.create_tensor([33, 2], dtype=pl.INT32)
    with pl.spmd(1, name_hint="qli_score_metadata"):
        _score_prepare(request_ids, lengths, metadata, METADATA, active_rows)
    return metadata


@pl.jit.inline(auto_scope=False)
def score_paged(
    query: pl.Tensor,
    query_scale: pl.Tensor,
    weights: pl.Tensor,
    key: pl.Tensor,
    key_scale: pl.Tensor,
    table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    metadata: pl.Tensor,
    scores: pl.Tensor,
    active: pl.Scalar[pl.INT32],
    block_scores: pl.Tensor,
    candidates: pl.constexpr = False,
):
    active_rows = pl.cast(active, pl.INT64)
    with pl.spmd(WORKERS, name_hint="qli_score"):
        if candidates:
            _score_candidates(query, query_scale, weights, key, key_scale, table, request_ids, lengths, metadata, scores, block_scores, SCORE_CANDIDATES, active_rows)
        else:
            _score(query, query_scale, weights, key, key_scale, table, request_ids, lengths, metadata, scores, active_rows, SCORE)
