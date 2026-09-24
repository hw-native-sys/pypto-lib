# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compensated Cube score reduction and pipelined Vector Top-K."""
from pathlib import Path
import pypto.language as pl
from pypto.runtime import pto_isa_include_dir

_SOURCE = Path(__file__).parent / "kernels" / "indexer_cube_score" / "score_fused.cpp"

@pl.jit.extern(core_type="mixed", aic_source=_SOURCE, aiv_source=_SOURCE,
               dual_aiv_dispatch=True, include_dirs=(pto_isa_include_dir(),))
def indexer_cube_score_topk(
    pair_arena: pl.Out[pl.Tensor],
    pipe_workspace: pl.InOut[pl.Tensor],
    query_i8: pl.Tensor,
    coefficients_hi: pl.Tensor,
    coefficients_lo: pl.Tensor,
    key_cache: pl.Tensor,
    key_scale: pl.Tensor,
    block_table: pl.Tensor,
    position_ids: pl.Tensor,
    kv_seq_lens: pl.Tensor,
    math_constants: pl.Tensor,
    max_leaves: pl.Scalar[pl.INDEX],
    table_stride: pl.Scalar[pl.INDEX],
) -> pl.Tensor:
    """Return half-leaf Top-512 pairs, transferring one FP32 per candidate."""
    ...
