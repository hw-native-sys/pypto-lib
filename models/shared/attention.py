# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared online-softmax primitive: ``@pl.inline``.

See ``~/workspace/subprojects/pypto/docs/guide-ct-args.md`` for the full
context on ``@pl.inline`` + CT kwargs (in the compiler subproject, not here).
"""

import pypto.language as pl


@pl.inline
def online_softmax_ab(mi, cur_mi):
    """Compute online-softmax coefficients from running ``mi`` and ``cur_mi``.

    Returns:
        Tuple ``(mi_new, alpha, beta)``.  All three are fresh variables
        (not loop-carried), so tuple unpacking is safe.
    """
    mi_new = pl.maximum(mi, cur_mi)
    alpha = pl.exp(pl.sub(mi, mi_new))
    beta = pl.exp(pl.sub(cur_mi, mi_new))
    return mi_new, alpha, beta
