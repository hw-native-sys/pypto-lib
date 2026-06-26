# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared SiLU activation primitive: ``@pl.inline``.

See ``~/workspace/subprojects/pypto/docs/guide-ct-args.md`` for the full
context on ``@pl.inline`` + CT kwargs (in the compiler subproject, not here).
"""

import pypto.language as pl


@pl.inline
def silu_activation(gate, up):
    """SiLU(gate) * up = (gate * sigmoid(gate)) * up.

    Args:
        gate: Gate tensor (the SiLU input).
        up: Up tensor (element-wise multiplier).

    Returns:
        ``gate * sigmoid(gate) * up`` tensor.
    """
    sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate)), 1.0))
    return pl.mul(pl.mul(gate, sigmoid), up)
