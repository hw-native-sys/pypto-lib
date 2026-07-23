# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Dynamic token symbol owned by the packed prefill layer entry."""

import pypto.language as pl


# Keep the symbol at the packed-layer boundary. Reusable inline attention functions
# accept generic runtime-shaped tensors so their dynamic variables stay in the
# caller that created each packed request view.
PREFILL_TOKENS_DYN = pl.dynamic("PREFILL_TOKENS_DYN")


__all__ = ["PREFILL_TOKENS_DYN"]
