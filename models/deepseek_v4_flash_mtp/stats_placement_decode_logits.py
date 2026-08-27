# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=8
# ci: no-sim
"""DeepSeek-V4 stats-shaped hash-route replay with offline expert placement."""

import sys


_STATS_PLACEMENT_FLAG = "--stats-placement"
if _STATS_PLACEMENT_FLAG not in sys.argv:
    sys.argv.append(_STATS_PLACEMENT_FLAG)

from eplb_decode_logits import main


if __name__ == "__main__":
    main()
