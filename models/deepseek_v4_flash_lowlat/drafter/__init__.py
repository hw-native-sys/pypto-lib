# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Load TP4/CP4/EP4 constants in an isolated model package."""

import sys

_saved_argv = sys.argv
try:
    sys.argv = [
        "native-drafter4",
        "--tp",
        "4",
        "--ep",
        "4",
        "--experts-per-rank",
        "64",
        "--dp",
        "1",
        "--weight-bank-size",
        "43",
    ]
    from .entry import drafter_inline
finally:
    sys.argv = _saved_argv
    del _saved_argv

__all__ = ["drafter_inline"]
