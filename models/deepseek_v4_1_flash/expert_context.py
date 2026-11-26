# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""Tensor layouts for the host-backed expert slot bank."""

import argparse

import pypto.language as pl
from models.common.expert_cache import expert_cache_shape
from config import FLASH, N_LOCAL_EXPERTS


_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--expert-slots", type=int, required=True)
EXPERT_SLOTS = _parser.parse_known_args()[0].expert_slots
if EXPERT_SLOTS <= 0:
    _parser.error("--expert-slots must be positive")

# Each mapped slot contains one contiguous, independently packed scale matrix.
SCALE_BANK_LAYOUT = pl.ND
EXPERT_LAYERS = FLASH.num_hidden_layers
CACHE_ROWS, CACHE_COLS = expert_cache_shape(EXPERT_SLOTS, EXPERT_LAYERS, N_LOCAL_EXPERTS)
