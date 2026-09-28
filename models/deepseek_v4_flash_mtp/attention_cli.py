# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Standalone attention model selection before shared kernel imports."""

import argparse

import config


def add_model_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=("flash", "pro"), default="flash",
                        help="Attention model preset (default: flash).")


def select_model() -> None:
    """Bind the process-local preset before kernel shapes are evaluated."""
    parser = argparse.ArgumentParser(add_help=False)
    add_model_argument(parser)
    args, _ = parser.parse_known_args()
    config.FLASH = config.PRESETS[args.model]
