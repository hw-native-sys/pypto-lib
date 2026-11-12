# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host identity helpers for native operator subprocesses.

Device allocation belongs to the caller. These helpers do not reserve a device
or launch a persistent process.
"""

import hashlib
import os
from pathlib import Path


def file_sha(path):
    """Hash a source, runtime library, or staging artifact."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(value):
    """Hash actual contiguous CPU tensor bytes, including BF16 payloads."""
    import torch

    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def validate_device(settings):
    """Require one explicit physical device mapped to logical device zero."""
    device = settings["physical_device"]
    if type(device) is not int or device < 0:
        raise ValueError("physical_device must be a nonnegative integer")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(device):
        raise RuntimeError("ASCEND_RT_VISIBLE_DEVICES must select exactly physical_device")
    return os.getpid()
