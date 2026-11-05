# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Print the loaded reference CANN libraries and mHC symbols for one NPU rank."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path


def main() -> int:
    import torch
    import torch_npu

    print("python", os.sys.executable)
    print("torch", torch.__version__)
    print("torch_npu", torch_npu.__version__)
    print("env", {key: os.environ.get(key) for key in (
        "SOC_VERSION", "TASK_DEVICE", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_HOME_PATH",
        "LD_LIBRARY_PATH", "PYTHONPATH",
    )})
    print("find_libopapi", ctypes.util.find_library("opapi"))
    try:
        library = ctypes.CDLL("libopapi.so", mode=ctypes.RTLD_GLOBAL)
        print("libopapi_name", library._name)
        for symbol in ("aclnnMhcPre", "aclnnMhcPreGetWorkspaceSize"):
            try:
                print("symbol", symbol, hex(ctypes.cast(getattr(library, symbol), ctypes.c_void_p).value))
            except AttributeError as exc:
                print("symbol_missing", symbol, repr(exc))
    except OSError as exc:
        print("libopapi_load_error", repr(exc))
    print("maps")
    maps = Path("/proc/self/maps").read_text(encoding="utf-8", errors="replace")
    for line in maps.splitlines():
        if "opapi" in line.lower() or "torch_npu" in line.lower() or "ascend" in line.lower():
            print(line)
    if hasattr(torch, "npu"):
        torch.npu.set_device(f"npu:{os.environ.get('TASK_DEVICE', '0').split(',')[0]}")
        print("npu_device", torch.npu.current_device())
        print("ascend_ops", [name for name in dir(torch.ops._C_ascend) if "hc" in name.lower()])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
