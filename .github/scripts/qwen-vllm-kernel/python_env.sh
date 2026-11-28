#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#     http://www.hiascend.com/en/software/cann/community
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Source once after activating the runtime venv. The cache fingerprints the
# PTOAS interpreter's Python path, so preparation and playback share this setup.
qwen_scripts="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
qwen_lib="$(cd "$qwen_scripts/../../.." && pwd)"
qwen_site="$(python -c 'import sysconfig; p=sysconfig.get_paths(); print(p["platlib"] + ":" + p["purelib"])')"
export PYTHONPATH="$qwen_scripts:$qwen_lib:$qwen_lib/models/qwen3_14b:$qwen_site${PYTHONPATH:+:$PYTHONPATH}:$ASCEND_HOME_PATH/python/site-packages"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
unset qwen_scripts qwen_lib qwen_site
