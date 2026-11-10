#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -eo pipefail
: "${TASK_DEVICE:?Run this script through task-submit}"
: "${PYPTO_QWEN3_MODEL_DIR:?Set the Qwen3-14B checkpoint path}"
: "${ASCEND_HOME_PATH:?Source the CANN environment first}"
here="$(cd "$(dirname "$0")" && pwd)"
simpler="$(realpath "$1")"
output="$(realpath "$2")"
bridge="$output/bridge/_simpler_framework_queue$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
test -f "$bridge"

"$simpler/.claude/skills/onboard-arch-precheck/check.sh" a2a3
unset ASCEND_RT_VISIBLE_DEVICES
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TASK_QUEUE_ENABLE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_USE_V2_MODEL_RUNNER=0
# vLLM-Ascend imports ACL from CANN. Keep the pinned runtime venv first.
runtime_site="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
lib_root="$(cd "$here/../../.." && pwd)"
export PYTHONPATH="$here:$lib_root:$runtime_site:$ASCEND_HOME_PATH/python/site-packages"
for runtime in tmr hbg; do
    for mode in eager capture; do
        python "$here/smoke.py" --model "$PYPTO_QWEN3_MODEL_DIR" --artifact "$output/$runtime" \
            --mode "$mode" --bridge "$bridge" --output "$output/$runtime-$mode-result.json" \
            |& tee "$output/$runtime-$mode.log"
    done
done
