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
"$simpler/.claude/skills/onboard-arch-precheck/check.sh" a2a3
unset ASCEND_RT_VISIBLE_DEVICES
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TASK_QUEUE_ENABLE=1
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
source "$here/python_env.sh"
for runtime in ${QWEN_SMOKE_RUNTIMES:-tmr hbg}; do
    for mode in ${QWEN_SMOKE_MODES:-eager capture}; do
        python "$here/smoke.py" --model "$PYPTO_QWEN3_MODEL_DIR" --runtime "$runtime" --prepared "$output" \
            --mode "$mode" --output "$output/$runtime-$mode-result.json" \
            |& tee "$output/$runtime-$mode.log"
        if grep -Eq 'PyPTO kernel (shutdown did not complete safely|cleanup completed after an earlier submission error)' \
            "$output/$runtime-$mode.log"; then
            echo "PyPTO reported a shutdown error for $runtime/$mode" >&2
            exit 1
        fi
    done
done
