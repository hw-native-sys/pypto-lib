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
simpler="$(realpath "$1")"
vllm="$(realpath "$2")"
ascend="$(realpath "$3")"
output="$(realpath "$4")"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 MAX_JOBS="${MAX_JOBS:-2}"
export PIP_RETRIES=10 PIP_TIMEOUT=120 PIP_RESUME_RETRIES=10

# Keep the compiler's installed Simpler pin in the compiler environment.
unset PYTHONPATH
if command -v uv >/dev/null; then
    uv venv --seed --python "$(command -v python)" "$output/venv"
else
    python -m venv "$output/venv"
fi
source "$output/venv/bin/activate"
python -m pip install --upgrade 'pip>=23'
python -m pip install 'setuptools>=77.0.3,<81' setuptools-scm wheel 'cmake>=3.26' ninja \
    scikit-build-core nanobind pybind11 'torch==2.10.0' 'torch-npu==2.10.0' 'torchvision==0.25.0' 'torchaudio==2.10.0'
VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=0.20.2 \
    python -m pip install --no-build-isolation -e "$vllm"
requirements="$output/ascend-requirements.txt"
awk '!/^(memfabric_hybrid|memcache_hybrid|triton-ascend==)/' "$ascend/requirements.txt" > "$requirements"
python -m pip install -r "$requirements"
# Preserve the pinned native vLLM prefill environment, including Triton-Ascend.
# Triton-Ascend overlays the community Triton installation.
python -m pip install 'triton==3.5.0'
python -m pip install --no-deps 'numpy==1.26.4' 'opencv-python-headless==4.13.0.92'
triton_release=https://github.com/triton-lang/triton-ascend/releases/download/v3.2.1
triton_wheel=triton_ascend-3.2.1-cp310-cp310-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
triton_sha256=d310f8776752fb82c7a9caf836d4ed289670332c257aea0cade2dad4dc7ebec6
wheel_path="$output/$triton_wheel"
cached_wheel="${PIP_CACHE_DIR:-$output/pip-cache}/qwen-artifacts/$triton_sha256/$triton_wheel"
if [[ -f "$cached_wheel" ]] && printf '%s  %s\n' "$triton_sha256" "$cached_wheel" | sha256sum --status --check -; then
    cp "$cached_wheel" "$wheel_path"
fi
# Resume interrupted release downloads, with a bounded time for each attempt.
for attempt in 1 2 3; do
    if [[ -f "$wheel_path" ]] && printf '%s  %s\n' "$triton_sha256" "$wheel_path" | sha256sum --status --check -; then
        break
    fi
    if curl --fail --location --connect-timeout 15 --max-time 300 --continue-at - \
        --output "$wheel_path" "$triton_release/$triton_wheel"; then
        break
    fi
    echo "Triton-Ascend download attempt $attempt failed; retaining partial bytes"
done
printf '%s  %s\n' "$triton_sha256" "$wheel_path" | sha256sum --check -
mkdir -p "$(dirname "$cached_wheel")"
cache_tmp="$(mktemp "$(dirname "$cached_wheel")/.triton-wheel.XXXXXX")"
cp "$wheel_path" "$cache_tmp"
mv -f "$cache_tmp" "$cached_wheel"
python -m pip install --no-deps "$wheel_path"
SETUPTOOLS_SCM_PRETEND_VERSION=0.20.2rc1 COMPILE_CUSTOM_KERNELS=0 \
    python -m pip install --no-build-isolation --no-deps -e "$ascend"

export PTO_ISA_ROOT="$output/runtime-pto-isa"
git init -q "$PTO_ISA_ROOT"
git -C "$PTO_ISA_ROOT" fetch --depth 1 https://github.com/hw-native-sys/pto-isa.git "$(cat "$simpler/pto_isa.pin")"
git -C "$PTO_ISA_ROOT" checkout --detach FETCH_HEAD
python -m pip install --no-build-isolation --no-deps \
    --config-settings=build.targets=build_package_a2a3 "$simpler"
python "$simpler/examples/workers/l2/framework_eager/main.py" --compile-only --build-dir "$output/bridge"
