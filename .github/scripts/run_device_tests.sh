#!/usr/bin/env bash
# Activate the same Python and runtime environment on the host and the allocated device.
set -eo pipefail
if [[ ${1:-} == --submit ]]; then
  ndev=$2
  shift 2
  flags=()
  if (( ndev > 1 )); then
    flags+=(--device-num "$ndev" --ignore-whitelist)
  fi
  printf -v command 'cd %q && PYPTO_PROG_BUILD_DIR=%q bash .github/scripts/run_device_tests.sh' \
    "$PWD" "${PYPTO_PROG_BUILD_DIR:-}"
  printf -v args ' %q' "$@"
  exec task-submit --device auto "${flags[@]}" --timeout 3600 --max-time 900 --run "$command$args"
fi

source activate.sh
export PYPTO_BENCH=0 PYPTO_LOG_LEVEL=error PYPTO_WARNING_LEVEL=none PYPTO_RUNTIME_LOG=error
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
if (( $# )); then
  unset PYPTO_CI_DEVICE_COUNT
  exec python "$@" -d "${TASK_DEVICE:?task-submit must allocate the devices}"
fi
# Host case processes import the dispatch helper when golden.run enters device CI.
export PYTHONPATH="$PYTHONPATH:$PWD/.github/scripts"
exec python .github/scripts/run_device_tests.py
