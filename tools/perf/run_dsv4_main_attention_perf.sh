#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

readonly SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
readonly REPO_ROOT="$(git -C "$(dirname "$SCRIPT_PATH")/../.." rev-parse --show-toplevel)"
readonly MANIFEST="$REPO_ROOT/tools/perf/suites/dsv4_main_attention.json"
readonly PARSER="$REPO_ROOT/tools/perf/dsv4_main_attention_metrics.py"
readonly SEED_LAUNCHER="$REPO_ROOT/tools/perf/deterministic_run.py"
readonly PLATFORM="a2a3"
readonly CANONICAL_DEVICE="4"
readonly SEED="1807"
readonly ROUNDS="100"
readonly WARMUP="5"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

DEVICE_ID="${TASK_DEVICE:-}"
OUTPUT_DIR=""
SELECTED_CASE="all"
PYTHON_BIN="${PYPTO_PERF_PYTHON:-python}"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: tools/perf/run_dsv4_main_attention_perf.sh [OPTIONS]

Run the fixed DeepSeek-V4 Flash CSA/HCA/SWA main-branch performance suite.
The official contract uses A2/A3 device 4, seed 1807, 5 warmup rounds, 100
measured rounds, raw samples, the effective_us median, and numerical PASS for
kv_cache and x_out. The runner never changes Git state or allocates a device.

Options:
  --device ID          Must be 4. Defaults to TASK_DEVICE.
  --output-dir DIR     New result directory. Defaults below build_output/.
  --case NAME          all, attention-csa, attention-hca, or attention-swa.
  --python PATH        Python executable (default: PYPTO_PERF_PYTHON or python).
  --dry-run            Print the resolved contract and commands without running.
  -h, --help           Show this help.

Example inside an existing task-submit allocation:
  tools/perf/run_dsv4_main_attention_perf.sh --device "$TASK_DEVICE"
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --device)
            [[ "$#" -ge 2 ]] || die "--device requires a value"
            DEVICE_ID="$2"
            shift 2
            ;;
        --output-dir)
            [[ "$#" -ge 2 ]] || die "--output-dir requires a value"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --case)
            [[ "$#" -ge 2 ]] || die "--case requires a value"
            SELECTED_CASE="$2"
            shift 2
            ;;
        --python)
            [[ "$#" -ge 2 ]] || die "--python requires a value"
            PYTHON_BIN="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

case "$SELECTED_CASE" in
    all|attention-csa|attention-hca|attention-swa) ;;
    *) die "unsupported --case: $SELECTED_CASE" ;;
esac
[[ -n "$DEVICE_ID" ]] || die "--device is required outside an existing allocation"
[[ "$DEVICE_ID" == "$CANONICAL_DEVICE" ]] || \
    die "official main-attention metrics require --device $CANONICAL_DEVICE"
[[ -f "$MANIFEST" ]] || die "suite manifest not found: $MANIFEST"
[[ -f "$PARSER" ]] || die "metric parser not found: $PARSER"
[[ -f "$SEED_LAUNCHER" ]] || die "deterministic launcher not found: $SEED_LAUNCHER"

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$REPO_ROOT/build_output/dsv4_main_attention_perf/$(date -u '+%Y%m%dT%H%M%SZ')"
fi

case_entrypoint() {
    case "$1" in
        attention-csa) printf '%s\n' "models/deepseek_v4_flash_mtp/decode_csa.py" ;;
        attention-hca) printf '%s\n' "models/deepseek_v4_flash_mtp/decode_hca.py" ;;
        attention-swa) printf '%s\n' "models/deepseek_v4_flash_mtp/decode_swa.py" ;;
        *) die "unknown case: $1" ;;
    esac
}

if [[ "$SELECTED_CASE" == "all" ]]; then
    CASES=(attention-csa attention-hca attention-swa)
else
    CASES=("$SELECTED_CASE")
fi

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

build_command() {
    local case_id="$1"
    local entrypoint
    entrypoint="$(case_entrypoint "$case_id")"
    CASE_COMMAND=(
        "$PYTHON_BIN" "$SEED_LAUNCHER" --seed "$SEED"
        "$REPO_ROOT/$entrypoint"
        -p "$PLATFORM" -d "$DEVICE_ID"
        --start-pos 8192 --enable-l2-swimlane 0
    )
}

if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'Suite: dsv4-main-attention\n'
    printf 'Contract: dsv4-main-attention-v1\n'
    printf 'Manifest: %s\n' "$MANIFEST"
    printf 'Output: %s/suite-result.json\n' "$OUTPUT_DIR"
    printf 'Device/platform: %s/%s\n' "$DEVICE_ID" "$PLATFORM"
    printf 'Sampling: seed=%s warmup=%s rounds=%s raw=1\n' "$SEED" "$WARMUP" "$ROUNDS"
    for case_id in "${CASES[@]}"; do
        printf '%s:\n' "$case_id"
        build_command "$case_id"
        print_command env \
            "PYTHONHASHSEED=$SEED" \
            PYPTO_BENCH=1 PYPTO_BENCH_RAW=1 \
            "PYPTO_BENCH_ROUNDS=$ROUNDS" "PYPTO_BENCH_WARMUP=$WARMUP" \
            PYPTO_RUNTIME_LOG=error SIMPLER_DEVICE_STRACE_ENABLE=1 \
            PTO2_RING_TASK_WINDOW=262144 PTO2_RING_DEP_POOL=262144 \
            PTO2_RING_HEAP=2147483648 \
            "${CASE_COMMAND[@]}"
    done
    exit 0
fi

if [[ "$PYTHON_BIN" == */* ]]; then
    [[ -x "$PYTHON_BIN" ]] || die "Python executable not found: $PYTHON_BIN"
else
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BIN"
fi
[[ -n "${PYPTO_DEVICE_MAPPING_JSON:-}" ]] || \
    die "PYPTO_DEVICE_MAPPING_JSON is required for a measured suite"
[[ ! -e "$OUTPUT_DIR" ]] || die "output directory already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/cases" "$OUTPUT_DIR/build"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
readonly STARTED_AT_UTC="$(date -u --iso-8601=seconds)"
export TASK_DEVICE="$DEVICE_ID"

failures=0
CASE_RESULT_ARGS=()
for case_id in "${CASES[@]}"; do
    log_path="$OUTPUT_DIR/logs/$case_id.log"
    case_result="$OUTPUT_DIR/cases/$case_id.json"
    build_command "$case_id"
    printf '[PERF SUITE] %s\n' "$case_id"
    print_command "${CASE_COMMAND[@]}"

    set +e
    PYTHONHASHSEED="$SEED" \
        PYPTO_PERF_SEED="$SEED" \
        PYPTO_BENCH=1 \
        PYPTO_BENCH_RAW=1 \
        PYPTO_BENCH_ROUNDS="$ROUNDS" \
        PYPTO_BENCH_WARMUP="$WARMUP" \
        PYPTO_RUNTIME_LOG=error \
        SIMPLER_DEVICE_STRACE_ENABLE=1 \
        PTO2_RING_TASK_WINDOW=262144 \
        PTO2_RING_DEP_POOL=262144 \
        PTO2_RING_HEAP=2147483648 \
        PYPTO_PROG_BUILD_DIR="$OUTPUT_DIR/build/$case_id" \
        "${CASE_COMMAND[@]}" 2>&1 | tee "$log_path"
    process_rc="${PIPESTATUS[0]}"
    set -e

    set +e
    "$PYTHON_BIN" "$PARSER" parse \
        --manifest "$MANIFEST" \
        --case "$case_id" \
        --log "$log_path" \
        --device "$DEVICE_ID" \
        --process-rc "$process_rc" \
        --output "$case_result" \
        --journal "$OUTPUT_DIR/case-results.jsonl"
    parser_rc="$?"
    set -e
    CASE_RESULT_ARGS+=(--case-result "$case_result")
    if [[ "$process_rc" -ne 0 || "$parser_rc" -ne 0 ]]; then
        failures=$((failures + 1))
    fi
done

readonly FINISHED_AT_UTC="$(date -u --iso-8601=seconds)"
set +e
"$PYTHON_BIN" "$PARSER" suite \
    --manifest "$MANIFEST" \
    --repo-root "$REPO_ROOT" \
    --device "$DEVICE_ID" \
    --started-at-utc "$STARTED_AT_UTC" \
    --finished-at-utc "$FINISHED_AT_UTC" \
    "${CASE_RESULT_ARGS[@]}" \
    --output "$OUTPUT_DIR/suite-result.json"
suite_rc="$?"
set -e

printf 'Suite result: %s\n' "$OUTPUT_DIR/suite-result.json"
if [[ "$failures" -ne 0 || "$suite_rc" -ne 0 ]]; then
    printf 'ERROR: %s case(s) failed execution or metric validation.\n' "$failures" >&2
    exit 1
fi
