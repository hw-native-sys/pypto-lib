#!/usr/bin/env bash
# Run the fixed DeepSeek-V4 MoE and EPLB benchmarks as one tracked suite.

set -euo pipefail

readonly SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
readonly REPO_ROOT="$(git -C "$(dirname "$SCRIPT_PATH")/.." rev-parse --show-toplevel)"
readonly DECODE_RUNNER="$REPO_ROOT/tools/run_dsv4_eplb_perf.sh"
readonly RESULT_HELPER="$REPO_ROOT/tools/dsv4_eplb_suite_results.py"
readonly SEED_LAUNCHER="$REPO_ROOT/tools/run_seeded_python.py"
readonly CANONICAL_DEVICE_SET="0,2,4,6,8,10,12,14"
readonly SUITE_CONTRACT_VERSION="dsv4-eplb-suite-v2"
readonly METRIC_CONTRACT_MOE="dsv4-moe-ep8-v1"
readonly METRIC_CONTRACT_EPLB="dsv4-eplb-v2"
readonly OFFICIAL_ROUNDS="100"
readonly OFFICIAL_WARMUP="5"
readonly OFFICIAL_SEED="1807"
readonly RING_TASK_WINDOW="262144"
readonly RING_DEP_POOL="262144"
readonly RING_HEAP="2147483648"

PLATFORM="a2a3"
DEVICE_SET="${TASK_DEVICE:-}"
OUTPUT_DIR=""
PYTHON_BIN="${PYPTO_PERF_PYTHON:-python}"
BENCH_ROUNDS="${PYPTO_BENCH_ROUNDS:-$OFFICIAL_ROUNDS}"
BENCH_WARMUP="${PYPTO_BENCH_WARMUP:-$OFFICIAL_WARMUP}"
FIXTURE_SEED="${PYPTO_PERF_SEED:-$OFFICIAL_SEED}"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: tools/run_dsv4_eplb_suite.sh [OPTIONS]

Run the formal eight-card DeepSeek-V4 suite in this fixed order:
  1. MoE EP8
  2. Decode Main / Compare3
  3. Decode MTP / Compare4

Every completed case is fsync-appended to case-results.jsonl. The runner also
atomically refreshes suite-result.json after each case and at finalization.

Options:
  --device IDS          Must be 0,2,4,6,8,10,12,14. Defaults to TASK_DEVICE.
  --output-dir DIR      Result directory. Defaults below build_output/.
  --platform NAME       Must be a2a3 (default: a2a3).
  --python PATH         Python executable used for models and parsers.
  --rounds N            Must be 100 (default: 100).
  --warmup N            Must be 5 (default: 5).
  --seed N              Must be 1807 (default: 1807).
  --dry-run             Print resolved commands without writing or running.
  -h, --help            Show this help.

Measured suites require PYPTO_DEVICE_MAPPING_JSON as an ordered JSON list with
logical_rank, requested_device_id, physical_device_id, and serial per rank.
Published anchors should also set PYPTO_TOOLCHAIN_EPOCH and PYPTO_DEVICE_EPOCH.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

require_uint() {
    local name="$1"
    local value="$2"
    [[ "$value" =~ ^[0-9]+$ ]] || die "$name must be a non-negative integer: $value"
}

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --device)
            [[ "$#" -ge 2 ]] || die "--device requires a value"
            DEVICE_SET="$2"
            shift 2
            ;;
        --output-dir)
            [[ "$#" -ge 2 ]] || die "--output-dir requires a value"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --platform)
            [[ "$#" -ge 2 ]] || die "--platform requires a value"
            PLATFORM="$2"
            shift 2
            ;;
        --python)
            [[ "$#" -ge 2 ]] || die "--python requires a value"
            PYTHON_BIN="$2"
            shift 2
            ;;
        --rounds)
            [[ "$#" -ge 2 ]] || die "--rounds requires a value"
            BENCH_ROUNDS="$2"
            shift 2
            ;;
        --warmup)
            [[ "$#" -ge 2 ]] || die "--warmup requires a value"
            BENCH_WARMUP="$2"
            shift 2
            ;;
        --seed)
            [[ "$#" -ge 2 ]] || die "--seed requires a value"
            FIXTURE_SEED="$2"
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

require_uint "--rounds" "$BENCH_ROUNDS"
require_uint "--warmup" "$BENCH_WARMUP"
require_uint "--seed" "$FIXTURE_SEED"
[[ -n "$DEVICE_SET" ]] || die "--device is required outside a task-submit allocation"
[[ "$DEVICE_SET" == "$CANONICAL_DEVICE_SET" ]] || \
    die "official DSV4 EPLB suite requires --device $CANONICAL_DEVICE_SET"
[[ "$PLATFORM" == "a2a3" ]] || die "official DSV4 EPLB suite requires --platform a2a3"
[[ "$BENCH_ROUNDS" == "$OFFICIAL_ROUNDS" ]] || \
    die "official DSV4 EPLB suite requires --rounds $OFFICIAL_ROUNDS"
[[ "$BENCH_WARMUP" == "$OFFICIAL_WARMUP" ]] || \
    die "official DSV4 EPLB suite requires --warmup $OFFICIAL_WARMUP"
[[ "$FIXTURE_SEED" == "$OFFICIAL_SEED" ]] || \
    die "official DSV4 EPLB suite requires --seed $OFFICIAL_SEED"

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$REPO_ROOT/build_output/dsv4_eplb_suite/$(date -u '+%Y%m%dT%H%M%SZ')"
fi

MOE_COMMAND=(
    "$PYTHON_BIN"
    "$REPO_ROOT/models/deepseek_v4_flash_mtp/moe.py"
    -p "$PLATFORM"
    --ep 8
    --experts-per-rank 16
    -d "$DEVICE_SET"
    --layer-id 0
    --num-tokens 8
    --balanced-routing
    --seed "$FIXTURE_SEED"
    --enable-l2-swimlane 0
)

decode_command() {
    local case_id="$1"
    local script="eplb_decode_logits.py"
    DECODE_COMMAND=(
        "$PYTHON_BIN"
        "$SEED_LAUNCHER"
        --seed "$FIXTURE_SEED"
        --
        "$REPO_ROOT/models/deepseek_v4_flash_mtp/$script"
        -p "$PLATFORM"
        -d "$DEVICE_SET"
        --ep 8
        --tp 4
        --experts-per-rank 16
        --start-pos 8192
        --num-tokens 8
        --enable-l2-swimlane 0
    )
    if [[ "$case_id" == "mtp-core" ]]; then
        DECODE_COMMAND[5]="$REPO_ROOT/models/deepseek_v4_flash_mtp/eplb_mtp_core.py"
        DECODE_COMMAND+=(--finite-only)
    fi
}

if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'Repository SHA: %s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD)"
    printf 'Output: %s\n' "$OUTPUT_DIR"
    printf 'Suite contract: %s\n' "$SUITE_CONTRACT_VERSION"
    printf 'Metric contracts: moe-ep8=%s decode-logits=%s mtp-core=%s\n' \
        "$METRIC_CONTRACT_MOE" "$METRIC_CONTRACT_EPLB" "$METRIC_CONTRACT_EPLB"
    printf 'Frozen environment: PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=%s PYPTO_BENCH=1 PYPTO_BENCH_RAW=1 PYPTO_BENCH_ROUNDS=%s PYPTO_BENCH_WARMUP=%s\n' \
        "$FIXTURE_SEED" "$BENCH_ROUNDS" "$BENCH_WARMUP"
    printf 'Provenance epochs: toolchain=%s device=%s physical_mapping=%s\n' \
        "${PYPTO_TOOLCHAIN_EPOCH:-unassigned}" "${PYPTO_DEVICE_EPOCH:-unassigned}" \
        "$([[ -n "${PYPTO_DEVICE_MAPPING_JSON:-}" ]] && printf provided || printf unavailable)"
    printf 'moe-ep8:\n'
    print_command "${MOE_COMMAND[@]}"
    printf 'decode-logits:\n'
    print_command "$DECODE_RUNNER" --device "$DEVICE_SET" --case decode-logits \
        --platform "$PLATFORM" --python "$PYTHON_BIN" --rounds "$BENCH_ROUNDS" \
        --warmup "$BENCH_WARMUP" --seed "$FIXTURE_SEED" \
        --output-dir "$OUTPUT_DIR/cases/decode-logits"
    printf 'mtp-core:\n'
    print_command "$DECODE_RUNNER" --device "$DEVICE_SET" --case mtp-core \
        --platform "$PLATFORM" --python "$PYTHON_BIN" --rounds "$BENCH_ROUNDS" \
        --warmup "$BENCH_WARMUP" --seed "$FIXTURE_SEED" \
        --output-dir "$OUTPUT_DIR/cases/mtp-core"
    exit 0
fi

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BIN"
[[ -f "$RESULT_HELPER" ]] || die "suite result helper not found: $RESULT_HELPER"
[[ -x "$DECODE_RUNNER" ]] || die "decode performance runner is not executable: $DECODE_RUNNER"
[[ -n "${PYPTO_DEVICE_MAPPING_JSON:-}" ]] || \
    die "PYPTO_DEVICE_MAPPING_JSON is required for a measured suite"
[[ ! -e "$OUTPUT_DIR" ]] || die "output directory already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/cases"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"

readonly CONTEXT_FILE="$OUTPUT_DIR/suite-context.json"
readonly JOURNAL_FILE="$OUTPUT_DIR/case-results.jsonl"
readonly SUITE_RESULT_FILE="$OUTPUT_DIR/suite-result.json"
readonly STARTED_AT="$(date -u --iso-8601=seconds)"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED="$FIXTURE_SEED"
export PYPTO_BENCH=1
export PYPTO_BENCH_RAW=1
export PYPTO_BENCH_ROUNDS="$BENCH_ROUNDS"
export PYPTO_BENCH_WARMUP="$BENCH_WARMUP"
export PYPTO_RUNTIME_LOG=error
export SIMPLER_DEVICE_STRACE_ENABLE=1
export PTO2_RING_TASK_WINDOW="$RING_TASK_WINDOW"
export PTO2_RING_DEP_POOL="$RING_DEP_POOL"
export PTO2_RING_HEAP="$RING_HEAP"

"$PYTHON_BIN" "$RESULT_HELPER" init \
    --repo "$REPO_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --context "$CONTEXT_FILE" \
    --suite-output "$SUITE_RESULT_FILE" \
    --device "$DEVICE_SET" \
    --rounds "$BENCH_ROUNDS" \
    --warmup "$BENCH_WARMUP" \
    --seed "$FIXTURE_SEED" \
    --ring-task-window "$RING_TASK_WINDOW" \
    --ring-dep-pool "$RING_DEP_POOL" \
    --ring-heap "$RING_HEAP" \
    --started-at "$STARTED_AT"

record_case() {
    local case_id="$1"
    local log_file="$2"
    local process_rc="$3"
    local case_started_at="$4"
    local case_finished_at="$5"
    shift 5
    local -a helper_command=(
        "$PYTHON_BIN" "$RESULT_HELPER" record
        --context "$CONTEXT_FILE"
        --journal "$JOURNAL_FILE"
        --suite-output "$SUITE_RESULT_FILE"
        --case "$case_id"
        --log "$log_file"
        --process-rc "$process_rc"
        --started-at "$case_started_at"
        --finished-at "$case_finished_at"
    )
    local argument
    for argument in "$@"; do
        helper_command+=("--command-arg=$argument")
    done
    "${helper_command[@]}"
}

failures=0

moe_log="$OUTPUT_DIR/cases/moe-ep8.log"
case_started_at="$(date -u --iso-8601=seconds)"
printf '[SUITE RUN] moe-ep8\n'
print_command "${MOE_COMMAND[@]}"
set +e
"${MOE_COMMAND[@]}" 2>&1 | tee "$moe_log"
process_rc="${PIPESTATUS[0]}"
set -e
case_finished_at="$(date -u --iso-8601=seconds)"
MOE_RECORDED_COMMAND=(
    python models/deepseek_v4_flash_mtp/moe.py
    -p "$PLATFORM" --ep 8 --experts-per-rank 16 -d "$DEVICE_SET"
    --layer-id 0 --num-tokens 8 --balanced-routing --seed "$FIXTURE_SEED"
    --enable-l2-swimlane 0
)
record_case "moe-ep8" "$moe_log" "$process_rc" "$case_started_at" "$case_finished_at" \
    "${MOE_RECORDED_COMMAND[@]}" || failures=$((failures + 1))

for case_id in decode-logits mtp-core; do
    case_dir="$OUTPUT_DIR/cases/$case_id"
    case_started_at="$(date -u --iso-8601=seconds)"
    printf '[SUITE RUN] %s\n' "$case_id"
    set +e
    "$DECODE_RUNNER" \
        --device "$DEVICE_SET" \
        --case "$case_id" \
        --platform "$PLATFORM" \
        --python "$PYTHON_BIN" \
        --rounds "$BENCH_ROUNDS" \
        --warmup "$BENCH_WARMUP" \
        --seed "$FIXTURE_SEED" \
        --output-dir "$case_dir"
    runner_rc="$?"
    set -e
    case_finished_at="$(date -u --iso-8601=seconds)"
    process_rc="$runner_rc"
    if [[ -f "$case_dir/results.tsv" ]]; then
        recorded_rc="$(awk -F '\t' 'NR == 2 { print $3 }' "$case_dir/results.tsv")"
        if [[ "$recorded_rc" =~ ^[0-9]+$ ]]; then
            process_rc="$recorded_rc"
        fi
    fi
    decode_command "$case_id"
    DECODE_RECORDED_COMMAND=(
        python tools/run_seeded_python.py --seed "$FIXTURE_SEED" --
        "models/deepseek_v4_flash_mtp/$(basename "${DECODE_COMMAND[5]}")"
    )
    DECODE_RECORDED_COMMAND+=("${DECODE_COMMAND[@]:6}")
    record_case "$case_id" "$case_dir/$case_id.log" "$process_rc" \
        "$case_started_at" "$case_finished_at" "${DECODE_RECORDED_COMMAND[@]}" || \
        failures=$((failures + 1))
done

finished_at="$(date -u --iso-8601=seconds)"
set +e
"$PYTHON_BIN" "$RESULT_HELPER" finalize \
    --repo "$REPO_ROOT" \
    --context "$CONTEXT_FILE" \
    --journal "$JOURNAL_FILE" \
    --suite-output "$SUITE_RESULT_FILE" \
    --finished-at "$finished_at"
finalize_rc="$?"
set -e

printf 'Case journal: %s\n' "$JOURNAL_FILE"
printf 'Suite result: %s\n' "$SUITE_RESULT_FILE"
if [[ "$failures" -ne 0 || "$finalize_rc" -ne 0 ]]; then
    printf 'ERROR: %s case(s) failed execution or official metric validation.\n' "$failures" >&2
    exit 1
fi
