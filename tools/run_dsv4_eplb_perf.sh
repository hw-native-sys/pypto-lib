#!/usr/bin/env bash
# Run the fixed DeepSeek-V4 EPLB benchmarks without changing Git state.

set -euo pipefail

readonly SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
readonly REPO_ROOT="$(git -C "$(dirname "$SCRIPT_PATH")/.." rev-parse --show-toplevel)"

PLATFORM="a2a3"
DEVICE_SET="${TASK_DEVICE:-}"
OUTPUT_DIR=""
SELECTED_CASE="all"
PYTHON_BIN="${PYPTO_PERF_PYTHON:-python}"
BENCH_ROUNDS="${PYPTO_BENCH_ROUNDS:-100}"
BENCH_WARMUP="${PYPTO_BENCH_WARMUP:-5}"
COMPILE_ONLY=0
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: tools/run_dsv4_eplb_perf.sh [OPTIONS]

Run the fixed EP8/TP4 DeepSeek-V4 EPLB performance cases in the current
checkout. The runner never fetches, rebases, commits, or pushes.
The MTP-core case performs finite-output validation before timing.

Options:
  --device IDS          Eight comma-separated device IDs. Defaults to TASK_DEVICE.
  --output-dir DIR      Result directory. Defaults below build_output/.
  --platform NAME       Runtime platform (default: a2a3).
  --case NAME           all, decode-logits, or mtp-core (default: all).
  --python PATH         Python executable (default: python).
  --rounds N            Measured rounds (default: 100).
  --warmup N            Warmup rounds (default: 5).
  --compile-only        Compile each case without requiring timing metrics.
  --dry-run             Print resolved commands without writing or running.
  -h, --help            Show this help.

Example inside an existing task-submit allocation:
  tools/run_dsv4_eplb_perf.sh --device "$TASK_DEVICE"
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
        --compile-only)
            COMPILE_ONLY=1
            shift
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
    all|decode-logits|mtp-core) ;;
    *) die "--case must be all, decode-logits, or mtp-core: $SELECTED_CASE" ;;
esac

require_uint "--rounds" "$BENCH_ROUNDS"
require_uint "--warmup" "$BENCH_WARMUP"
[[ "$BENCH_ROUNDS" -gt 0 ]] || die "--rounds must be greater than zero"
[[ -n "$DEVICE_SET" ]] || die "--device is required outside a task-submit allocation"

IFS=',' read -r -a DEVICE_IDS <<<"$DEVICE_SET"
[[ "${#DEVICE_IDS[@]}" -eq 8 ]] || die "the fixed EP8 benchmark requires exactly eight device IDs"
declare -A SEEN_DEVICE_IDS=()
for device_id in "${DEVICE_IDS[@]}"; do
    [[ "$device_id" =~ ^[0-9]+$ ]] || die "invalid device ID: $device_id"
    [[ -z "${SEEN_DEVICE_IDS[$device_id]+present}" ]] || die "duplicate device ID: $device_id"
    SEEN_DEVICE_IDS["$device_id"]=1
done

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$REPO_ROOT/build_output/dsv4_eplb_perf/$(date -u '+%Y%m%dT%H%M%SZ')"
fi

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

resolve_ptoas() {
    if [[ -n "${PTOAS_ROOT:-}" && -x "$PTOAS_ROOT/ptoas" ]]; then
        printf '%s\n' "$PTOAS_ROOT/ptoas"
    elif [[ -n "${PTOAS_ROOT:-}" && -x "$PTOAS_ROOT/bin/ptoas" ]]; then
        printf '%s\n' "$PTOAS_ROOT/bin/ptoas"
    elif command -v ptoas >/dev/null 2>&1; then
        command -v ptoas
    fi
    return 0
}

CASE_COMMAND_ARGS=()

build_case_command() {
    local script="$1"
    CASE_COMMAND_ARGS=(
        "$PYTHON_BIN"
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
    if [[ "$script" == "eplb_mtp_core.py" ]]; then
        CASE_COMMAND_ARGS+=(--finite-only)
    fi
    if [[ "$COMPILE_ONLY" -eq 1 ]]; then
        CASE_COMMAND_ARGS+=(--compile-only)
    fi
}

if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'Repository SHA: %s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD)"
    printf 'Output: %s\n' "$OUTPUT_DIR"
    printf 'Benchmark environment: PYPTO_BENCH=1 PYPTO_BENCH_RAW=1 PYPTO_BENCH_ROUNDS=%s PYPTO_BENCH_WARMUP=%s\n' \
        "$BENCH_ROUNDS" "$BENCH_WARMUP"
    if [[ "$SELECTED_CASE" == "all" || "$SELECTED_CASE" == "decode-logits" ]]; then
        printf 'decode-logits:\n'
        build_case_command "eplb_decode_logits.py"
        print_command "${CASE_COMMAND_ARGS[@]}"
    fi
    if [[ "$SELECTED_CASE" == "all" || "$SELECTED_CASE" == "mtp-core" ]]; then
        printf 'mtp-core:\n'
        build_case_command "eplb_mtp_core.py"
        print_command "${CASE_COMMAND_ARGS[@]}"
    fi
    exit 0
fi

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BIN"
[[ ! -e "$OUTPUT_DIR" ]] || die "output directory already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
PTOAS_BIN="$(resolve_ptoas)"

git -C "$REPO_ROOT" status --short >"$OUTPUT_DIR/source-status.txt"
{
    printf 'started_at_utc\t%s\n' "$(date -u --iso-8601=seconds)"
    printf 'git_sha\t%s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD)"
    printf 'git_branch\t%s\n' "$(git -C "$REPO_ROOT" branch --show-current)"
    printf 'main_sha\t%s\n' "$(git -C "$REPO_ROOT" rev-parse upstream/main 2>/dev/null || printf 'unavailable')"
    printf 'platform\t%s\n' "$PLATFORM"
    printf 'device_set\t%s\n' "$DEVICE_SET"
    printf 'ep_size\t8\n'
    printf 'tp_size\t4\n'
    printf 'experts_per_rank\t16\n'
    printf 'start_pos\t8192\n'
    printf 'num_tokens\t8\n'
    printf 'rounds\t%s\n' "$BENCH_ROUNDS"
    printf 'warmup\t%s\n' "$BENCH_WARMUP"
    printf 'python\t%s\n' "$($PYTHON_BIN --version 2>&1)"
    if [[ -n "$PTOAS_BIN" ]]; then
        printf 'ptoas\t%s\n' "$($PTOAS_BIN --version 2>&1 | head -n 1)"
    else
        printf 'ptoas\tunavailable-on-PATH\n'
    fi
    if [[ -n "${PTO_ISA_ROOT:-}" ]] && git -C "$PTO_ISA_ROOT" rev-parse HEAD >/dev/null 2>&1; then
        printf 'pto_isa_sha\t%s\n' "$(git -C "$PTO_ISA_ROOT" rev-parse HEAD)"
    else
        printf 'pto_isa_sha\tunavailable\n'
    fi
} >"$OUTPUT_DIR/metadata.tsv"
printf 'case\tstatus\tmedian_us\tmean_us\tlog\n' >"$OUTPUT_DIR/results.tsv"

run_case() {
    local label="$1"
    local script="$2"
    local log_file="$OUTPUT_DIR/$label.log"
    local median_us="-"
    local mean_us="-"
    local status
    local rc

    build_case_command "$script"
    [[ -f "${CASE_COMMAND_ARGS[1]}" ]] || die "benchmark script not found: ${CASE_COMMAND_ARGS[1]}"

    printf '[RUN] %s\n' "$label"
    print_command "${CASE_COMMAND_ARGS[@]}"
    set +e
    PYPTO_BENCH=1 \
        PYPTO_BENCH_RAW=1 \
        PYPTO_BENCH_ROUNDS="$BENCH_ROUNDS" \
        PYPTO_BENCH_WARMUP="$BENCH_WARMUP" \
        "${CASE_COMMAND_ARGS[@]}" 2>&1 | tee "$log_file"
    rc="${PIPESTATUS[0]}"
    set -e

    if [[ "$rc" -ne 0 ]]; then
        status="fail"
    elif [[ "$COMPILE_ONLY" -eq 1 ]]; then
        status="pass_compile"
    else
        median_us="$(sed -nE 's/.*effective_us .*median=([0-9]+([.][0-9]+)?).*/\1/p' "$log_file" | tail -n 1)"
        mean_us="$(sed -nE 's/.*effective_us .*mean=([0-9]+([.][0-9]+)?).*/\1/p' "$log_file" | tail -n 1)"
        if [[ -n "$median_us" && -n "$mean_us" ]]; then
            status="pass"
        else
            status="missing_metric"
            median_us="${median_us:--}"
            mean_us="${mean_us:--}"
        fi
    fi

    printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$status" "$median_us" "$mean_us" "$(basename "$log_file")" \
        >>"$OUTPUT_DIR/results.tsv"
    [[ "$status" == "pass" || "$status" == "pass_compile" ]]
}

failures=0
if [[ "$SELECTED_CASE" == "all" || "$SELECTED_CASE" == "decode-logits" ]]; then
    run_case "decode-logits" "eplb_decode_logits.py" || failures=$((failures + 1))
fi
if [[ "$SELECTED_CASE" == "all" || "$SELECTED_CASE" == "mtp-core" ]]; then
    run_case "mtp-core" "eplb_mtp_core.py" || failures=$((failures + 1))
fi

printf 'finished_at_utc\t%s\n' "$(date -u --iso-8601=seconds)" >>"$OUTPUT_DIR/metadata.tsv"
printf 'Results: %s\n' "$OUTPUT_DIR/results.tsv"
if [[ "$failures" -ne 0 ]]; then
    printf 'ERROR: %s case(s) failed or emitted no timing metric.\n' "$failures" >&2
    exit 1
fi
