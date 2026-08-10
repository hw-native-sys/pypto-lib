#!/usr/bin/env bash
# Run the fixed DeepSeek-V4 EPLB benchmarks without changing Git state.

set -euo pipefail

readonly SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
readonly REPO_ROOT="$(git -C "$(dirname "$SCRIPT_PATH")/.." rev-parse --show-toplevel)"
readonly METRIC_PARSER="$REPO_ROOT/tools/dsv4_eplb_perf_metrics.py"
readonly CANONICAL_DEVICE_SET="0,2,4,6,8,10,12,14"
readonly METRIC_CONTRACT_VERSION="dsv4-eplb-v1"
readonly DECODE_LOGITS_BASELINE_US="36144.680"
readonly MTP_CORE_BASELINE_US="1162.050"
readonly RING_TASK_WINDOW="262144"
readonly RING_DEP_POOL="262144"
readonly RING_HEAP="2147483648"

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
The MTP-core case performs finite-output validation before timing. Results use
the Compare3/Compare4 contract: select the smallest per-rank median, and for
MTP-core measure slot 0 compute samples only while excluding slot 1 cleanup.

Options:
  --device IDS          Must be 0,2,4,6,8,10,12,14. Defaults to TASK_DEVICE.
  --output-dir DIR      Result directory. Defaults below build_output/.
  --platform NAME       Must be a2a3 for measured runs (default: a2a3).
  --case NAME           all, decode-logits, or mtp-core (default: all).
  --python PATH         Python executable (default: python).
  --rounds N            Must be 100 for measured runs (default: 100).
  --warmup N            Must be 5 for measured runs (default: 5).
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
[[ "$DEVICE_SET" == "$CANONICAL_DEVICE_SET" ]] || \
    die "official EPLB metrics require --device $CANONICAL_DEVICE_SET"
if [[ "$COMPILE_ONLY" -eq 0 ]]; then
    [[ "$PLATFORM" == "a2a3" ]] || die "official EPLB metrics require --platform a2a3"
    [[ "$BENCH_ROUNDS" == "100" ]] || die "official EPLB metrics require --rounds 100"
    [[ "$BENCH_WARMUP" == "5" ]] || die "official EPLB metrics require --warmup 5"
fi

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
    if [[ -n "${PTOAS_ROOT:-}" && -f "$PTOAS_ROOT/ptoas" && -x "$PTOAS_ROOT/ptoas" ]]; then
        printf '%s\n' "$PTOAS_ROOT/ptoas"
    elif [[ -n "${PTOAS_ROOT:-}" && -f "$PTOAS_ROOT/bin/ptoas" && -x "$PTOAS_ROOT/bin/ptoas" ]]; then
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
    printf 'Metric contract: %s (Compare3=%s us, Compare4=%s us)\n' \
        "$METRIC_CONTRACT_VERSION" "$DECODE_LOGITS_BASELINE_US" "$MTP_CORE_BASELINE_US"
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
[[ -f "$METRIC_PARSER" ]] || die "metric parser not found: $METRIC_PARSER"
[[ ! -e "$OUTPUT_DIR" ]] || die "output directory already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
PTOAS_BIN="$(resolve_ptoas)"
UPSTREAM_MAIN_SHA="$(git -C "$REPO_ROOT" rev-parse upstream/main 2>/dev/null || printf 'unavailable')"
BASE_MAIN_SHA="$(git -C "$REPO_ROOT" merge-base HEAD upstream/main 2>/dev/null || printf 'unavailable')"
read -r COMMITS_AHEAD COMMITS_BEHIND < <(
    git -C "$REPO_ROOT" rev-list --left-right --count HEAD...upstream/main 2>/dev/null || printf 'unavailable unavailable\n'
)

git -C "$REPO_ROOT" status --short >"$OUTPUT_DIR/source-status.txt"
{
    printf 'started_at_utc\t%s\n' "$(date -u --iso-8601=seconds)"
    printf 'git_sha\t%s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD)"
    printf 'git_tree\t%s\n' "$(git -C "$REPO_ROOT" rev-parse 'HEAD^{tree}')"
    printf 'git_branch\t%s\n' "$(git -C "$REPO_ROOT" branch --show-current)"
    printf 'upstream_main_sha\t%s\n' "$UPSTREAM_MAIN_SHA"
    printf 'base_main_sha\t%s\n' "$BASE_MAIN_SHA"
    printf 'commits_ahead\t%s\n' "$COMMITS_AHEAD"
    printf 'commits_behind\t%s\n' "$COMMITS_BEHIND"
    printf 'metric_contract_version\t%s\n' "$METRIC_CONTRACT_VERSION"
    printf 'decode_logits_baseline_us\t%s\n' "$DECODE_LOGITS_BASELINE_US"
    printf 'mtp_core_baseline_us\t%s\n' "$MTP_CORE_BASELINE_US"
    printf 'platform\t%s\n' "$PLATFORM"
    printf 'device_set\t%s\n' "$DEVICE_SET"
    printf 'ep_size\t8\n'
    printf 'tp_size\t4\n'
    printf 'experts_per_rank\t16\n'
    printf 'start_pos\t8192\n'
    printf 'num_tokens\t8\n'
    printf 'rounds\t%s\n' "$BENCH_ROUNDS"
    printf 'warmup\t%s\n' "$BENCH_WARMUP"
    printf 'pto2_ring_task_window\t%s\n' "$RING_TASK_WINDOW"
    printf 'pto2_ring_dep_pool\t%s\n' "$RING_DEP_POOL"
    printf 'pto2_ring_heap\t%s\n' "$RING_HEAP"
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
printf '%s\n' \
    $'case\tstatus\tprocess_rc\tmetric_contract_version\tmetric_scope\tselection_policy\trounds\twarmup\trank_count\tdispatches_per_round\tselected_rank\tselected_device\tselected_pid\tsamples\tmin_us\tmedian_us\tmean_us\tmax_us\tcleanup_median_us\tbaseline_median_us\tdelta_us\tdelta_pct\tmetric_source\tmapping_basis\tlog' \
    >"$OUTPUT_DIR/results.tsv"
printf '%s\n' \
    $'case\tmetric_scope\tlogical_rank\tdevice_id\tpid\tslot\ttask\tselected\tsamples\tmin_us\tmedian_us\tmean_us\tmax_us' \
    >"$OUTPUT_DIR/rank-results.tsv"

readonly EMPTY_METRIC_FIELDS=$'-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-'

run_case() {
    local label="$1"
    local script="$2"
    local log_file="$OUTPUT_DIR/$label.log"
    local metric_error_file="$OUTPUT_DIR/$label.metric-parser.stderr"
    local metric_fields="$EMPTY_METRIC_FIELDS"
    local metric_validation_error=""
    local -a metric_columns=()
    local metric_output
    local metric_rc
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
        PYPTO_RUNTIME_LOG=error \
        SIMPLER_DEVICE_STRACE_ENABLE=1 \
        PTO2_RING_TASK_WINDOW="$RING_TASK_WINDOW" \
        PTO2_RING_DEP_POOL="$RING_DEP_POOL" \
        PTO2_RING_HEAP="$RING_HEAP" \
        "${CASE_COMMAND_ARGS[@]}" 2>&1 | tee "$log_file"
    rc="${PIPESTATUS[0]}"
    set -e

    if [[ "$COMPILE_ONLY" -eq 1 ]]; then
        if [[ "$rc" -eq 0 ]]; then
            status="pass_compile"
        else
            status="fail"
        fi
    else
        set +e
        metric_output="$(
            "$PYTHON_BIN" "$METRIC_PARSER" \
                --case "$label" \
                --log "$log_file" \
                --rounds "$BENCH_ROUNDS" \
                --warmup "$BENCH_WARMUP" \
                --device "$DEVICE_SET" \
                --rank-output "$OUTPUT_DIR/rank-results.tsv" \
                2>"$metric_error_file"
        )"
        metric_rc="$?"
        set -e
        if [[ -s "$metric_error_file" ]]; then
            printf '[METRIC STDERR] %s\n' "$label" >&2
            sed 's/^/  /' "$metric_error_file" >&2
        fi
        if [[ "$metric_rc" -ne 0 ]]; then
            metric_validation_error="metric parser exited with status $metric_rc"
        elif [[ -z "$metric_output" ]]; then
            metric_validation_error="metric parser emitted an empty summary"
        elif [[ "$metric_output" == *$'\n'* ]]; then
            metric_validation_error="metric parser emitted more than one summary line"
        else
            IFS=$'\t' read -r -a metric_columns <<<"$metric_output"
            if [[ "${#metric_columns[@]}" -ne 21 ]]; then
                metric_validation_error="metric parser emitted ${#metric_columns[@]} fields, expected 21"
            fi
        fi
        if [[ -n "$metric_validation_error" ]]; then
            printf 'ERROR: %s: %s\n' "$label" "$metric_validation_error" >&2
            if [[ "$rc" -eq 0 ]]; then
                status="invalid_metric"
            else
                status="fail"
            fi
        else
            metric_fields="$metric_output"
            printf '[METRIC] %s\n' "$metric_output"
            if [[ "$rc" -eq 0 ]]; then
                status="pass"
            else
                status="metric_valid_execution_failed"
            fi
        fi
    fi

    printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$status" "$rc" "$metric_fields" "$(basename "$log_file")" \
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
    printf 'ERROR: %s case(s) failed execution or official metric validation.\n' "$failures" >&2
    exit 1
fi
