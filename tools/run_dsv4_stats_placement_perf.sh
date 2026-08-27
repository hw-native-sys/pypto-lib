#!/usr/bin/env bash
# Compare contiguous, stats-derived, and EPLB expert placement on DSV4 workloads.

set -euo pipefail

readonly SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
readonly REPO_ROOT="$(git -C "$(dirname "$SCRIPT_PATH")/.." rev-parse --show-toplevel)"
readonly MODEL_DIR="$REPO_ROOT/models/deepseek_v4_flash_mtp"
readonly METRIC_PARSER="$REPO_ROOT/tools/dsv4_eplb_perf_metrics.py"
readonly SEED_LAUNCHER="$REPO_ROOT/tools/run_seeded_python.py"
readonly DEFAULT_MANIFEST="$MODEL_DIR/stats_placement_decode_manifest.json"
readonly CANONICAL_DEVICE_SET="0,2,4,6,8,10,12,14"
readonly ALLOCATOR_DEVICE_SET="1,3,5,7,9,11,13,15"
readonly COMPARISON_CONTRACT_VERSION="dsv4-stats-placement-compare-v2"
readonly STATS_VS_EPLB_CONTRACT_VERSION="dsv4-stats-vs-eplb-compare-v2"
readonly METRIC_CONTRACT_VERSION="dsv4-stats-placement-numeric-v4"
readonly MOE_METRIC_CONTRACT_VERSION="dsv4-stats-placement-moe-ep8-v3"
readonly MOE_WORKLOAD_CONTRACT_VERSION="dsv4-stats-placement-moe-ep8x32-v1"
readonly EPLB_ALGORITHM="deepseek-eplb-balanced-packing-no-redundancy"
readonly EPLB_ALGORITHM_VERSION="d52c72d5b2f2fb4c41afbf8eb21366820239913d"
readonly EPLB_LOAD_SOURCE="replayed-logical-route-counts"
readonly EPLB_MAPPING_BASIS="current-replay-histogram"
readonly EPLB_EXPERT_ORDER="upstream-fp32-torch-sort-descending"
readonly EPLB_CONTROL_SCOPE="EP8x32 placement-only, not legacy EP8x16 branch baseline"
readonly EPLB_CONTROL_KIND="placement-quality-oracle"
readonly OFFICIAL_SEED="1807"
readonly OFFICIAL_ROUNDS="100"
readonly OFFICIAL_WARMUP="5"
readonly RING_TASK_WINDOW="262144"
readonly RING_DEP_POOL="262144"
readonly RING_HEAP="2147483648"

PLATFORM="a2a3"
DEVICE_SET="${TASK_DEVICE:-}"
OUTPUT_DIR=""
MANIFEST_PATH="$DEFAULT_MANIFEST"
SELECTED_CASE="all"
SELECTED_PLACEMENT="all"
PYTHON_BIN="${PYPTO_PERF_PYTHON:-python}"
BENCH_ROUNDS="${PYPTO_BENCH_ROUNDS:-$OFFICIAL_ROUNDS}"
BENCH_WARMUP="${PYPTO_BENCH_WARMUP:-$OFFICIAL_WARMUP}"
FIXTURE_SEED="${PYPTO_PERF_SEED:-$OFFICIAL_SEED}"
COMPILE_ONLY=0
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: tools/run_dsv4_stats_placement_perf.sh [OPTIONS]

Compare contiguous, stats-derived, and EPLB expert placement on identical
EP8x32 logical stats-shaped routes. All variants use the same manifest, seed,
topology, entrypoint, logical weights, and logical inputs. Physical route IDs
and expert-indexed tensors follow each placement. MTP reports its compute child
and excludes the cleanup child.
The comparison winner uses the largest slot-0 per-rank median, which exposes
persistent expert-load imbalance. Fastest-rank median remains a separate field
for the standard distributed-kernel convention.

Options:
  --device IDS          Must be 0,2,4,6,8,10,12,14 or 1,3,5,7,9,11,13,15.
                        Defaults to TASK_DEVICE.
  --manifest PATH       Placement manifest (default: checked-in decode manifest).
  --output-dir DIR      Result directory. Defaults below build_output/.
  --platform NAME       Must be a2a3 for measured runs (default: a2a3).
  --case NAME           all, moe-ep8, decode-logits, or mtp-core (default: all).
  --placement NAME      all, contiguous, stats, or eplb (default: all).
  --python PATH         Python executable (default: python).
  --rounds N            Must be 100 for measured runs (default: 100).
  --warmup N            Must be 5 for measured runs (default: 5).
  --seed N              Fixture seed for all three variants (default: 1807).
  --compile-only        Compile selected variants without requiring metrics.
  --dry-run             Print resolved commands without writing or running.
  -h, --help            Show this help.

comparison.tsv reports stats-minus-contiguous deltas and uses its v2
schema. stats-vs-eplb.tsv reports stats-minus-EPLB deltas. Negative latency
deltas mean that stats placement is faster; negative spread means it is more
balanced. EPLB solve and expert-weight movement are outside the timed interval.
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
        --manifest)
            [[ "$#" -ge 2 ]] || die "--manifest requires a value"
            MANIFEST_PATH="$2"
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
        --placement)
            [[ "$#" -ge 2 ]] || die "--placement requires a value"
            SELECTED_PLACEMENT="$2"
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
    all|moe-ep8|decode-logits|mtp-core) ;;
    *) die "--case must be all, moe-ep8, decode-logits, or mtp-core: $SELECTED_CASE" ;;
esac
case "$SELECTED_PLACEMENT" in
    all|contiguous|stats|eplb) ;;
    *) die "--placement must be all, contiguous, stats, or eplb: $SELECTED_PLACEMENT" ;;
esac

require_uint "--rounds" "$BENCH_ROUNDS"
require_uint "--warmup" "$BENCH_WARMUP"
require_uint "--seed" "$FIXTURE_SEED"
[[ "$BENCH_ROUNDS" -gt 0 ]] || die "--rounds must be greater than zero"
[[ -n "$DEVICE_SET" ]] || die "--device is required outside a task-submit allocation"
[[ "$DEVICE_SET" == "$CANONICAL_DEVICE_SET" || "$DEVICE_SET" == "$ALLOCATOR_DEVICE_SET" ]] || \
    die "the comparison metric requires --device $CANONICAL_DEVICE_SET or $ALLOCATOR_DEVICE_SET"
[[ -f "$MANIFEST_PATH" ]] || die "placement manifest not found: $MANIFEST_PATH"
MANIFEST_PATH="$(realpath "$MANIFEST_PATH")"
if [[ "$COMPILE_ONLY" -eq 0 ]]; then
    [[ "$PLATFORM" == "a2a3" ]] || die "measured comparisons require --platform a2a3"
    [[ "$BENCH_ROUNDS" == "$OFFICIAL_ROUNDS" ]] || \
        die "the EPLB metric parser requires --rounds $OFFICIAL_ROUNDS"
    [[ "$BENCH_WARMUP" == "$OFFICIAL_WARMUP" ]] || \
        die "the EPLB metric parser requires --warmup $OFFICIAL_WARMUP"
    [[ "$FIXTURE_SEED" == "$OFFICIAL_SEED" ]] || \
        die "the EPLB metric parser requires --seed $OFFICIAL_SEED"
fi

IFS=',' read -r -a DEVICE_IDS <<<"$DEVICE_SET"
[[ "${#DEVICE_IDS[@]}" -eq 8 ]] || die "the EP8 comparison requires exactly eight device IDs"

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$REPO_ROOT/build_output/dsv4_stats_placement_perf/$(date -u '+%Y%m%dT%H%M%SZ')"
fi

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

CASE_COMMAND_ARGS=()

build_case_command() {
    local case_name="$1"
    local placement="$2"
    local script
    if [[ "$case_name" == "moe-ep8" ]]; then
        script="$MODEL_DIR/stats_placement_moe.py"
    elif [[ "$case_name" == "decode-logits" ]]; then
        script="$MODEL_DIR/stats_placement_decode_logits.py"
    else
        script="$MODEL_DIR/stats_placement_mtp_core.py"
    fi
    CASE_COMMAND_ARGS=(
        "$PYTHON_BIN"
        "$SEED_LAUNCHER"
        --seed "$FIXTURE_SEED"
        --
        "$script"
        --expert-placement "$placement"
        --placement-manifest "$MANIFEST_PATH"
        -p "$PLATFORM"
        -d "$DEVICE_SET"
        --ep 8
        --experts-per-rank 32
    )
    if [[ "$case_name" == "moe-ep8" ]]; then
        CASE_COMMAND_ARGS+=(
            --layer-id 0
            --num-tokens 8
            --enable-l2-swimlane 0
        )
    else
        CASE_COMMAND_ARGS+=(
            --tp 4
            --start-pos 8192
            --num-tokens 8
            --enable-l2-swimlane 0
        )
    fi
    if [[ "$COMPILE_ONLY" -eq 1 ]]; then
        CASE_COMMAND_ARGS+=(--compile-only)
    fi
}

placement_selected() {
    local placement="$1"
    [[ "$SELECTED_PLACEMENT" == "all" || "$SELECTED_PLACEMENT" == "$placement" ]]
}

case_selected() {
    local case_name="$1"
    [[ "$SELECTED_CASE" == "all" || "$SELECTED_CASE" == "$case_name" ]]
}

print_selected_commands() {
    local case_name
    local placement
    for case_name in moe-ep8 decode-logits mtp-core; do
        case_selected "$case_name" || continue
        for placement in contiguous stats eplb; do
            placement_selected "$placement" || continue
            printf '%s-%s:\n' "$case_name" "$placement"
            build_case_command "$case_name" "$placement"
            print_command "${CASE_COMMAND_ARGS[@]}"
        done
    done
}

if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'Repository SHA: %s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD)"
    printf 'Output: %s\n' "$OUTPUT_DIR"
    printf 'Comparison contract: %s\n' "$COMPARISON_CONTRACT_VERSION"
    printf 'Stats-vs-EPLB contract: %s\n' "$STATS_VS_EPLB_CONTRACT_VERSION"
    printf 'Metric contract: %s\n' "$METRIC_CONTRACT_VERSION"
    printf 'MoE metric contract: %s\n' "$MOE_METRIC_CONTRACT_VERSION"
    printf 'MoE workload contract: %s\n' "$MOE_WORKLOAD_CONTRACT_VERSION"
    printf 'Manifest: %s\n' "$MANIFEST_PATH"
    printf 'Comparison: same EP8x32 stats-shaped routes; only expert placement changes\n'
    printf 'Logical workload: same routes, weights, and inputs; physical expert IDs and tensors follow placement\n'
    printf 'EPLB algorithm: %s at %s\n' "$EPLB_ALGORITHM" "$EPLB_ALGORITHM_VERSION"
    printf 'EPLB expert order: %s\n' "$EPLB_EXPERT_ORDER"
    printf 'EPLB timing: solve_in_timed_region=false weight_migration_in_timed_region=false\n'
    printf 'EPLB control kind: %s\n' "$EPLB_CONTROL_KIND"
    printf 'Benchmark environment: PYTHONHASHSEED=%s PYPTO_BENCH=1 PYPTO_BENCH_RAW=1 PYPTO_BENCH_ROUNDS=%s PYPTO_BENCH_WARMUP=%s\n' \
        "$FIXTURE_SEED" "$BENCH_ROUNDS" "$BENCH_WARMUP"
    printf 'Validation:'
    if case_selected "moe-ep8"; then
        printf ' moe-ep8=numeric_golden'
    fi
    if case_selected "decode-logits"; then
        printf ' decode-logits=numeric_golden'
    fi
    if case_selected "mtp-core"; then
        printf ' mtp-core=numeric_golden'
    fi
    printf ' golden_replayed=false\n'
    print_selected_commands
    exit 0
fi

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BIN"
[[ -f "$METRIC_PARSER" ]] || die "metric parser not found: $METRIC_PARSER"
[[ -f "$SEED_LAUNCHER" ]] || die "seed launcher not found: $SEED_LAUNCHER"
[[ ! -e "$OUTPUT_DIR" ]] || die "output directory already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"

git -C "$REPO_ROOT" status --short >"$OUTPUT_DIR/source-status.txt"
{
    printf 'started_at_utc\t%s\n' "$(date -u --iso-8601=seconds)"
    printf 'git_sha\t%s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD)"
    printf 'git_branch\t%s\n' "$(git -C "$REPO_ROOT" branch --show-current)"
    printf 'comparison_contract_version\t%s\n' "$COMPARISON_CONTRACT_VERSION"
    printf 'stats_vs_eplb_contract_version\t%s\n' "$STATS_VS_EPLB_CONTRACT_VERSION"
    printf 'metric_contract_version\t%s\n' "$METRIC_CONTRACT_VERSION"
    printf 'moe_metric_contract_version\t%s\n' "$MOE_METRIC_CONTRACT_VERSION"
    printf 'moe_workload_contract_version\t%s\n' "$MOE_WORKLOAD_CONTRACT_VERSION"
    printf 'comparison_case_set\tmoe-ep8,decode-logits,mtp-core\n'
    printf 'manifest\t%s\n' "$MANIFEST_PATH"
    printf 'manifest_sha256\t%s\n' "$(sha256sum "$MANIFEST_PATH" | awk '{print $1}')"
    printf 'platform\t%s\n' "$PLATFORM"
    printf 'device_set\t%s\n' "$DEVICE_SET"
    printf 'ep_size\t8\n'
    printf 'tp_size\t4\n'
    printf 'moe_tp_size\tnot_applicable\n'
    printf 'moe_layer_id\t0\n'
    printf 'experts_per_rank\t32\n'
    printf 'start_pos\t8192\n'
    printf 'num_tokens\t8\n'
    printf 'rounds\t%s\n' "$BENCH_ROUNDS"
    printf 'warmup\t%s\n' "$BENCH_WARMUP"
    printf 'fixture_seed\t%s\n' "$FIXTURE_SEED"
    printf 'fastest_metric\tminimum slot-0 per-rank median from the case-specific metric contract\n'
    printf 'comparison_winner_basis\tmaximum slot-0 per-rank median\n'
    printf 'rank_spread_basis\tmaximum minus minimum slot-0 per-rank median\n'
    printf 'decode_validation_mode\tnumeric_golden\n'
    printf 'moe_validation_mode\tnumeric_golden\n'
    printf 'mtp_validation_mode\tnumeric_golden\n'
    printf 'golden_replayed\tfalse\n'
    printf 'delta_convention\tstats minus contiguous; negative is faster\n'
    printf 'stats_vs_eplb_delta_convention\tstats minus eplb; negative is faster\n'
    printf 'eplb_algorithm\t%s\n' "$EPLB_ALGORITHM"
    printf 'eplb_algorithm_version\t%s\n' "$EPLB_ALGORITHM_VERSION"
    printf 'eplb_load_source\t%s\n' "$EPLB_LOAD_SOURCE"
    printf 'eplb_mapping_basis\t%s\n' "$EPLB_MAPPING_BASIS"
    printf 'eplb_expert_order\t%s\n' "$EPLB_EXPERT_ORDER"
    printf 'eplb_solver_torch_version\t%s\n' "$("$PYTHON_BIN" -c 'import torch; print(torch.__version__)')"
    printf 'eplb_redundant_experts\t0\n'
    printf 'eplb_solve_in_timed_region\tfalse\n'
    printf 'eplb_weight_migration_in_timed_region\tfalse\n'
    printf 'eplb_control_scope\t%s\n' "$EPLB_CONTROL_SCOPE"
    printf 'eplb_control_kind\t%s\n' "$EPLB_CONTROL_KIND"
    printf 'python\t%s\n' "$("$PYTHON_BIN" --version 2>&1)"
} >"$OUTPUT_DIR/metadata.tsv"

printf '%s\n' \
    $'variant\tcase\tplacement\tstatus\tvalidation_mode\tgolden_replayed\tprocess_rc\tmetric_contract_version\tmetric_scope\tselection_policy\trounds\twarmup\trank_count\tdispatches_per_round\tselected_rank\tselected_device\tselected_pid\tsamples\tmin_us\tmedian_us\tmean_us\tmax_us\tcleanup_median_us\tbaseline_median_us\tdelta_us\tdelta_pct\tmetric_source\tmapping_basis\tlog' \
    >"$OUTPUT_DIR/results.tsv"
printf '%s\n' \
    $'case\tcontiguous_fastest_rank_median_us\tstats_fastest_rank_median_us\tfastest_stats_minus_contiguous_us\tfastest_stats_minus_contiguous_pct\tcontiguous_max_rank_median_us\tstats_max_rank_median_us\tmax_rank_stats_minus_contiguous_us\tmax_rank_stats_minus_contiguous_pct\tcontiguous_rank_median_spread_us\tstats_rank_median_spread_us\tspread_stats_minus_contiguous_us\twinner_by_max_rank_median\tvalidation_mode\tgolden_replayed' \
    >"$OUTPUT_DIR/comparison.tsv"
printf '%s\n' \
    $'case\teplb_fastest_rank_median_us\tstats_fastest_rank_median_us\tfastest_stats_minus_eplb_us\tfastest_stats_minus_eplb_pct\teplb_max_rank_median_us\tstats_max_rank_median_us\tmax_rank_stats_minus_eplb_us\tmax_rank_stats_minus_eplb_pct\teplb_rank_median_spread_us\tstats_rank_median_spread_us\tspread_stats_minus_eplb_us\twinner_by_max_rank_median\tvalidation_mode\tgolden_replayed' \
    >"$OUTPUT_DIR/stats-vs-eplb.tsv"
printf '%s\n' \
    $'case\tplacement\tfastest_rank_median_us\tmax_rank_median_us\trank_median_spread_us' \
    >"$OUTPUT_DIR/rank-balance.tsv"

readonly EMPTY_METRIC_FIELDS=$'-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-'

run_case() {
    local case_name="$1"
    local placement="$2"
    local variant="$case_name-$placement"
    local log_file="$OUTPUT_DIR/$variant.log"
    local rank_file="$OUTPUT_DIR/$variant.rank-results.tsv"
    local metric_error_file="$OUTPUT_DIR/$variant.metric-parser.stderr"
    local metric_fields="$EMPTY_METRIC_FIELDS"
    local metric_output=""
    local validation_mode="numeric_golden"
    local status
    local rc

    build_case_command "$case_name" "$placement"
    printf '[RUN] %s\n' "$variant"
    print_command "${CASE_COMMAND_ARGS[@]}"
    set +e
    PYPTO_BENCH=1 \
        PYPTO_BENCH_RAW=1 \
        PYPTO_BENCH_ROUNDS="$BENCH_ROUNDS" \
        PYPTO_BENCH_WARMUP="$BENCH_WARMUP" \
        PYPTO_RUNTIME_LOG=error \
        PYTHONHASHSEED="$FIXTURE_SEED" \
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
        printf '%s\n' \
            $'case\tmetric_scope\tlogical_rank\tdevice_id\tpid\tslot\ttask\tselected\tsamples\tmin_us\tmedian_us\tmean_us\tmax_us' \
            >"$rank_file"
        set +e
        metric_output="$(
            "$PYTHON_BIN" "$METRIC_PARSER" \
                --case "$case_name" \
                --log "$log_file" \
                --rounds "$BENCH_ROUNDS" \
                --warmup "$BENCH_WARMUP" \
                --device "$DEVICE_SET" \
                --seed "$FIXTURE_SEED" \
                --validation-profile stats-placement-numeric \
                --rank-output "$rank_file" \
                2>"$metric_error_file"
        )"
        local metric_rc="$?"
        set -e
        if [[ "$metric_rc" -ne 0 || -z "$metric_output" || "$metric_output" == *$'\n'* ]]; then
            status="invalid_metric"
            if [[ "$rc" -ne 0 ]]; then
                status="fail"
            fi
            if [[ -s "$metric_error_file" ]]; then
                sed 's/^/[METRIC STDERR] /' "$metric_error_file" >&2
            fi
        else
            local -a metric_columns=()
            IFS=$'\t' read -r -a metric_columns <<<"$metric_output"
            if [[ "${#metric_columns[@]}" -ne 21 ]]; then
                status="invalid_metric"
            elif [[ "$rc" -eq 0 ]]; then
                status="pass"
                metric_fields="$metric_output"
                printf '[METRIC] %s\n' "$metric_output"
            else
                status="metric_valid_execution_failed"
                metric_fields="$metric_output"
            fi
        fi
    fi

    printf '%s\t%s\t%s\t%s\t%s\tfalse\t%s\t%s\t%s\n' \
        "$variant" "$case_name" "$placement" "$status" "$validation_mode" "$rc" \
        "$metric_fields" "$(basename "$log_file")" \
        >>"$OUTPUT_DIR/results.tsv"
    [[ "$status" == "pass" || "$status" == "pass_compile" ]]
}

failures=0
for case_name in moe-ep8 decode-logits mtp-core; do
    case_selected "$case_name" || continue
    for placement in contiguous stats eplb; do
        placement_selected "$placement" || continue
        run_case "$case_name" "$placement" || failures=$((failures + 1))
    done
done

if [[ "$COMPILE_ONLY" -eq 0 ]]; then
    for case_name in moe-ep8 decode-logits mtp-core; do
        case_selected "$case_name" || continue
        for placement in contiguous stats eplb; do
            placement_selected "$placement" || continue
            variant="$case_name-$placement"
            if ! awk -F '\t' -v case_name="$case_name" -v placement="$placement" '
                NR > 1 && $6 == 0 {
                    median = $11 + 0.0
                    if (count == 0 || median < minimum) {
                        minimum = median
                    }
                    if (count == 0 || median > maximum) {
                        maximum = median
                    }
                    count++
                }
                END {
                    if (count != 8) {
                        printf "expected 8 slot-0 rank medians, got %d\n", count > "/dev/stderr"
                        exit 1
                    }
                    printf "%s\t%s\t%.3f\t%.3f\t%.3f\n", \
                        case_name, placement, minimum, maximum, maximum - minimum
                }
            ' "$OUTPUT_DIR/$variant.rank-results.tsv" >>"$OUTPUT_DIR/rank-balance.tsv"; then
                printf 'ERROR: invalid per-rank balance metrics for %s.\n' "$variant" >&2
                failures=$((failures + 1))
            fi
        done
    done

    awk -F '\t' '
        NR > 1 {
            key = $1 SUBSEP $2
            fastest[key] = $3
            maximum[key] = $4
            spread[key] = $5
        }
        END {
            split("moe-ep8 decode-logits mtp-core", cases, " ")
            for (case_index = 1; case_index <= 3; case_index++) {
                name = cases[case_index]
                contiguous_key = name SUBSEP "contiguous"
                stats_key = name SUBSEP "stats"
                if (!(contiguous_key in fastest) || !(stats_key in fastest)) {
                    continue
                }
                contiguous_fastest = fastest[contiguous_key] + 0.0
                stats_fastest = fastest[stats_key] + 0.0
                fastest_delta = stats_fastest - contiguous_fastest
                fastest_pct = contiguous_fastest == 0.0 ? \
                    0.0 : fastest_delta / contiguous_fastest * 100.0
                contiguous_maximum = maximum[contiguous_key] + 0.0
                stats_maximum = maximum[stats_key] + 0.0
                maximum_delta = stats_maximum - contiguous_maximum
                maximum_pct = contiguous_maximum == 0.0 ? \
                    0.0 : maximum_delta / contiguous_maximum * 100.0
                contiguous_spread = spread[contiguous_key] + 0.0
                stats_spread = spread[stats_key] + 0.0
                spread_delta = stats_spread - contiguous_spread
                winner = maximum_delta < 0.0 ? \
                    "stats" : (maximum_delta > 0.0 ? "contiguous" : "tie")
                validation_mode = "numeric_golden"
                printf "%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s\t%s\tfalse\n", \
                    name, contiguous_fastest, stats_fastest, fastest_delta, fastest_pct, \
                    contiguous_maximum, stats_maximum, maximum_delta, maximum_pct, \
                    contiguous_spread, stats_spread, spread_delta, winner, validation_mode
            }
        }
    ' "$OUTPUT_DIR/rank-balance.tsv" >>"$OUTPUT_DIR/comparison.tsv"

    awk -F '\t' '
        NR > 1 {
            key = $1 SUBSEP $2
            fastest[key] = $3
            maximum[key] = $4
            spread[key] = $5
        }
        END {
            split("moe-ep8 decode-logits mtp-core", cases, " ")
            for (case_index = 1; case_index <= 3; case_index++) {
                name = cases[case_index]
                eplb_key = name SUBSEP "eplb"
                stats_key = name SUBSEP "stats"
                if (!(eplb_key in fastest) || !(stats_key in fastest)) {
                    continue
                }
                eplb_fastest = fastest[eplb_key] + 0.0
                stats_fastest = fastest[stats_key] + 0.0
                fastest_delta = stats_fastest - eplb_fastest
                fastest_pct = eplb_fastest == 0.0 ? \
                    0.0 : fastest_delta / eplb_fastest * 100.0
                eplb_maximum = maximum[eplb_key] + 0.0
                stats_maximum = maximum[stats_key] + 0.0
                maximum_delta = stats_maximum - eplb_maximum
                maximum_pct = eplb_maximum == 0.0 ? \
                    0.0 : maximum_delta / eplb_maximum * 100.0
                eplb_spread = spread[eplb_key] + 0.0
                stats_spread = spread[stats_key] + 0.0
                spread_delta = stats_spread - eplb_spread
                winner = maximum_delta < 0.0 ? \
                    "stats" : (maximum_delta > 0.0 ? "eplb" : "tie")
                validation_mode = "numeric_golden"
                printf "%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s\t%s\tfalse\n", \
                    name, eplb_fastest, stats_fastest, fastest_delta, fastest_pct, \
                    eplb_maximum, stats_maximum, maximum_delta, maximum_pct, \
                    eplb_spread, stats_spread, spread_delta, winner, validation_mode
            }
        }
    ' "$OUTPUT_DIR/rank-balance.tsv" >>"$OUTPUT_DIR/stats-vs-eplb.tsv"
fi

printf 'finished_at_utc\t%s\n' "$(date -u --iso-8601=seconds)" >>"$OUTPUT_DIR/metadata.tsv"
printf 'Results: %s\n' "$OUTPUT_DIR/results.tsv"
printf 'Comparison: %s\n' "$OUTPUT_DIR/comparison.tsv"
printf 'Stats vs EPLB: %s\n' "$OUTPUT_DIR/stats-vs-eplb.tsv"
if [[ "$failures" -ne 0 ]]; then
    printf 'ERROR: %s variant(s) failed execution or metric validation.\n' "$failures" >&2
    exit 1
fi
