#!/usr/bin/env bash
#
# Rebase the DeepSeek V4 performance branch onto upstream main, publish the
# rebased branch with force-with-lease, and run the requested NPU benchmarks.
#
# The scheduled run uses an isolated temporary clone. It never rebases, resets,
# or cleans the developer checkout containing this script.

set -euo pipefail

readonly SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
readonly REPO_ROOT="$(git -C "$(dirname "$SCRIPT_PATH")/.." rev-parse --show-toplevel)"
readonly PERF_BRANCH="${PYPTO_DAILY_PERF_BRANCH:-perf/dsv4-eplb-benchmark}"
readonly MAIN_BRANCH="${PYPTO_DAILY_MAIN_BRANCH:-main}"
readonly REMOTE_NAME="${PYPTO_DAILY_REMOTE_NAME:-upstream}"
readonly DEFAULT_REMOTE_URL="git@github.com:hw-native-sys/pypto-lib.git"
readonly CONDA_BASE="${PYPTO_DAILY_CONDA_BASE:-$(conda info --base 2>/dev/null || true)}"
readonly CONDA_ENV="${PYPTO_DAILY_CONDA_ENV:-wq3}"
readonly CANN_ROOT="${PYPTO_DAILY_CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}"
readonly PTOAS_BIN="${PYPTO_DAILY_PTOAS_BIN:-/usr/local/bin/ptoas-bin}"
readonly PTO_ISA_DIR="${PYPTO_DAILY_PTO_ISA_ROOT:-$(realpath "$REPO_ROOT/../pto-isa")}"
readonly ATTENTION_DEVICE="${PYPTO_DAILY_ATTENTION_DEVICE:-4}"
readonly MOE_DEVICE="${PYPTO_DAILY_MOE_DEVICE:-2,0}"
readonly DECODE_DEVICE="${PYPTO_DAILY_DECODE_DEVICE:-0,2}"
readonly DEFAULT_SCHEDULE_HOUR="${PYPTO_DAILY_PERF_HOUR:-05}"

readonly STATE_ROOT="${PYPTO_DAILY_PERF_STATE_ROOT:-$REPO_ROOT/.cache/dsv4-daily-perf}"
readonly RESULT_ROOT="$STATE_ROOT/results"
readonly CRON_LOG="$STATE_ROOT/cron.log"
readonly CRON_BEGIN="# BEGIN pypto-lib dsv4 daily performance"
readonly CRON_END="# END pypto-lib dsv4 daily performance"

export TZ=Asia/Shanghai
export PATH="$PTOAS_BIN:$CONDA_BASE/condabin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"
export PTOAS_ROOT="$PTOAS_BIN"
export PTO_ISA_ROOT="$PTO_ISA_DIR"
export PTO2_RING_TASK_WINDOW=262144
export PTO2_RING_DEP_POOL=262144
export PTO2_RING_HEAP=2147483648
export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -o BatchMode=yes}"

RUN_TEMP_ROOT=""

cleanup_temp_checkout() {
    case "${RUN_TEMP_ROOT:-}" in
        "$STATE_ROOT"/tmp/run.*)
            rm -rf -- "$RUN_TEMP_ROOT"
            ;;
    esac
}

trap cleanup_temp_checkout EXIT

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"
}

die() {
    log "ERROR: $*"
    exit 1
}

usage() {
    cat <<EOF
Usage:
  $SCRIPT_PATH run
  $SCRIPT_PATH scheduled
  $SCRIPT_PATH dry-run
  $SCRIPT_PATH install-cron [SHANGHAI_HOUR]
  $SCRIPT_PATH remove-cron

Commands:
  run           Rebase, push, and execute all benchmarks now.
  scheduled     Run only when the current Shanghai hour matches the configured
                schedule hour. This is the entry point installed in crontab.
  dry-run       Print the resolved configuration and benchmark commands.
  install-cron  Install the daily cron entry; default hour is 05:00 Shanghai.
  remove-cron   Remove only this script's managed cron block.

Environment overrides:
  PYPTO_DAILY_ATTENTION_DEVICE  Single card for CSA/SWA/HCA (default: 4)
  PYPTO_DAILY_MOE_DEVICE        Explicit EP2 card pair (default: 2,0)
  PYPTO_DAILY_DECODE_DEVICE     Full-decode card pair (default: 0,2)
  PYPTO_DAILY_CANN_ROOT         CANN installation (default: /usr/local/Ascend/cann-9.0.0)
  PYPTO_DAILY_PERF_HOUR         Shanghai schedule hour (default: 05)
  PYPTO_DAILY_PERF_STATE_ROOT   Persistent result and scheduler state directory
EOF
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

remote_url() {
    git -C "$REPO_ROOT" remote get-url "$REMOTE_NAME" 2>/dev/null || printf '%s\n' "$DEFAULT_REMOTE_URL"
}

activate_environment() {
    local conda_sh="$CONDA_BASE/etc/profile.d/conda.sh"
    local cann_set_env="$CANN_ROOT/set_env.sh"

    [[ -f "$conda_sh" ]] || die "Conda activation script not found: $conda_sh"
    [[ -f "$cann_set_env" ]] || die "CANN environment script not found: $cann_set_env"
    [[ -d "$PTOAS_BIN" ]] || die "PTOAS directory not found: $PTOAS_BIN"
    [[ -d "$PTO_ISA_DIR" ]] || die "PTO ISA directory not found: $PTO_ISA_DIR"

    # Validate the child environment without leaking Conda's libraries into
    # the parent Git/SSH process.
    (
        set +u
        # shellcheck disable=SC1090
        source "$conda_sh"
        conda activate "$CONDA_ENV"
        # shellcheck disable=SC1090
        source "$cann_set_env"
        export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        python -c 'import ctypes; ctypes.CDLL("libhccl.so")'
    )

    require_command git
    require_command task-submit
    require_command flock
}

print_case_plan() {
    printf '%-24s ptoas=%-4s device=%-5s %s\n' \
        "decode_attention_csa_8k" "0.48" "$ATTENTION_DEVICE" \
        'python models/deepseek/v4-flash/decode_attention_csa.py -p a2a3 -d "$TASK_DEVICE" --start-pos 8192 --enable-l2-swimlane 0'
    printf '%-24s ptoas=%-4s device=%-5s %s\n' \
        "decode_attention_swa_8k" "0.48" "$ATTENTION_DEVICE" \
        'python models/deepseek/v4-flash/decode_attention_swa.py -p a2a3 -d "$TASK_DEVICE" --start-pos 8192 --enable-l2-swimlane 0'
    printf '%-24s ptoas=%-4s device=%-5s %s\n' \
        "decode_attention_hca_8k" "0.48" "$ATTENTION_DEVICE" \
        'python models/deepseek/v4-flash/decode_attention_hca.py -p a2a3 -d "$TASK_DEVICE" --start-pos 8192 --enable-l2-swimlane 0'
    printf '%-24s ptoas=%-4s device=%-5s %s\n' \
        "moe_ep2" "0.54" "$MOE_DEVICE" \
        'python models/deepseek/v4-flash/moe.py -p a2a3 -d "$TASK_DEVICE" --ep 2 --enable-l2-swimlane 1'
    printf '%-24s ptoas=%-4s device=%-5s %s\n' \
        "decode_fwd_43l_ep2_8k" "0.54" "$DECODE_DEVICE" \
        'python models/deepseek/v4-flash/decode_fwd.py -p a2a3 --ep 2 --tp 2 -d "$TASK_DEVICE" --start-pos 8192 --num-tokens 8 --enable-l2-swimlane 0'
}

dry_run() {
    local url
    url="$(remote_url)"

    printf 'Repository:       %s\n' "$REPO_ROOT"
    printf 'Remote:           %s (%s)\n' "$REMOTE_NAME" "$url"
    printf 'Performance ref:  %s\n' "$PERF_BRANCH"
    printf 'Main ref:         %s\n' "$MAIN_BRANCH"
    printf 'Conda env:        %s (%s)\n' "$CONDA_ENV" "$CONDA_BASE"
    printf 'CANN root:        %s\n' "$CANN_ROOT"
    printf 'PTOAS root:       %s\n' "$PTOAS_BIN"
    printf 'PTO ISA root:     %s\n' "$PTO_ISA_DIR"
    printf 'State root:       %s\n' "$STATE_ROOT"
    printf 'Daily schedule:   %s:00 Asia/Shanghai\n' "$DEFAULT_SCHEDULE_HOUR"
    printf '\nBenchmarks:\n'
    print_case_plan
}

extract_mean_us() {
    local log_file="$1"
    sed -nE 's/.*effective_us .*mean=([0-9]+([.][0-9]+)?).*/\1/p' "$log_file" | tail -n 1
}

run_case() {
    local checkout="$1"
    local result_dir="$2"
    local git_sha="$3"
    local label="$4"
    local ptoas_version="$5"
    local device="$6"
    local device_num="$7"
    local python_command="$8"
    local benchmark_log_name="$9"
    local benchmark_log="$result_dir/$benchmark_log_name"
    local submit_log="$result_dir/${label}.task-submit.log"
    local checkout_q conda_sh_q conda_env_q cann_set_env_q benchmark_log_q
    local child_command rc mean_us status
    local -a submit_command

    printf -v checkout_q '%q' "$checkout"
    printf -v conda_sh_q '%q' "$CONDA_BASE/etc/profile.d/conda.sh"
    printf -v conda_env_q '%q' "$CONDA_ENV"
    printf -v cann_set_env_q '%q' "$CANN_ROOT/set_env.sh"
    printf -v benchmark_log_q '%q' "$benchmark_log"

    child_command="cd $checkout_q && source $conda_sh_q && conda activate $conda_env_q && source $cann_set_env_q && export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\" && PYPTO_BENCH=1 $python_command > $benchmark_log_q 2>&1"
    submit_command=(
        task-submit
        --ptoas "$ptoas_version"
        --device "$device"
        --env PTO_ISA_ROOT
        --env TZ
        --env PTO2_RING_TASK_WINDOW
        --env PTO2_RING_DEP_POOL
        --env PTO2_RING_HEAP
    )
    if [[ -n "$device_num" ]]; then
        submit_command+=(--device-num "$device_num")
    fi
    submit_command+=(
        --timeout 0
        --max-time 1800
        --run "$child_command"
    )

    : >"$benchmark_log"
    log "Starting $label (ptoas=$ptoas_version, device=$device${device_num:+, count=$device_num})"
    if "${submit_command[@]}" >"$submit_log" 2>&1; then
        rc=0
    else
        rc=$?
    fi

    mean_us="$(extract_mean_us "$benchmark_log")"
    if [[ "$rc" -ne 0 ]]; then
        status="fail"
    elif [[ -z "$mean_us" ]]; then
        status="missing_metric"
    else
        status="pass"
    fi
    [[ -n "$mean_us" ]] || mean_us="-"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" "$status" "$ptoas_version" "$device" "$mean_us" "$git_sha" "$benchmark_log_name" \
        >>"$result_dir/results.tsv"

    log "Finished $label: status=$status, mean_us=$mean_us"
    if [[ -s "$submit_log" ]]; then
        tail -n 20 "$submit_log"
    fi
    if [[ -s "$benchmark_log" ]]; then
        tail -n 20 "$benchmark_log"
    fi

    [[ "$status" == "pass" ]]
}

run_benchmarks() {
    local checkout="$1"
    local result_dir="$2"
    local git_sha="$3"
    local failures=0

    printf 'case\tstatus\tptoas\tdevice\tmean_us\tgit_sha\tlog\n' >"$result_dir/results.tsv"

    run_case \
        "$checkout" "$result_dir" "$git_sha" \
        "decode_attention_csa_8k" "0.48" "$ATTENTION_DEVICE" "" \
        'python models/deepseek/v4-flash/decode_attention_csa.py -p a2a3 -d "$TASK_DEVICE" --start-pos 8192 --enable-l2-swimlane 0' \
        "decode_attention_csa_8k.log" || failures=$((failures + 1))

    run_case \
        "$checkout" "$result_dir" "$git_sha" \
        "decode_attention_swa_8k" "0.48" "$ATTENTION_DEVICE" "" \
        'python models/deepseek/v4-flash/decode_attention_swa.py -p a2a3 -d "$TASK_DEVICE" --start-pos 8192 --enable-l2-swimlane 0' \
        "decode_attention_swa_8k.log" || failures=$((failures + 1))

    run_case \
        "$checkout" "$result_dir" "$git_sha" \
        "decode_attention_hca_8k" "0.48" "$ATTENTION_DEVICE" "" \
        'python models/deepseek/v4-flash/decode_attention_hca.py -p a2a3 -d "$TASK_DEVICE" --start-pos 8192 --enable-l2-swimlane 0' \
        "decode_attention_hca_8k.log" || failures=$((failures + 1))

    run_case \
        "$checkout" "$result_dir" "$git_sha" \
        "moe_ep2" "0.54" "$MOE_DEVICE" "" \
        'python models/deepseek/v4-flash/moe.py -p a2a3 -d "$TASK_DEVICE" --ep 2 --enable-l2-swimlane 1' \
        "moe_ep2_2.log" || failures=$((failures + 1))

    run_case \
        "$checkout" "$result_dir" "$git_sha" \
        "decode_fwd_43l_ep2_8k" "0.54" "$DECODE_DEVICE" "" \
        'python models/deepseek/v4-flash/decode_fwd.py -p a2a3 --ep 2 --tp 2 -d "$TASK_DEVICE" --start-pos 8192 --num-tokens 8 --enable-l2-swimlane 0' \
        "decode_fwd_43l_ep2_8k_perf.log" || failures=$((failures + 1))

    if [[ "$failures" -ne 0 ]]; then
        log "$failures benchmark case(s) failed or emitted no effective_us mean."
        return 1
    fi

    log "All benchmark cases completed successfully."
}

run_daily() {
    local url timestamp result_dir temp_root checkout old_oid rebased_oid
    local rebase_log push_log

    require_command flock
    mkdir -p "$RESULT_ROOT" "$STATE_ROOT/tmp"
    exec 9>"$STATE_ROOT/run.lock"
    if ! flock -n 9; then
        log "Another daily performance run is active; skipping."
        return 0
    fi

    activate_environment
    url="$(remote_url)"
    timestamp="$(date '+%Y%m%d-%H%M%S')"
    result_dir="$RESULT_ROOT/$timestamp"
    mkdir -p "$result_dir"
    ln -sfn "$result_dir" "$STATE_ROOT/latest"

    temp_root="$(mktemp -d "$STATE_ROOT/tmp/run.XXXXXX")"
    RUN_TEMP_ROOT="$temp_root"
    checkout="$temp_root/checkout"

    {
        printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
        printf 'remote_url=%s\n' "$url"
        printf 'performance_branch=%s\n' "$PERF_BRANCH"
        printf 'main_branch=%s\n' "$MAIN_BRANCH"
        printf 'conda_env=%s\n' "$CONDA_ENV"
        printf 'cann_root=%s\n' "$CANN_ROOT"
        printf 'ptoas_root=%s\n' "$PTOAS_BIN"
        printf 'pto_isa_root=%s\n' "$PTO_ISA_DIR"
    } >"$result_dir/metadata.txt"

    log "Cloning $url into an isolated checkout."
    git clone --quiet --no-tags "$url" "$checkout"
    git -C "$checkout" fetch --quiet --prune origin \
        "+refs/heads/$MAIN_BRANCH:refs/remotes/origin/$MAIN_BRANCH" \
        "+refs/heads/$PERF_BRANCH:refs/remotes/origin/$PERF_BRANCH"
    old_oid="$(git -C "$checkout" rev-parse "origin/$PERF_BRANCH")"
    git -C "$checkout" checkout --quiet -B "$PERF_BRANCH" "origin/$PERF_BRANCH"

    printf 'old_performance_sha=%s\n' "$old_oid" >>"$result_dir/metadata.txt"
    printf 'main_sha=%s\n' "$(git -C "$checkout" rev-parse "origin/$MAIN_BRANCH")" \
        >>"$result_dir/metadata.txt"

    rebase_log="$result_dir/rebase.log"
    log "Rebasing $PERF_BRANCH onto origin/$MAIN_BRANCH."
    if GIT_EDITOR=true GIT_SEQUENCE_EDITOR=true git -C "$checkout" rebase "origin/$MAIN_BRANCH" \
        >"$rebase_log" 2>&1; then
        cat "$rebase_log"
    else
        local rebase_rc=$?
        cat "$rebase_log"
        git -C "$checkout" diff --name-only --diff-filter=U >"$result_dir/rebase-conflicts.txt"
        git -C "$checkout" status --short >"$result_dir/rebase-status.txt"
        git -C "$checkout" rebase --abort
        log "Rebase failed; remote branch and benchmarks were left untouched."
        return "$rebase_rc"
    fi

    git -C "$checkout" diff --check "origin/$MAIN_BRANCH...HEAD"
    rebased_oid="$(git -C "$checkout" rev-parse HEAD)"
    printf 'rebased_performance_sha=%s\n' "$rebased_oid" >>"$result_dir/metadata.txt"

    push_log="$result_dir/push.log"
    log "Publishing rebased branch with force-with-lease."
    if git -C "$checkout" push \
        --force-with-lease="refs/heads/$PERF_BRANCH:$old_oid" \
        origin "HEAD:refs/heads/$PERF_BRANCH" >"$push_log" 2>&1; then
        cat "$push_log"
    else
        local push_rc=$?
        cat "$push_log"
        log "Push lease failed; benchmarks were not started."
        return "$push_rc"
    fi

    if run_benchmarks "$checkout" "$result_dir" "$rebased_oid"; then
        printf 'finished_at=%s\nstatus=pass\n' "$(date --iso-8601=seconds)" \
            >>"$result_dir/metadata.txt"
        return 0
    fi

    printf 'finished_at=%s\nstatus=fail\n' "$(date --iso-8601=seconds)" \
        >>"$result_dir/metadata.txt"
    return 1
}

validate_hour() {
    local hour="$1"
    [[ "$hour" =~ ^([01][0-9]|2[0-3])$ ]] || die "Hour must be two digits from 00 to 23: $hour"
}

filter_managed_cron() {
    awk -v begin="$CRON_BEGIN" -v end="$CRON_END" '
        $0 == begin { managed = 1; next }
        $0 == end { managed = 0; next }
        !managed { print }
    '
}

install_cron() {
    local hour="${1:-$DEFAULT_SCHEDULE_HOUR}"
    local existing filtered script_q cron_log_q conda_base_q

    validate_hour "$hour"
    require_command crontab
    [[ -n "$CONDA_BASE" ]] || die "Cannot resolve the Conda base directory."
    mkdir -p "$STATE_ROOT"

    existing="$(crontab -l 2>/dev/null || true)"
    filtered="$(printf '%s\n' "$existing" | filter_managed_cron)"
    printf -v script_q '%q' "$SCRIPT_PATH"
    printf -v cron_log_q '%q' "$CRON_LOG"
    printf -v conda_base_q '%q' "$CONDA_BASE"

    {
        if [[ -n "${filtered//[$'\n\r\t ']}" ]]; then
            printf '%s\n' "$filtered"
        fi
        printf '%s\n' "$CRON_BEGIN"
        printf '0 * * * * PYPTO_DAILY_PERF_HOUR=%s PYPTO_DAILY_CONDA_BASE=%s /bin/bash %s scheduled >> %s 2>&1\n' \
            "$hour" "$conda_base_q" "$script_q" "$cron_log_q"
        printf '%s\n' "$CRON_END"
    } | crontab -

    log "Installed daily run for $hour:00 Asia/Shanghai."
    log "Cron output: $CRON_LOG"
}

remove_cron() {
    local existing filtered

    require_command crontab
    existing="$(crontab -l 2>/dev/null || true)"
    filtered="$(printf '%s\n' "$existing" | filter_managed_cron)"
    if [[ -n "${filtered//[$'\n\r\t ']}" ]]; then
        printf '%s\n' "$filtered" | crontab -
    else
        crontab -r 2>/dev/null || true
    fi
    log "Removed the managed daily performance cron entry."
}

scheduled_run() {
    local hour="${PYPTO_DAILY_PERF_HOUR:-$DEFAULT_SCHEDULE_HOUR}"

    validate_hour "$hour"
    if [[ "$(date '+%H')" != "$hour" ]]; then
        return 0
    fi
    run_daily
}

main() {
    case "${1:-}" in
        run)
            run_daily
            ;;
        scheduled)
            scheduled_run
            ;;
        dry-run)
            dry_run
            ;;
        install-cron)
            install_cron "${2:-$DEFAULT_SCHEDULE_HOUR}"
            ;;
        remove-cron)
            remove_cron
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
}

main "$@"
