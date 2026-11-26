#!/usr/bin/env bash
# Run the PR and daily device sweeps with the same execution contract.
set -eo pipefail
source activate.sh
case "$PLATFORM" in a2a3|a5) ;; *) echo "Unsupported device platform: $PLATFORM"; exit 1 ;; esac

: > results.tsv
failed=()
set -f
for file in $FILES; do
  echo "::group::$file"
  test -f "$file"
  ndev=$(grep -oP '#\s*ci:\s*devices=\K[0-9]+' "$file" | head -1 || true)
  ndev=${ndev:-1}
  printf -v command 'cd %q && source activate.sh && PYPTO_BENCH=0 PYPTO_LOG_LEVEL=error PYPTO_WARNING_LEVEL=none PYPTO_RUNTIME_LOG=error PYTHONPATH=%q"${PYTHONPATH:+:$PYTHONPATH}" python %q -p %q' \
    "$PWD" "$PWD" "$file" "$PLATFORM"
  compile_case=$(sed -nE 's/^#[[:space:]]*ci:[[:space:]]*(compile-only([[:space:]].*)?)$/\1/p' "$file" | head -1)
  if [ -n "$compile_case" ]; then
    read -ra compile_args <<< "--$compile_case"
    printf -v flags ' %q' "${compile_args[@]}"
    command+="$flags"
    device_flags=(--no-device)
  else
    device_flags=(--device auto)
    command+=' -d "$TASK_DEVICE"'
    if [ "$ndev" -gt 1 ]; then
      device_flags+=(--device-num "$ndev" --ignore-whitelist)
    fi
  fi
  if task-submit "${device_flags[@]}" --timeout 3600 --max-time 900 --run "$command"; then
    status=pass
  else
    status=fail
    failed+=("$file")
    echo "::error file=${file}::FAIL"
  fi
  echo "[device][$PLATFORM] $file: $status"
  printf '%s\t%s\n' "$file" "$status" >> results.tsv
  echo "::endgroup::"
done
if [ ${#failed[@]} -ne 0 ]; then
  printf 'FAILED: %s\n' "${failed[@]}"
  exit 1
fi
echo "All selected files passed."
