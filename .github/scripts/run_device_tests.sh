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
  device_flags=()
  if [ "$ndev" -gt 1 ]; then
    device_flags+=(--device-num "$ndev" --ignore-whitelist)
  fi
  printf -v command 'cd %q && source activate.sh && PYPTO_BENCH=0 PYPTO_LOG_LEVEL=error PYPTO_WARNING_LEVEL=none PYPTO_RUNTIME_LOG=error PYTHONPATH=%q python %q -p %q -d "$TASK_DEVICE"' \
    "$PWD" "$PWD" "$file" "$PLATFORM"
  if task-submit --device auto "${device_flags[@]}" --timeout 3600 --max-time 900 --run "$command"; then
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
