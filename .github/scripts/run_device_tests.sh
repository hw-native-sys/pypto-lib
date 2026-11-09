#!/usr/bin/env bash
# Run the PR and daily device sweeps with the same execution contract.
set -eo pipefail
source activate.sh
case "$PLATFORM" in a2a3|a5) ;; *) echo "Unsupported device platform: $PLATFORM"; exit 1 ;; esac

: > results.tsv
rm -rf ci_failure_data/decode_layer
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
  extra_args=()
  capture_decode_layer=false
  if [ "$PLATFORM" = a2a3 ] && [ "$file" = models/deepseek_v4_flash_lowlat/decode_layer.py ]; then
    extra_args+=(--save-data)
    capture_decode_layer=true
  fi
  printf -v command 'cd %q && source activate.sh && PYPTO_BENCH=0 PYPTO_LOG_LEVEL=error PYPTO_WARNING_LEVEL=none PYPTO_RUNTIME_LOG=error PYTHONPATH=%q python %q -p %q -d "$TASK_DEVICE" %s' \
    "$PWD" "$PWD" "$file" "$PLATFORM" "${extra_args[*]}"
  if task-submit --device auto "${device_flags[@]}" --timeout 3600 --max-time 900 --run "$command"; then
    status=pass
  else
    status=fail
    failed+=("$file")
    if [ "$capture_decode_layer" = true ]; then
      input_dir=$(find build_output -type d -path '*/data/in' \
        -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
      if [ -n "$input_dir" ]; then
        data_dir=${input_dir%/in}
        mkdir -p ci_failure_data/decode_layer
        cp -a "$data_dir" ci_failure_data/decode_layer/
        printf 'platform=%s\ncase=%s\nseed=default (see CI log)\n' \
          "$PLATFORM" "$file" > ci_failure_data/decode_layer/README.txt
        echo "Saved decode_layer failure tensors from $data_dir"
      else
        echo "::warning::decode_layer failed, but no saved tensor directory was found under build_output"
      fi
    fi
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
