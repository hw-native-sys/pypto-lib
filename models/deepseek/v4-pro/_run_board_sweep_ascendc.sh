#!/usr/bin/env bash
# Full pypto2 board sweep with AscendC-aligned error ratios
# (ATOL_RTOL in mx_quant_common.py + per-op compare_fn).
set -u
source "$HOME/Desktop/pypto-env.sh"
export PYTHONPATH="$HOME/Desktop/pypto2/python:$HOME/Desktop/pypto2/runtime:$HOME/Desktop/pypto-lib:$PYTHONPATH"
cd "$HOME/Desktop/pypto-lib"
DEV="${TASK_DEVICE:-0}"
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="$HOME/board_logs/sweep_ascendc_${STAMP}"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/SUMMARY.txt"
{
  echo "device=$DEV stamp=$STAMP"
  echo "pypto=$(python -c 'import pypto; print(pypto.__file__)')"
  echo "ATOL_RTOL pct (AscendC-aligned):"
  PYTHONPATH="models/deepseek/v4-pro:$PYTHONPATH" python - <<'PY'
from mx_quant_common import ATOL_RTOL
for k, v in ATOL_RTOL.items():
    print(f"  {k}: pct={v['pct']} atol={v['atol']} rtol={v['rtol']}")
PY
} | tee "$SUMMARY"

OPS="
expert_shared.py
expert_routed.py
moe.py
qkv_proj_rope.py
mtp_projection.py
decode_indexer.py
prefill_indexer.py
decode_indexer_compressor.py
prefill_indexer_compressor.py
decode_sparse_attn.py
decode_sparse_attn_swa.py
decode_sparse_attn_hca.py
prefill_sparse_attn.py
decode_compressor_ratio4.py
decode_compressor_ratio128.py
prefill_compressor_ratio4.py
prefill_compressor_ratio128.py
"

pass_n=0
fail_n=0
timeout_n=0
for f in $OPS; do
  name="${f%.py}"
  echo ""
  echo "########## OP: $f (pypto2, a5 dev $DEV, AscendC tol) ##########"
  timeout 540 python "models/deepseek/v4-pro/$f" -p a5 -d "$DEV" > "$LOGDIR/a5_${name}.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "RESULT $f : PASS (exit 0)" | tee -a "$SUMMARY"
    pass_n=$((pass_n + 1))
  elif [ $rc -eq 124 ]; then
    echo "RESULT $f : TIMEOUT (>540s)" | tee -a "$SUMMARY"
    timeout_n=$((timeout_n + 1))
  else
    echo "RESULT $f : FAIL (exit $rc)" | tee -a "$SUMMARY"
    fail_n=$((fail_n + 1))
    grep -iE "expects .* operands|PartialCodegenError|Failed to compile|ModuleNotFound|assert|Error:|FAIL |bad=|ratio=|pct=" \
      "$LOGDIR/a5_${name}.log" | head -10 | tee -a "$SUMMARY"
  fi
done
echo "" | tee -a "$SUMMARY"
echo "===== SWEEP DONE AscendC-tol pass=$pass_n fail=$fail_n timeout=$timeout_n =====" | tee -a "$SUMMARY"
echo "logs: $LOGDIR"
[ $((fail_n + timeout_n)) -eq 0 ]
