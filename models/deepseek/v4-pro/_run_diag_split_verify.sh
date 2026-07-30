#!/usr/bin/env bash
set -euo pipefail
source "$HOME/Desktop/pypto-env.sh"
export PYTHONPATH="$HOME/Desktop/pypto2/python:$HOME/Desktop/pypto2/runtime:$HOME/Desktop/pypto-lib:$PYTHONPATH"
cd "$HOME/Desktop/pypto-lib"
echo "python=$(command -v python)"
python -c "import pypto; print(pypto.__file__)"
rm -rf models/deepseek/v4-pro/build_output/_jit_aiv_quant_store_* 2>/dev/null || true
exec python models/deepseek/v4-pro/_diag_prefill_qr_proj_instr.py -p a5 -d "${TASK_DEVICE:-0}" --mode split
