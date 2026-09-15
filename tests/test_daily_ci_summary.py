# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise Daily CI's status-only shell writer and summary without hardware."""

import json
from pathlib import Path
import re
import subprocess
import sys
import textwrap

import pytest


WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/daily_ci.yml"


def summary_source():
    text = WORKFLOW.read_text()
    start = text.index('          import pathlib', text.index('- name: Build combined summary table'))
    end = text.index('          EOF', start)
    return textwrap.dedent(text[start:end])


def artifact(root, platform, text):
    path = root / "results" / f"results-{platform}" / "results.tsv"
    path.parent.mkdir(parents=True)
    path.write_text(text)


def summarize(root):
    return subprocess.run([sys.executable, "-c", summary_source()], cwd=root,
                          capture_output=True, text=True, check=True).stdout


def test_summary_preserves_statuses_counts_and_e2e(tmp_path):
    artifact(tmp_path, "a2a3", "model/a.py\tpass\nmodel/b.py\tfail\n")
    artifact(tmp_path, "a2a3sim", "model/a.py\tpass\n")
    artifact(tmp_path, "a5", "models/deepseek_v4_pro/prefill_mtp.py\tpass\n")
    artifact(tmp_path, "a5sim", "")
    e2e = tmp_path / "results/results-e2e-a5/e2e.json"
    e2e.parent.mkdir()
    e2e.write_text(json.dumps({"status": "skipped", "note": "no test weights"}))
    output = summarize(tmp_path)
    assert "| Case | a2a3 | a2a3sim |" in output
    assert "| Case | a5 | a5sim |" in output
    assert "**a2a3** 1/2" in output
    assert "| `model/b.py` | :x: | :heavy_minus_sign: |" in output
    assert "only a5" in output
    assert "no test weights" in output and "skipped" in output
    assert "effective" not in output


def test_missing_artifacts_remain_visible(tmp_path):
    output = summarize(tmp_path)
    assert "No results artifact" in output
    assert "a2a3" in output and "a5" in output and "e2e-a5" in output


def test_historical_third_column_does_not_reappear(tmp_path):
    artifact(tmp_path, "a2a3", "model/a.py\tpass\t12345.6\n")
    output = summarize(tmp_path)
    assert "model/a.py" in output and "12345.6" not in output


@pytest.mark.parametrize("job", ["model-tests-a2a3", "model-tests-a5"])
@pytest.mark.parametrize("returncode", [0, 1])
def test_case_writer_uses_two_columns_and_propagates_failure(tmp_path, job, returncode):
    text = WORKFLOW.read_text().split(f"  {job}:\n", 1)[1]
    match = re.search(r"          run_case\(\) \{.*?\n          \}\n", text, re.S)
    assert match is not None
    fn = textwrap.dedent(match[0])
    model = tmp_path / "model.py"
    model.write_text("# ci: devices=2\n")
    # This shell function shadows the external queue command. No task is submitted.
    script = f'''task-submit() {{
      printf '[RUN] effective_us (100 rounds) min=1 median=2 mean=3 max=4\\n'
      return {returncode}
    }}
    DEVICE_ID=auto
    failed=()
    : > results.tsv
    {fn}
    run_case model.py model.py
    printf '%s' "${{#failed[@]}}" > failures.txt
    '''
    subprocess.run(["bash", "-c", script], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert (tmp_path / "results.tsv").read_text() == f"model.py\t{'pass' if returncode == 0 else 'fail'}\n"
    assert (tmp_path / "failures.txt").read_text() == str(returncode)
    assert "PYPTO_BENCH=0" in fn
