"""Run all L3 consumer boundaries in the order used by the decoder.

This is intentionally a small process-level driver.  PyPTO and vLLM both
initialize device and custom-op state globally, so keeping the two validators
as independent stages avoids stale compiled signatures while still providing
one reproducible L3 command.  Each validator uses the same capacity/active
bucket and has its own vLLM reference invocation; the optional dump directory
keeps the named PyPTO boundaries inspectable after the run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--active", type=int, default=7)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "--dump-dir",
        type=Path,
        help="save each validator's active PyPTO nodes under this directory",
    )
    parser.add_argument(
        "--vllm-dir",
        type=Path,
        help="save each vLLM-Ascend reference boundary under this directory",
    )
    args = parser.parse_args()
    # QLI currently validates every row in its fixed-capacity signature.  The
    # other consumer boundaries carry an explicit active-row scalar because
    # their cache/output ABIs have a visible-prefix tail.  Keep the command
    # construction stage-specific so the assembly driver matches each public
    # validator ABI instead of forwarding an unsupported ``--active`` flag to
    # QLI.
    common = ["--capacity", str(args.capacity)]
    stages = (
        (
            "qli-select",
            "models.deepseek_v4_1_flash_a2.validate_layer3_indexer_select",
            "qli_select.pt",
        ),
        (
            "indexer-prepare",
            "models.deepseek_v4_1_flash_a2.validate_layer3_indexer_prepare",
            "indexer_prepare.pt",
        ),
        (
            "indexer-query",
            "models.deepseek_v4_1_flash_a2.validate_layer3_components",
            "indexer_query.pt",
        ),
        (
            "sparse-attention",
            "models.deepseek_v4_1_flash_a2.validate_layer3_sparse_attention",
            "sparse_attention.pt",
        ),
        (
            "output-projection",
            "models.deepseek_v4_1_flash_a2.validate_layer3_output",
            "output_projection.pt",
        ),
    )
    if args.dump_dir is not None:
        runtime_dir = args.dump_dir
        runtime_dir.mkdir(parents=True, exist_ok=True)
        temporary_dir = None
    else:
        temporary_dir = tempfile.TemporaryDirectory(prefix="dsv41_l3_assembly_")
        runtime_dir = Path(temporary_dir.name)
    if args.vllm_dir is not None:
        args.vllm_dir.mkdir(parents=True, exist_ok=True)
    qli_vllm_path = None
    for stage, module, dump_name in stages:
        command = [sys.executable, "-m", module, *common]
        if stage not in ("qli-select", "indexer-prepare"):
            command.extend(["--active", str(args.active)])
        if args.sim:
            command.append("--sim")
        else:
            command.extend(["--device", str(args.device)])
        actual_path = runtime_dir / dump_name
        command.extend(["--actual-out", str(actual_path)])
        vllm_path = None
        if not args.sim and stage in (
            "qli-select", "indexer-prepare", "sparse-attention",
            "indexer-query", "output-projection",
        ):
            vllm_root = args.vllm_dir if args.vllm_dir is not None else runtime_dir
            vllm_path = vllm_root / dump_name
            command.extend(["--vllm-out", str(vllm_path)])
            if stage == "qli-select":
                qli_vllm_path = vllm_path
        if stage == "indexer-prepare":
            selected_path = (runtime_dir / "qli_select.pt") if args.sim else qli_vllm_path
            if selected_path is None:
                raise RuntimeError("missing QLI handoff path")
            command.extend(["--selected-indices", str(selected_path)])
        elif stage == "sparse-attention":
            command.extend(["--compressed-indices", str(runtime_dir / "indexer_prepare.pt")])
        print(f"[L3 assembly] {stage}: running {' '.join(command)}", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            print(f"[L3 assembly] {stage}: FAIL (rc={completed.returncode})", flush=True)
            if temporary_dir is not None:
                temporary_dir.cleanup()
            return completed.returncode
        print(f"[L3 assembly] {stage}: PASS", flush=True)
    print(
        "[L3 assembly] QLI selection -> index preparation -> indexer query -> ratio-2 sparse attention -> "
        "inverse RoPE -> Wo-A -> Wo-B PASS",
        flush=True,
    )
    if temporary_dir is not None:
        temporary_dir.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
