"""Run the L3 index selection handoff through sparse attention.

The driver keeps each PyPTO/vLLM invocation in a separate process and passes
the prepared INT32 indices between stages:

    QLI -> prepare_indexer_indices -> compressed-cache sparse attention

On NPU, the prepare stage consumes the native vLLM QLI dump, and both sparse
attention implementations consume the same prepared dump.  Simulator mode
uses the PyPTO QLI dump as the equivalent CPU oracle input.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def _run(label: str, command: list[str]) -> None:
    print(f"[L3 chain] {label}: running {' '.join(command)}", flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit(f"[L3 chain] {label}: FAIL (rc={completed.returncode})")
    print(f"[L3 chain] {label}: PASS", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--active", type=int, default=7)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "--dump-dir",
        type=Path,
        help="persist PyPTO stage dumps; otherwise use a temporary directory",
    )
    parser.add_argument(
        "--vllm-dir",
        type=Path,
        help="persist vLLM-Ascend stage dumps on NPU",
    )
    args = parser.parse_args()
    if args.capacity <= 0 or args.capacity % 16:
        raise ValueError("capacity must be a positive multiple of 16")
    if not 0 < args.active <= args.capacity:
        raise ValueError("active must be in range [1, capacity]")

    temporary = None
    if args.dump_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="dsv41_l3_chain_")
        dump_dir = Path(temporary.name)
    else:
        dump_dir = args.dump_dir
        dump_dir.mkdir(parents=True, exist_ok=True)
    vllm_dir = args.vllm_dir
    if vllm_dir is not None:
        vllm_dir.mkdir(parents=True, exist_ok=True)
    print(f"[L3 chain] dump_dir={dump_dir}", flush=True)

    common = ["--capacity", str(args.capacity)]
    runtime = ["--sim"] if args.sim else ["--device", str(args.device)]
    qli_actual = dump_dir / "qli_select.pt"
    qli_vllm = (vllm_dir / "qli_select.pt") if vllm_dir is not None else None
    qli_command = [
        sys.executable,
        "-m",
        "models.deepseek_v4_1_flash_a2.validate_layer3_indexer_select",
        *common,
        *runtime,
        "--actual-out",
        str(qli_actual),
    ]
    if qli_vllm is not None:
        qli_command.extend(["--vllm-out", str(qli_vllm)])
    _run("qli-select", qli_command)

    # The native vLLM QLI output is the NPU handoff.  Simulator mode uses the
    # PyPTO output because it is the deterministic CPU QLI oracle there.
    selected_input = qli_actual if args.sim or qli_vllm is None else qli_vllm
    prepare_actual = dump_dir / "indexer_prepare.pt"
    prepare_vllm = (vllm_dir / "indexer_prepare.pt") if vllm_dir is not None else None
    prepare_command = [
        sys.executable,
        "-m",
        "models.deepseek_v4_1_flash_a2.validate_layer3_indexer_prepare",
        *common,
        *runtime,
        "--selected-indices",
        str(selected_input),
        "--actual-out",
        str(prepare_actual),
    ]
    if prepare_vllm is not None:
        prepare_command.extend(["--vllm-out", str(prepare_vllm)])
    _run("indexer-prepare", prepare_command)

    sparse_actual = dump_dir / "sparse_attention.pt"
    sparse_vllm = (vllm_dir / "sparse_attention.pt") if vllm_dir is not None else None
    sparse_command = [
        sys.executable,
        "-m",
        "models.deepseek_v4_1_flash_a2.validate_layer3_sparse_attention",
        *common,
        "--active",
        str(args.active),
        *runtime,
        "--compressed-indices",
        str(prepare_actual),
        "--actual-out",
        str(sparse_actual),
    ]
    if sparse_vllm is not None:
        sparse_command.extend(["--vllm-out", str(sparse_vllm)])
    _run("sparse-attention", sparse_command)

    print(
        "[L3 chain] QLI -> indexer_prepare -> prepared INT32 indices -> "
        "ratio-2 sparse attention PASS",
        flush=True,
    )
    if temporary is not None:
        temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
