"""Compile the step3p5 ``Step3p5PrefillFwd`` program (final RMSNorm + LM head).

Compile-only driver — no NPU execution. Builds the multi-card
``Step3p5PrefillFwd`` ``@pl.program`` from ``prefill_fwd.py`` and runs it
through ``pypto.ir.compile`` against the ``a2a3sim`` simulator backend.

Phase 14.G target: the top-level prefill forward pass program (45 layers
staged from host_orch via per-layer programs; this program compiles the
final RMSNorm + vocab-sliced LM head chip_orch plus the host orchestration
shell that routes per-rank calls).

Usage (from pypto-lib/):
    cd /data/chensiyu/hw_project/pypto/workspace/pypto-lib
    PYPTO_PROG_BUILD_DIR=/tmp/p14g \\
      python -m models.step3p5._compile_prefill_fwd
"""
from __future__ import annotations

import argparse
import sys

from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

from .config import TP_WORLD_SIZE
from .prefill_fwd import Step3p5PrefillFwd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", default="a2a3sim",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    args = parser.parse_args()

    prog_name = getattr(Step3p5PrefillFwd, "name", None) or type(Step3p5PrefillFwd).__name__
    print(f"[14.G] compiling Step3p5PrefillFwd program={prog_name}", flush=True)

    dist_cfg = DistributedConfig(
        device_ids=list(range(TP_WORLD_SIZE)),
        num_sub_workers=0,
    )

    compiled = ir.compile(
        Step3p5PrefillFwd,
        platform=args.platform,
        distributed_config=dist_cfg,
        skip_ptoas=False,
        dump_passes=True,
    )
    print(f"[14.G] OK output_dir={compiled.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
