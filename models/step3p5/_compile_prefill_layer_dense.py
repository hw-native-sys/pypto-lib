"""Compile a single step3p5 prefill layer (dense, TP=8) for codegen verification.

DEFERRED — current prefill per-layer programs overflow 192KB Vec (PREFILL_T=128
monolithic + FP32 buffers); needs token-tiling (BATCH<=8) restructuring, tracked
for Phase 17.

Compile-only driver — no NPU execution. Builds a multi-card
``PrefillLayerDense`` ``@pl.program`` and runs it through
``pypto.ir.compile`` against the ``a2a3sim`` simulator backend.

Phase 14.H target: prefill dense layer codegen.

Usage (from pypto-lib/):
    cd /data/chensiyu/hw_project/pypto/workspace/pypto-lib
    PYPTO_PROG_BUILD_DIR=/tmp/p14h_dense \\
      python -m models.step3p5._compile_prefill_layer_dense

The output dir under ``$PYPTO_PROG_BUILD_DIR`` will contain
``kernel_config.py``, per-kernel ``.cpp`` wrappers, ``.pto`` MLIR
dumps, and ``passes/`` IR snapshots when codegen succeeds.
"""
from __future__ import annotations

import argparse
import sys

from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

from .config import TP_WORLD_SIZE
from .prefill_fwd import select_prefill_layer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", default="a2a3sim",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument(
        "--layer-idx", type=int, default=0,
        help="Layer index; 0..2 are full_dense. Default: 0",
    )
    args = parser.parse_args()

    program, kind = select_prefill_layer(args.layer_idx)
    prog_name = getattr(program, "name", None) or type(program).__name__
    print(f"[14.H] compiling layer {args.layer_idx} kind={kind} program={prog_name}",
          flush=True)

    dist_cfg = DistributedConfig(
        device_ids=list(range(TP_WORLD_SIZE)),
        num_sub_workers=0,
    )

    compiled = ir.compile(
        program,
        platform=args.platform,
        distributed_config=dist_cfg,
        skip_ptoas=False,
        dump_passes=True,
    )
    print(f"[14.H] OK output_dir={compiled.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
