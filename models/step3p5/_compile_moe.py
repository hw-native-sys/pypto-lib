"""Compile the standalone step3p5 ``EpTpMoE`` block (TP+EP=8) for codegen.

Compile-only driver — no NPU execution. Builds the multi-card
``EpTpMoE`` ``@pl.program`` from ``moe.py`` and runs it through
``pypto.ir.compile`` against the ``a2a3sim`` simulator backend.

This is the Phase 14.E target: the standalone MoE block, sibling of the
``DecodeLayerMoE`` MoE path compiled in 14.D. Layer 3 selects the
(routed=0, shared=0) specialisation (layers 3..42).

Usage (from pypto-lib/):
    cd /data/chensiyu/hw_project/pypto/workspace/pypto-lib
    PYPTO_PROG_BUILD_DIR=/path/to/p14 \\
      python -m models.step3p5._compile_moe
"""
from __future__ import annotations

import argparse
import sys

from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

from .config import TP_WORLD_SIZE
from .moe import select_moe_block


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", default="a2a3sim",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument(
        "--layer-idx", type=int, default=3,
        help="MoE layer index; first MoE layer is 3. Default: 3",
    )
    args = parser.parse_args()

    program = select_moe_block(args.layer_idx)
    prog_name = getattr(program, "name", None) or type(program).__name__
    print(f"[14.E] compiling EpTpMoE layer {args.layer_idx} program={prog_name}",
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
    print(f"[14.E] OK output_dir={compiled.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
