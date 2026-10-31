"""Run the real layer-0 shared W8A8 expert bundle on A2/A3."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .moe import make_shared_expert_w8a8, shared_expert_reference
from .weights import load_layer0_shared_expert


DEFAULT_WEIGHTS = Path("/data/disk/deepseek-v41-flash/v41-moe-layer0-a2-w8a8/shared-layer0-w8a8.pt")
CAPACITY = 16
ACTIVE = 7
HIDDEN = 5120
INTERMEDIATE = 2304


def _specs(weights, capacity: int, active: int):
    from golden import ScalarSpec, TensorSpec

    generator = torch.Generator().manual_seed(410101)
    value = torch.randn((capacity, HIDDEN), generator=generator).to(torch.bfloat16)
    return [
        TensorSpec("value", [capacity, HIDDEN], torch.bfloat16, init_value=value),
        TensorSpec("gate_up_weight", [HIDDEN, INTERMEDIATE * 2], torch.int8, init_value=weights["gate_up_weight"]),
        TensorSpec("gate_up_scale", [INTERMEDIATE * 2], torch.float32, init_value=weights["gate_up_scale"]),
        TensorSpec("down_weight", [INTERMEDIATE, HIDDEN], torch.int8, init_value=weights["down_weight"]),
        TensorSpec("down_scale", [HIDDEN], torch.float32, init_value=weights["down_scale"]),
        TensorSpec("output", [capacity, HIDDEN], torch.bfloat16, init_value=torch.zeros(capacity, HIDDEN, dtype=torch.bfloat16)),
        ScalarSpec("active", torch.int32, active),
    ]


def _golden(values, active: int):
    # PyPTO receives canonical [K,N] weights; the Torch reference consumes
    # output-major [N,K] rows for its explicit ``value @ weight.T`` product.
    gate_up = values["gate_up_weight"].transpose(0, 1).contiguous()
    down = values["down_weight"].transpose(0, 1).contiguous()
    expected = shared_expert_reference(
        values["value"], gate_up, values["gate_up_scale"], down, values["down_scale"]
    )
    values["output"][:active] = expected[:active]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--capacity", type=int, default=CAPACITY)
    parser.add_argument("--active", type=int, default=ACTIVE)
    args = parser.parse_args()
    if args.capacity <= 0 or args.capacity % 16 or not 0 < args.active <= args.capacity:
        raise ValueError("capacity must be a positive multiple of 16 and active must be in (0, capacity]")
    weights = load_layer0_shared_expert(args.weights)
    compute = make_shared_expert_w8a8(args.capacity, HIDDEN, INTERMEDIATE)
    from golden import run

    result = run(
        fn=compute,
        specs=_specs(weights, args.capacity, args.active),
        golden_fn=lambda values: _golden(values, args.active),
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        compile_only=args.compile_only,
        rtol=5.0e-2,
        atol=5.0e-2,
    )
    print(result)
    return int(not result.passed)


if __name__ == "__main__":
    raise SystemExit(main())
