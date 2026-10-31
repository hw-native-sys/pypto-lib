"""Simulator validation for the composed W8A8 shared/routed expert leaf."""

from __future__ import annotations

import argparse

import pypto.language as pl
import torch

from .moe import make_shared_expert_w8a8, shared_expert_reference
from .quant import quantize_per_channel_int8


CAPACITY = 16
ACTIVE = 7
HIDDEN = 512
INTERMEDIATE = 256
GATE_UP = INTERMEDIATE * 2
_expert = make_shared_expert_w8a8(CAPACITY, HIDDEN, INTERMEDIATE)


def build_specs():
    from golden import ScalarSpec, TensorSpec

    torch.manual_seed(48)
    value = torch.randn(CAPACITY, HIDDEN, dtype=torch.bfloat16)
    gate_up_src = torch.randn(GATE_UP, HIDDEN)
    down_src = torch.randn(HIDDEN, INTERMEDIATE)
    gate_up, gate_up_scale = quantize_per_channel_int8(gate_up_src)
    down, down_scale = quantize_per_channel_int8(down_src)
    return [
        TensorSpec("value", [CAPACITY, HIDDEN], torch.bfloat16, init_value=value),
        TensorSpec("gate_up_weight", [HIDDEN, GATE_UP], torch.int8, init_value=gate_up.transpose(0, 1).contiguous()),
        TensorSpec("gate_up_scale", [GATE_UP], torch.float32, init_value=gate_up_scale.reshape(-1)),
        TensorSpec("down_weight", [INTERMEDIATE, HIDDEN], torch.int8, init_value=down.transpose(0, 1).contiguous()),
        TensorSpec("down_scale", [HIDDEN], torch.float32, init_value=down_scale.reshape(-1)),
        TensorSpec("output", [CAPACITY, HIDDEN], torch.bfloat16, init_value=torch.zeros(CAPACITY, HIDDEN, dtype=torch.bfloat16)),
        ScalarSpec("active", torch.int32, ACTIVE),
    ]


def golden(values):
    gate_up = values["gate_up_weight"].transpose(0, 1).contiguous()
    down = values["down_weight"].transpose(0, 1).contiguous()
    result = shared_expert_reference(
        values["value"], gate_up, values["gate_up_scale"], down, values["down_scale"]
    )
    values["output"][:ACTIVE] = result[:ACTIVE]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    args = parser.parse_args()
    from golden import run

    result = run(
        fn=_expert,
        specs=build_specs(),
        golden_fn=golden,
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        rtol=5.0e-2,
        atol=5.0e-2,
    )
    print(result)
    return int(not result.passed)


if __name__ == "__main__":
    raise SystemExit(main())
