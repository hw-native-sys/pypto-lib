"""Capture the native vLLM-Ascend layer-0 shared W8A8 expert."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        default=Path("/data/disk/deepseek-v41-flash/v41-moe-layer0-a2-w8a8/shared-layer0-w8a8.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--active", type=int, default=7)
    parser.add_argument("-d", "--device", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.active <= args.capacity:
        raise ValueError("active must be in (0, capacity]")

    weights = torch.load(args.weights, map_location="cpu", weights_only=True)
    generator = torch.Generator().manual_seed(410200)
    value = torch.randn(args.capacity, 5120, generator=generator).to(torch.bfloat16)

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.models.deepseek_v41.model import DeepseekV41MLP
    from vllm_ascend.utils import enable_custom_op
    from .capture_vllm_qkv import FrozenW8A8Config

    torch.npu.set_device(args.device)
    device = torch.device("npu", args.device)
    description = {
        "layers.0.ffn.shared_experts.gate_up_proj.weight": "W8A8_DYNAMIC",
        "layers.0.ffn.shared_experts.down_proj.weight": "W8A8_DYNAMIC",
    }
    quant_config = FrozenW8A8Config(description)
    config = VllmConfig()
    config.quant_config = quant_config
    result = {"capture_kind": "vllm_ascend_native_shared_w8a8", "device": args.device, "cases": {}}
    case_name = f"layer0-shared-cap{args.capacity}-active{args.active}"
    with set_current_vllm_config(config), torch.inference_mode():
        init_ascend_config(config)
        enable_custom_op()
        rendezvous = args.output.parent / (args.output.stem + "-rendezvous")
        init_distributed_environment(1, 0, "file://" + str(rendezvous), args.device, "hccl")
        initialize_model_parallel(1, 1, backend="hccl")
        module = DeepseekV41MLP(
            5120, 2304, "silu", swiglu_limit=10.0, quant_config=quant_config,
            reduce_results=False, is_sequence_parallel=True,
            prefix="layers.0.ffn.shared_experts",
        )
        for name, stem in (("gate_up_proj", "gate_up"), ("down_proj", "down")):
            layer = getattr(module, name)
            layer.weight.data = weights[stem + "_weight"].T.contiguous().to(device)
            layer.weight_scale.data = weights[stem + "_scale"][:, None].to(device)
            layer.weight_offset.data = torch.zeros_like(layer.weight_scale)
            layer.quant_method.process_weights_after_loading(layer)

        nodes = {}
        def hook(name):
            def save(_module, _inputs, output):
                value_out = output[0] if isinstance(output, tuple) else output
                nodes[name] = value_out.detach().cpu().clone()
            return save

        module.gate_up_proj.register_forward_hook(hook("shared_gate_up"))
        module.act_fn.register_forward_hook(hook("shared_activation"))
        output = module(value[:args.active].to(device))
        torch.npu.synchronize()
        nodes["shared_output"] = output.detach().cpu().clone()
        result["cases"][case_name] = {
            "active": args.active, "capacity": args.capacity, "nodes": nodes,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(f"capture={args.output} case={case_name} nodes={sorted(nodes)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
