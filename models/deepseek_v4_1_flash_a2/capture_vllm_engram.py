"""Capture the native vLLM-Ascend A2/A3 layer-1 Engram boundary.

The fixture deliberately matches ``validate_layer1_engram``.  The capture is
kept as a single case with named nodes so it can be consumed by
``compare_vllm_dump`` without importing vLLM in the PyPTO validator process.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401


def _inputs(table_rows: int, capacity: int, active: int):
    generator = torch.Generator().manual_seed(410201)
    hash_ids = torch.randint(0, table_rows, (capacity, 24), generator=generator, dtype=torch.int32)
    hash_ids[0, :3] = torch.tensor([0, 4095, 0], dtype=torch.int32)
    hidden = torch.randn(capacity, 4, 5120, generator=generator).to(torch.bfloat16)
    hidden[1].zero_()
    token_mask = torch.ones(capacity, dtype=torch.bool)
    token_mask[2] = False
    return hash_ids, hidden, token_mask, active


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        default=Path("/data/disk/deepseek-v41-flash/v41-engram-layer1-bf16-v1/weights.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--active", type=int, default=7)
    parser.add_argument("-d", "--device", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.active <= args.capacity:
        raise ValueError("active must be in (0, capacity]")

    weights = torch.load(args.weights, map_location="cpu", weights_only=True)
    hash_ids, hidden, token_mask, active = _inputs(int(weights["embed"].shape[0]), args.capacity, args.active)

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm_ascend.ascend_config import init_ascend_config
    import vllm_ascend.ops  # noqa: F401, register custom operators
    from vllm_ascend.models.deepseek_v41.engram.layer import AscendEngram
    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(args.device)
    device = torch.device("npu", args.device)
    config = VllmConfig()
    case_name = f"layer1-cap{args.capacity}-active{active}"
    result = {
        "capture_kind": "vllm_ascend_native_engram_bf16",
        "device": args.device,
        "cases": {},
    }
    with set_current_vllm_config(config), torch.inference_mode():
        init_ascend_config(config)
        enable_custom_op()
        rendezvous = args.output.parent / (args.output.stem + "-rendezvous")
        init_distributed_environment(1, 0, "file://" + str(rendezvous), args.device, "hccl")
        initialize_model_parallel(1, 1, backend="hccl")
        module = AscendEngram(
            SimpleNamespace(hidden_size=5120, hc_mult=4, rms_norm_eps=1.0e-20,
                            engram_max_ngram_size=4, engram_n_heads=8, engram_head_dim=256),
            None, "layers.1.engram",
        )
        module.wkv.weight.data = weights["wkv"].T.contiguous().to(device)
        module.q_weight.data = weights["q_weight"].to(device)
        module.k_weight.data = weights["k_weight"].to(device)
        module.wkv.quant_method.process_weights_after_loading(module.wkv)

        nodes = {}
        def save_projection(_module, _inputs, output):
            nodes["kv"] = (output[0] if isinstance(output, tuple) else output).detach().cpu().clone()

        module.wkv.register_forward_hook(save_projection)
        rows = weights["embed"][hash_ids.long()].flatten(1).to(device)
        output = module(hidden[:active].to(device), rows[:active], token_mask[:active].to(device), None)
        torch.npu.synchronize()
        nodes["lookup"] = rows[:active].cpu().clone()
        nodes["output"] = output.cpu().clone()
        result["cases"][case_name] = {
            "active": active,
            "capacity": args.capacity,
            "nodes": nodes,
            "inputs": {"hash_ids": hash_ids[:active], "hidden": hidden[:active],
                       "token_mask": token_mask[:active]},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(f"capture={args.output} case={case_name} nodes={sorted(nodes)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
