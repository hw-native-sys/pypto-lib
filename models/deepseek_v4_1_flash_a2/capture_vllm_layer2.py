"""Capture native vLLM-Ascend layer-2 compressor/indexer boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from .qkv import qkv_reference


def _linear(in_features, out_features, weight, device, prefix, quant_config=None):
    from vllm.model_executor.layers.linear import ReplicatedLinear

    layer = ReplicatedLinear(
        in_features, out_features, bias=False, params_dtype=torch.bfloat16,
        quant_config=quant_config, prefix=prefix, return_bias=False, disable_tp=True,
    )
    layer.weight.data = weight.T.contiguous().to(device)
    return layer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        default=Path("/data/disk/deepseek-v41-flash/v41-layer2-a2-w8a8-20261021-v3/layer2-a2-w8a8.pt"))
    parser.add_argument("--config", type=Path,
                        default=Path("/data/disk/deepseek-v41-flash/v41-source/config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--active", type=int, default=7)
    parser.add_argument("-d", "--device", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.active <= args.capacity:
        raise ValueError("active must be in (0, capacity]")

    weights = torch.load(args.weights, map_location="cpu", weights_only=True)
    generator = torch.Generator().manual_seed(410202)
    value = torch.randn(args.capacity, 5120, generator=generator).to(torch.bfloat16)
    positions = torch.arange(args.capacity, dtype=torch.int64)
    rope_dim = 64
    angles = positions.float()[:, None] * (10000.0 ** (-torch.arange(0, rope_dim, 2).float() / rope_dim))[None]
    cosine, sine = angles.cos(), angles.sin()
    qkv = qkv_reference(
        value,
        {"wq_a": weights["wq_a"], "wq_b": weights["wq_b"], "wkv": weights["wkv"],
         "wq_a_scale": weights["wq_a_scale"], "wq_b_scale": weights["wq_b_scale"],
         "wkv_scale": weights["wkv_scale"], "q_norm_weight": weights["q_norm"],
         "kv_norm_weight": weights["kv_norm"]},
        cosine, sine,
        {"hidden_size": 5120, "q_lora_rank": 1280, "num_attention_heads": 64,
         "head_dim": 512, "qk_rope_head_dim": rope_dim, "rms_norm_eps": 1.0e-20},
    )

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm_ascend.ascend_config import init_ascend_config
    import vllm_ascend.ops  # noqa: F401
    from vllm_ascend.ops.layernorm import AscendRMSNorm
    from vllm_ascend.utils import enable_custom_op
    from .capture_vllm_qkv import FrozenW8A8Config

    torch.npu.set_device(args.device)
    device = torch.device("npu", args.device)
    quant_config = FrozenW8A8Config({"layers.2.attn.indexer.wq_b.weight": "W8A8_DYNAMIC"})
    config = VllmConfig()
    config.quant_config = quant_config
    result = {"capture_kind": "vllm_ascend_native_layer2", "device": args.device, "cases": {}}
    case_name = f"layer2-cap{args.capacity}-active{args.active}"
    with set_current_vllm_config(config), torch.inference_mode():
        init_ascend_config(config)
        enable_custom_op()
        rendezvous = args.output.parent / (args.output.stem + "-rendezvous")
        init_distributed_environment(1, 0, "file://" + str(rendezvous), args.device, "hccl")
        initialize_model_parallel(1, 1, backend="hccl")
        index_query = _linear(1280, 4096, weights["index_wq_b"], device, "layers.2.attn.indexer.wq_b",
                              quant_config=quant_config)
        index_query.weight_scale.data = weights["index_wq_b_scale"][:, None].to(device)
        index_query.weight_offset.data = torch.zeros_like(index_query.weight_scale)
        index_query.quant_method.process_weights_after_loading(index_query)

        def norm(weight, width):
            layer = AscendRMSNorm(width, eps=1.0e-20, dtype=torch.bfloat16)
            layer.weight.data = weight.to(device)
            return layer

        weights_proj = _linear(5120, 32, weights["index_weights_proj"], device, "layers.2.attn.indexer.weights_proj")
        compressor_wkv = _linear(5120, 512, weights["compressor_wkv"], device, "layers.2.attn.compressor.wkv")
        compressor_wgate = _linear(5120, 512, weights["compressor_wgate"], device, "layers.2.attn.compressor.wgate")
        compressor_norm = norm(weights["compressor_norm"], 512)
        index_key = _linear(512, 128, weights["index_wk"], device, "layers.2.attn.indexer.wk")
        index_key_norm = norm(weights["index_k_norm"], 128)

        x = value[:args.active].to(device)
        qlatent = qkv["query_latent"][:args.active].to(device)
        nodes = {
            "index_query": index_query(qlatent).detach().cpu(),
            "index_weights_proj": weights_proj(x).detach().cpu(),
            "compressor_wkv": compressor_wkv(x).detach().cpu(),
            "compressor_wgate": compressor_wgate(x).detach().cpu(),
        }
        nodes["compressor_norm"] = compressor_norm(nodes["compressor_wkv"].to(device)).detach().cpu()
        nodes["index_key_proj"] = index_key(nodes["compressor_norm"].to(device)).detach().cpu()
        nodes["index_key_norm"] = index_key_norm(nodes["index_key_proj"].to(device)).detach().cpu()
        vectors = nodes["index_key_norm"].reshape(args.active, 1, 128).expand(args.active, 32, 128).contiguous().to(device)
        payload, scales = torch_npu.npu_dynamic_quant(vectors, dst_type=torch.int8)
        nodes["index_cache_payload"] = payload.cpu()
        nodes["index_cache_scales"] = scales.to(torch.float16).cpu()
        torch.npu.synchronize()
        result["cases"][case_name] = {
            "active": args.active,
            "capacity": args.capacity,
            "nodes": nodes,
            "inputs": {
                "value": value.cpu(),
                "positions": positions,
                "rope_cos": cosine,
                "rope_sin": sine,
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(f"capture={args.output} case={case_name} nodes={sorted(nodes)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
