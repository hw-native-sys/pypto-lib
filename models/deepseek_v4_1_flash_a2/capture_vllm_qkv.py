"""Capture layer-0 QKV boundaries with vLLM-Ascend W8A8 on a real NPU.

Some vLLM-Ascend revisions do not register the fused partial-RoPE custom op
outside the full model runner.  This producer still uses vLLM's linear,
quantization and RMSNorm modules, and evaluates the supplied FP32 RoPE table
with an equivalent adjacent-pair Torch expression.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.ops.layernorm import AscendRMSNorm
from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig
from vllm_ascend.quantization.method_adapters import AscendLinearMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import enable_custom_op


class FrozenW8A8Config(AscendModelSlimConfig):
    """Bind the isolated projections to vLLM-Ascend dynamic W8A8."""

    def get_quant_method(self, layer, prefix, tid2eid=None):
        if self.quant_description[prefix + ".weight"] != "W8A8_DYNAMIC":
            raise ValueError("quantization description mismatch")
        return AscendLinearMethod(AscendW8A8DynamicLinearMethod())


class VllmQKVModule(torch.nn.Module):
    """vLLM linear and RMSNorm modules for the isolated QKV boundaries."""

    def __init__(self, config, weights, quant_config, device):
        super().__init__()
        for name in ("wq_a", "wq_b", "wkv"):
            stored = weights[name]
            layer = ReplicatedLinear(
                stored.shape[0], stored.shape[1], bias=False,
                params_dtype=torch.bfloat16, quant_config=quant_config,
                prefix="layers.0.attn." + name, return_bias=False, disable_tp=True,
            )
            layer.weight.data = stored.T.contiguous().to(device)
            layer.weight_scale.data = weights[name + "_scale"][:, None].to(device)
            layer.weight_offset.data = torch.zeros_like(layer.weight_scale)
            layer.quant_method.process_weights_after_loading(layer)
            self.add_module(name, layer)
        for name, width in (("q_norm", config["q_lora_rank"]), ("kv_norm", config["head_dim"])):
            norm = AscendRMSNorm(width, eps=config["rms_norm_eps"], dtype=torch.bfloat16)
            norm.weight.data = weights[name + "_weight"].to(device)
            self.add_module(name, norm)


def _rotate(value, cosine, sine, heads: int, head_dim: int, rope_dim: int):
    shaped = value.reshape(value.shape[0], heads, head_dim)
    prefix = shaped[..., : head_dim - rope_dim]
    tail = shaped[..., head_dim - rope_dim :].float().reshape(value.shape[0], heads, rope_dim // 2, 2)
    even, odd = tail[..., 0], tail[..., 1]
    c = cosine[:, None, :].float()
    s = sine[:, None, :].float()
    rotated = torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).flatten(-2)
    return torch.cat((prefix, rotated.to(torch.bfloat16)), dim=-1).reshape(value.shape[0], -1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("-d", "--device", type=int, default=0)
    args = parser.parse_args()

    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    weights = torch.load(args.weights, map_location="cpu", weights_only=True)
    config = fixture["config"]
    description = {"layers.0.attn." + name + ".weight": "W8A8_DYNAMIC" for name in ("wq_a", "wq_b", "wkv")}
    quant_config = FrozenW8A8Config(description)

    torch.npu.set_device(args.device)
    device = torch.device("npu", args.device)
    vllm_config = VllmConfig()
    vllm_config.quant_config = quant_config
    result = {
        "capture_kind": "vllm_ascend_w8a8_qkv_manual_rope",
        "quant_description": description,
        "device": args.device,
        "cases": {},
    }
    with set_current_vllm_config(vllm_config), torch.inference_mode():
        init_ascend_config(vllm_config)
        enable_custom_op()
        module = VllmQKVModule(config, weights, quant_config, device)
        nodes = {}

        def hook(name):
            def save(_layer, _arguments, value):
                nodes[name] = value.detach().clone()

            return save

        for name in ("wq_a", "wq_b", "wkv"):
            getattr(module, name).register_forward_hook(hook(name))
        for case in fixture["cases"]:
            active = int(case["active"])
            inputs = case["inputs"]
            x = inputs["x"][:active].to(device)
            cosine = inputs["rope_cos"][:active].to(device)
            sine = inputs["rope_sin"][:active].to(device)
            qa = module.wq_a(x)
            latent = module.q_norm(qa)
            qb = module.wq_b(latent)
            kv0 = module.wkv(x)
            kv_normed = module.kv_norm(kv0)
            query = _rotate(qb, cosine, sine, config["num_attention_heads"], config["head_dim"], config["qk_rope_head_dim"])
            window_kv = _rotate(kv_normed, cosine, sine, 1, config["head_dim"], config["qk_rope_head_dim"])
            torch.npu.synchronize()
            result["cases"][case["name"]] = {
                "query_latent": latent.cpu(),
                "query": query.cpu(),
                "window_kv": window_kv.cpu(),
                "diagnostics": {
                    "wq_a": nodes["wq_a"].cpu(),
                    "wq_b": nodes["wq_b"].cpu(),
                    "wkv": nodes["wkv"].cpu(),
                    "wkv_normed": kv_normed.cpu(),
                },
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(json.dumps({"output": str(args.output), "cases": len(result["cases"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
