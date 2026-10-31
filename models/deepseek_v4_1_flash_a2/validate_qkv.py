"""Simulator validation for the composed A2/A3 QKV W8A8 leaf."""

from __future__ import annotations

import argparse

import pypto.language as pl
import torch

from .qkv import make_qkv_w8a8, qkv_reference
from .quant import quantize_per_channel_int8


CAPACITY = 16
ACTIVE = 5
CONFIG = {
    "hidden_size": 512,
    "q_lora_rank": 256,
    "num_attention_heads": 2,
    "head_dim": 256,
    "qk_rope_head_dim": 64,
    "rms_norm_eps": 1.0e-20,
}

qkv_single_op = make_qkv_w8a8(CAPACITY, CONFIG)


def build_specs():
    from golden import ScalarSpec, TensorSpec

    torch.manual_seed(47)
    value = torch.randn(CAPACITY, 512, dtype=torch.bfloat16)
    weights = {}
    for name, shape in {"wq_a": (512, 256), "wq_b": (256, 512), "wkv": (512, 256)}.items():
        quantized, weights[name + "_scale"] = quantize_per_channel_int8(torch.randn(shape[1], shape[0]))
        weights[name] = quantized.transpose(0, 1).contiguous()
    cosine = torch.randn(CAPACITY, 32)
    sine = torch.randn(CAPACITY, 32)
    return [
        TensorSpec("value", [CAPACITY, 512], torch.bfloat16, init_value=value),
        TensorSpec("wq_a", [512, 256], torch.int8, init_value=weights["wq_a"]),
        TensorSpec("wq_a_scale", [256], torch.float32, init_value=weights["wq_a_scale"].reshape(-1)),
        TensorSpec("q_norm_weight", [256], torch.bfloat16, init_value=torch.randn(256, dtype=torch.bfloat16)),
        TensorSpec("wq_b", [256, 512], torch.int8, init_value=weights["wq_b"]),
        TensorSpec("wq_b_scale", [512], torch.float32, init_value=weights["wq_b_scale"].reshape(-1)),
        TensorSpec("wkv", [512, 256], torch.int8, init_value=weights["wkv"]),
        TensorSpec("wkv_scale", [256], torch.float32, init_value=weights["wkv_scale"].reshape(-1)),
        TensorSpec("kv_norm_weight", [256], torch.bfloat16, init_value=torch.randn(256, dtype=torch.bfloat16)),
        TensorSpec("cosine", [CAPACITY, 32], torch.float32, init_value=cosine),
        TensorSpec("sine", [CAPACITY, 32], torch.float32, init_value=sine),
        TensorSpec("positions", [CAPACITY], torch.int64, init_value=torch.arange(CAPACITY, dtype=torch.int64)),
        TensorSpec("query_latent", [CAPACITY, 256], torch.bfloat16, init_value=torch.full((CAPACITY, 256), -3.0, dtype=torch.bfloat16)),
        TensorSpec("query", [CAPACITY, 512], torch.bfloat16, init_value=torch.full((CAPACITY, 512), -3.0, dtype=torch.bfloat16)),
        TensorSpec("window_kv", [CAPACITY, 256], torch.bfloat16, init_value=torch.full((CAPACITY, 256), -3.0, dtype=torch.bfloat16)),
        TensorSpec("wq_a_output", [CAPACITY, 256], torch.bfloat16, init_value=torch.full((CAPACITY, 256), -3.0, dtype=torch.bfloat16)),
        TensorSpec("wq_b_output", [CAPACITY, 512], torch.bfloat16, init_value=torch.full((CAPACITY, 512), -3.0, dtype=torch.bfloat16)),
        TensorSpec("wkv_output", [CAPACITY, 256], torch.bfloat16, init_value=torch.full((CAPACITY, 256), -3.0, dtype=torch.bfloat16)),
        TensorSpec("wkv_normed_output", [CAPACITY, 256], torch.bfloat16, init_value=torch.full((CAPACITY, 256), -3.0, dtype=torch.bfloat16)),
        TensorSpec("cache", [2, 128, 256], torch.bfloat16, init_value=torch.randn(2, 128, 256, dtype=torch.bfloat16)),
        TensorSpec("cache_lens", [2], torch.int32, init_value=torch.tensor([7, 8], dtype=torch.int32)),
        ScalarSpec("active", torch.int32, ACTIVE),
    ]


def golden(values):
    weights = {name: values[name] for name in ("wq_a", "wq_b", "wkv", "q_norm_weight", "kv_norm_weight")}
    for name in ("wq_a_scale", "wq_b_scale", "wkv_scale"):
        weights[name] = values[name]
    expected = qkv_reference(values["value"], weights, values["cosine"], values["sine"], CONFIG)
    for name in ("query_latent", "query", "window_kv", "wq_a_output", "wq_b_output", "wkv_output", "wkv_normed_output"):
        values[name][:ACTIVE] = expected[name][:ACTIVE]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    from golden import run

    result = run(
        fn=qkv_single_op,
        specs=build_specs(),
        golden_fn=golden,
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        compile_only=args.compile_only,
        rtol=5.0e-2,
        atol=5.0e-2,
    )
    print(result)
    return int(not result.passed)


if __name__ == "__main__":
    raise SystemExit(main())
