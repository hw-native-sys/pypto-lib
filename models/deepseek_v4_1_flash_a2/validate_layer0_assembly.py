"""Validate the assembled L0 attention path with the real converted assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .layer0 import layer0_reference, make_l0_attention
from .weights import load_layer0_output, load_layer0_qkv


DEFAULT_CONFIG = Path("/data/disk/deepseek-v41-flash/v41-source/config.json")
DEFAULT_QKV = Path("/data/disk/deepseek-v41-flash/v41-w8a8-layer0/w8a8-layer0.pt")
DEFAULT_OUTPUT = Path("/data/disk/deepseek-v41-flash/v41-output-layer0/a2-output-layer0.pt")
CAPACITY = 16
ACTIVE = 7


def _config(path: Path) -> dict:
    value = json.loads(path.read_text())
    return value.get("text_config", value)


def _specs(config: dict, qkv: dict, output: dict, capacity: int, active: int):
    from golden import ScalarSpec, TensorSpec

    hidden = int(config["hidden_size"])
    latent = int(config["q_lora_rank"])
    heads = int(config["num_attention_heads"])
    head_dim = int(config["head_dim"])
    width = heads * head_dim
    pairs = int(config["qk_rope_head_dim"]) // 2
    window = int(config["sliding_window"])
    generator = torch.Generator().manual_seed(410200)
    value = torch.randn(capacity, hidden, generator=generator).to(torch.bfloat16)
    positions = torch.arange(capacity, dtype=torch.int64)
    frequencies = float(config["rope_theta"]) ** (-torch.arange(0, int(config["qk_rope_head_dim"]), 2).float() / int(config["qk_rope_head_dim"]))
    angles = positions.float()[:, None] * frequencies[None]
    cosine, sine = angles.cos(), angles.sin()
    slots = torch.full((capacity, 2), -1, dtype=torch.int64)
    slots[:active, 0] = 0
    slots[:active, 1] = torch.arange(active, dtype=torch.int64)
    indices = torch.arange(window, dtype=torch.int32).reshape(1, window).expand(capacity, window).clone()
    cache = torch.zeros((2, 128, 1, head_dim), dtype=torch.bfloat16)
    fill = lambda shape, dtype=torch.bfloat16: torch.zeros(shape, dtype=dtype)
    return [
        TensorSpec("value", [capacity, hidden], torch.bfloat16, init_value=value),
        TensorSpec("wq_a", list(qkv["wq_a"].shape), torch.int8, init_value=qkv["wq_a"]),
        TensorSpec("wq_a_scale", [latent], torch.float32, init_value=qkv["wq_a_scale"]),
        TensorSpec("q_norm_weight", [latent], torch.bfloat16, init_value=qkv["q_norm_weight"]),
        TensorSpec("wq_b", list(qkv["wq_b"].shape), torch.int8, init_value=qkv["wq_b"]),
        TensorSpec("wq_b_scale", [width], torch.float32, init_value=qkv["wq_b_scale"]),
        TensorSpec("wkv", list(qkv["wkv"].shape), torch.int8, init_value=qkv["wkv"]),
        TensorSpec("wkv_scale", [head_dim], torch.float32, init_value=qkv["wkv_scale"]),
        TensorSpec("kv_norm_weight", [head_dim], torch.bfloat16, init_value=qkv["kv_norm_weight"]),
        TensorSpec("wo_a", list(output["wo_a"].shape), torch.bfloat16, init_value=output["wo_a"]),
        TensorSpec("wo_b", list(output["wo_b"].shape), torch.int8, init_value=output["wo_b"]),
        TensorSpec("wo_b_scale", [hidden], torch.float32, init_value=output["wo_b_scale"]),
        TensorSpec("cosine", [capacity, pairs], torch.float32, init_value=cosine),
        TensorSpec("sine", [capacity, pairs], torch.float32, init_value=sine),
        TensorSpec("slots", [capacity, 2], torch.int64, init_value=slots),
        TensorSpec("indices", [capacity, window], torch.int32, init_value=indices),
        TensorSpec("sink", [heads], torch.float32, init_value=torch.zeros(heads)),
        TensorSpec("positions", [capacity], torch.int64, init_value=positions),
        TensorSpec("query_latent", [capacity, latent], torch.bfloat16, init_value=fill((capacity, latent))),
        TensorSpec("query", [capacity, width], torch.bfloat16, init_value=fill((capacity, width))),
        TensorSpec("window_kv", [capacity, head_dim], torch.bfloat16, init_value=fill((capacity, head_dim))),
        TensorSpec("cache", [2, 128, 1, head_dim], torch.bfloat16, init_value=cache),
        TensorSpec("cache_lens", [2], torch.int32, init_value=torch.zeros(2, dtype=torch.int32)),
        TensorSpec("attended", [capacity, width], torch.bfloat16, init_value=fill((capacity, width))),
        TensorSpec("output", [capacity, hidden], torch.bfloat16, init_value=fill((capacity, hidden))),
        ScalarSpec("active", torch.int32, active),
    ]


def _golden(config: dict, values: dict, active: int):
    qkv_names = ("wq_a", "wq_b", "wkv", "q_norm_weight", "kv_norm_weight")
    qkv = {name: values[name] for name in qkv_names}
    qkv.update({name: values[name] for name in ("wq_a_scale", "wq_b_scale", "wkv_scale")})
    output = {name: values[name] for name in ("wo_a", "wo_b", "wo_b_scale")}
    expected = layer0_reference(
        values["value"], qkv, output,
        {"gate_up_weight": torch.zeros((5120, 4608), dtype=torch.int8),
         "gate_up_scale": torch.ones(4608),
         "down_weight": torch.zeros((2304, 5120), dtype=torch.int8),
         "down_scale": torch.ones(5120)},
        cosine=values["cosine"], sine=values["sine"], slots=values["slots"],
        indices=values["indices"], cache=values["cache"], sink=values["sink"],
        active=active, config=config,
    )
    for name in ("query_latent", "query", "window_kv", "attended", "output"):
        values[name].zero_()
        values[name][:active] = expected[name][:active]
    values["cache"][:] = expected["cache_after"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--qkv", type=Path, default=DEFAULT_QKV)
    parser.add_argument("--output-weights", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--capacity", type=int, default=CAPACITY)
    parser.add_argument("--active", type=int, default=ACTIVE)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.active < 0 or args.active > args.capacity:
        raise ValueError("active must be within capacity")
    config = _config(args.config)
    qkv = load_layer0_qkv(args.qkv)
    output = load_layer0_output(args.output_weights)
    compute = make_l0_attention(args.capacity, config)
    from golden import run

    result = run(
        fn=compute,
        specs=_specs(config, qkv, output, args.capacity, args.active),
        golden_fn=lambda values: _golden(config, values, args.active),
        config=dict(backend_type="a2a3", execution_mode=args.sim, device_id=args.device),
        compile_only=args.compile_only,
        rtol=8.0e-2,
        atol=8.0e-2,
    )
    print(result)
    print(f"work_dir={result.work_dir}")
    return int(not result.passed)


if __name__ == "__main__":
    raise SystemExit(main())
