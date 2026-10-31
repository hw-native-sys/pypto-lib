"""Run the real layer-0 W8A8 QKV contract against a CPU golden.

This harness intentionally does not fabricate a vLLM result.  It validates
that the prepared A2/A3 checkpoint assets can be loaded into the exact PyPTO
signature; a vLLM-Ascend capture can then be compared with
``compare_vllm_dump`` using the resulting named tensors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pypto.language as pl
import torch

from .qkv import make_qkv_w8a8, qkv_reference
from .weights import load_layer0_qkv


DEFAULT_CONFIG = Path("/data/disk/deepseek-v41-flash/v41-source/config.json")
DEFAULT_WEIGHTS = Path("/data/disk/deepseek-v41-flash/v41-w8a8-layer0/w8a8-layer0.pt")
DEFAULT_CAPACITY = 16
DEFAULT_ACTIVE = 7


def _load_config(path: Path) -> dict:
    payload = json.loads(path.read_text())
    return payload.get("text_config", payload)


def _make_specs(
    config: dict,
    weights: dict,
    capacity: int,
    active: int,
    fixture_inputs: dict | None = None,
):
    from golden import ScalarSpec, TensorSpec

    hidden = int(config["hidden_size"])
    latent = int(config["q_lora_rank"])
    heads = int(config["num_attention_heads"])
    head_dim = int(config["head_dim"])
    query_width = heads * head_dim
    pairs = int(config["qk_rope_head_dim"]) // 2
    if fixture_inputs is None:
        generator = torch.Generator().manual_seed(410100)
        value = torch.randn((capacity, hidden), generator=generator).to(torch.bfloat16)
        positions = torch.tensor(
            ([0, 1, 127, 128, 65535, 65536, 1048575] * 3)[:capacity],
            dtype=torch.int64,
        )
        frequencies = float(config["rope_theta"]) ** (
            -torch.arange(0, int(config["qk_rope_head_dim"]), 2).float()
            / int(config["qk_rope_head_dim"])
        )
        angles = positions.float()[:, None] * frequencies[None]
        cosine, sine = angles.cos(), angles.sin()
        generator_cache = torch.randn((2, 128, head_dim), generator=generator).to(torch.bfloat16)
    else:
        value = fixture_inputs["x"].to(torch.bfloat16)
        positions = fixture_inputs["positions"].to(torch.int64)
        cosine = fixture_inputs["rope_cos"].to(torch.float32)
        sine = fixture_inputs["rope_sin"].to(torch.float32)
        generator_cache = fixture_inputs["cache"].to(torch.bfloat16)
        expected_shapes = {
            "x": (capacity, hidden),
            "positions": (capacity,),
            "rope_cos": (capacity, pairs),
            "rope_sin": (capacity, pairs),
            "cache": (2, 128, head_dim),
        }
        for name, shape in expected_shapes.items():
            if tuple(fixture_inputs[name].shape) != shape:
                raise ValueError(
                    f"fixture input {name!r} has shape {tuple(fixture_inputs[name].shape)}, "
                    f"expected {shape}"
                )
    output_fill = lambda shape: torch.full(shape, -3.0, dtype=torch.bfloat16)
    return [
        TensorSpec("value", [capacity, hidden], torch.bfloat16, init_value=value),
        TensorSpec("wq_a", list(weights["wq_a"].shape), torch.int8, init_value=weights["wq_a"]),
        TensorSpec("wq_a_scale", [latent], torch.float32, init_value=weights["wq_a_scale"]),
        TensorSpec("q_norm_weight", [latent], torch.bfloat16, init_value=weights["q_norm_weight"]),
        TensorSpec("wq_b", list(weights["wq_b"].shape), torch.int8, init_value=weights["wq_b"]),
        TensorSpec("wq_b_scale", [query_width], torch.float32, init_value=weights["wq_b_scale"]),
        TensorSpec("wkv", list(weights["wkv"].shape), torch.int8, init_value=weights["wkv"]),
        TensorSpec("wkv_scale", [head_dim], torch.float32, init_value=weights["wkv_scale"]),
        TensorSpec("kv_norm_weight", [head_dim], torch.bfloat16, init_value=weights["kv_norm_weight"]),
        TensorSpec("cosine", [capacity, pairs], torch.float32, init_value=cosine),
        TensorSpec("sine", [capacity, pairs], torch.float32, init_value=sine),
        TensorSpec("positions", [capacity], torch.int64, init_value=positions),
        TensorSpec("query_latent", [capacity, latent], torch.bfloat16, init_value=output_fill((capacity, latent))),
        TensorSpec("query", [capacity, query_width], torch.bfloat16, init_value=output_fill((capacity, query_width))),
        TensorSpec("window_kv", [capacity, head_dim], torch.bfloat16, init_value=output_fill((capacity, head_dim))),
        TensorSpec("wq_a_output", [capacity, latent], torch.bfloat16, init_value=output_fill((capacity, latent))),
        TensorSpec("wq_b_output", [capacity, query_width], torch.bfloat16, init_value=output_fill((capacity, query_width))),
        TensorSpec("wkv_output", [capacity, head_dim], torch.bfloat16, init_value=output_fill((capacity, head_dim))),
        TensorSpec("wkv_normed_output", [capacity, head_dim], torch.bfloat16, init_value=output_fill((capacity, head_dim))),
        TensorSpec("cache", [2, 128, head_dim], torch.bfloat16, init_value=generator_cache),
        TensorSpec(
            "cache_lens", [2], torch.int32,
            init_value=(fixture_inputs["cache_lens"].to(torch.int32) if fixture_inputs is not None else torch.tensor([127, 128], dtype=torch.int32)),
        ),
        ScalarSpec("active", torch.int32, active),
    ]


def _golden(config: dict, values: dict, active: int):
    names = ("wq_a", "wq_b", "wkv", "q_norm_weight", "kv_norm_weight")
    weights = {name: values[name] for name in names}
    for name in ("wq_a_scale", "wq_b_scale", "wkv_scale"):
        weights[name] = values[name]
    expected = qkv_reference(values["value"], weights, values["cosine"], values["sine"], config)
    for name in (
        "query_latent", "query", "window_kv", "wq_a_output", "wq_b_output",
        "wkv_output", "wkv_normed_output",
    ):
        values[name][:active] = expected[name][:active]


def _capture_compare(output_path: Path, active: int):
    """Save active PyPTO boundaries while preserving the golden comparison."""

    def compare(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        del expected_outputs, inputs
        dump = {
            "capture_kind": "pypto_a2a3_w8a8_qkv",
            "query_latent": actual_outputs["query_latent"].cpu()[:active].clone(),
            "query": actual_outputs["query"].cpu()[:active].clone(),
            "window_kv": actual_outputs["window_kv"].cpu()[:active].clone(),
            "diagnostics": {
                "wq_a": actual_outputs["wq_a_output"].cpu()[:active].clone(),
                "wq_b": actual_outputs["wq_b_output"].cpu()[:active].clone(),
                "wkv": actual_outputs["wkv_output"].cpu()[:active].clone(),
                "wkv_normed": actual_outputs["wkv_normed_output"].cpu()[:active].clone(),
            },
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dump, output_path)
        lhs, rhs = actual.cpu(), expected.cpu()
        ok = torch.allclose(lhs, rhs, rtol=rtol, atol=atol)
        return bool(ok), ""

    compare.__name__ = "capture_and_allclose"
    return compare


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--fixture", type=Path, help="vLLM fixture containing exact inputs")
    parser.add_argument("--case", default="layer0-cap16-active7", help="fixture case name")
    parser.add_argument("--capacity", type=int, default=DEFAULT_CAPACITY)
    parser.add_argument("--active", type=int, default=DEFAULT_ACTIVE)
    parser.add_argument("--actual-out", type=Path, help="save active PyPTO named dump")
    parser.add_argument("--save-data", action="store_true", help="persist golden inputs/outputs under work_dir")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    fixture_inputs = None
    if args.fixture is not None:
        fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
        cases = fixture.get("cases", [])
        selected = next((case for case in cases if case.get("name") == args.case), None)
        if selected is None:
            available = [case.get("name") for case in cases]
            raise ValueError(f"unknown fixture case {args.case!r}; available={available}")
        args.capacity = int(selected["capacity"])
        args.active = int(selected["active"])
        fixture_inputs = selected["inputs"]
        config = dict(fixture.get("config", _load_config(args.config)))
    else:
        config = _load_config(args.config)
    if args.active < 0 or args.active > args.capacity:
        raise ValueError(f"active={args.active} must be in [0, {args.capacity}]")
    weights = load_layer0_qkv(args.weights)
    compute = make_qkv_w8a8(args.capacity, config)
    from golden import run
    from golden import ratio_allclose

    compare_fn = {
        "query": ratio_allclose(
            atol=5.0e-2,
            rtol=5.0e-2,
            max_error_ratio=1.0e-5,
            valid_rows=args.active,
        ),
    }
    if args.actual_out is not None:
        compare_fn["query_latent"] = _capture_compare(args.actual_out, args.active)

    result = run(
        fn=compute,
        specs=_make_specs(config, weights, args.capacity, args.active, fixture_inputs),
        golden_fn=lambda values: _golden(config, values, args.active),
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        compile_only=args.compile_only,
        rtol=5.0e-2,
        atol=5.0e-2,
        compare_fn=compare_fn,
        save_data=args.save_data,
    )
    print(result)
    print(f"work_dir={result.work_dir}")
    if args.actual_out is not None:
        print(f"actual_out={args.actual_out}")
    return int(not result.passed)


if __name__ == "__main__":
    raise SystemExit(main())
