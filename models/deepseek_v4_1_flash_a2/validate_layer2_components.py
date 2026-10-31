"""Validate real layer-2 A2/A3 compressor and indexer components."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .index_cache import dequantize_index_cache, quantize_index_cache
from .norm_rope import rms_norm_reference
from .qkv import qkv_reference
from .staged import make_bf16_linear, make_fp32_linear, make_index_quantize, make_linear, make_norm
from .weights import load_tensor_bundle


DEFAULT_WEIGHTS = Path("/data/disk/deepseek-v41-flash/v41-layer2-a2-w8a8-20261021-v3/layer2-a2-w8a8.pt")
CAPACITY = 16
ACTIVE = 7


def _spec(name, value):
    from golden import TensorSpec

    return TensorSpec(name, list(value.shape), value.dtype, init_value=value)


def _scalar(value):
    from golden import ScalarSpec

    return ScalarSpec("active", torch.int32, value)


def _active(values, name, expected, active):
    values[name][:active].copy_(expected[:active])


def _capture_compare(dump, node_name, output_name, active):
    def compare(actual, expected, *, actual_outputs, **kwargs):
        del kwargs
        value = actual_outputs[output_name].detach().cpu().clone()
        dump[node_name] = value[:active].clone()
        close = torch.isclose(actual.float(), expected.float(), rtol=8.0e-2, atol=8.0e-2)
        return bool(close.all()), f"{node_name} mismatch"

    return compare


def _run(name, fn, specs, golden, *, sim, device, compile_only, dump=None, active=None,
         output_names=("output",), node_names=None):
    from golden import run

    print(f"[L2] {name}: compile", flush=True)
    compare_fn = None
    if dump is not None:
        if active is None:
            raise ValueError("active is required when capturing a stage")
        node_names = node_names or output_names
        if len(node_names) != len(output_names):
            raise ValueError("node_names and output_names must have the same length")
        compare_fn = {
            output_name: _capture_compare(dump, node_name, output_name, active)
            for output_name, node_name in zip(output_names, node_names)
        }
    result = run(
        fn=fn,
        specs=specs,
        golden_fn=golden,
        config=dict(backend_type="a2a3", execution_mode=sim, device_id=device),
        compile_only=compile_only,
        rtol=8.0e-2,
        atol=8.0e-2,
        compare_fn=compare_fn,
    )
    print(f"[L2] {name}: {result}", flush=True)
    if not result.passed:
        raise RuntimeError(f"L2 component {name} failed: {result.error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--actual-out", type=Path, help="save active PyPTO named nodes")
    parser.add_argument("--capacity", type=int, default=CAPACITY)
    parser.add_argument("--active", type=int, default=ACTIVE)
    args = parser.parse_args()
    if args.capacity <= 0 or args.capacity % 16 or not 0 < args.active <= args.capacity:
        raise ValueError("capacity must be a positive multiple of 16 and active must be in (0, capacity]")
    weights = load_tensor_bundle(args.weights)
    generator = torch.Generator().manual_seed(410202)
    hidden = 5120
    latent = 1280
    value = torch.randn(args.capacity, hidden, generator=generator).to(torch.bfloat16)
    positions = torch.arange(args.capacity, dtype=torch.int64)
    rope_dim = 64
    frequencies = 10000.0 ** (-torch.arange(0, rope_dim, 2).float() / rope_dim)
    angles = positions.float()[:, None] * frequencies[None]
    cosine, sine = angles.cos(), angles.sin()
    qkv = qkv_reference(
        value,
        {
            "wq_a": weights["wq_a"], "wq_b": weights["wq_b"], "wkv": weights["wkv"],
            "wq_a_scale": weights["wq_a_scale"], "wq_b_scale": weights["wq_b_scale"], "wkv_scale": weights["wkv_scale"],
            "q_norm_weight": weights["q_norm"], "kv_norm_weight": weights["kv_norm"],
        }, cosine, sine,
        {"hidden_size": hidden, "q_lora_rank": latent, "num_attention_heads": 64,
         "head_dim": 512, "qk_rope_head_dim": rope_dim, "rms_norm_eps": 1.0e-20},
    )
    qlatent = qkv["query_latent"]
    dump = {} if args.actual_out is not None else None
    query_for_index = torch.zeros(args.capacity, 4096, dtype=torch.bfloat16)
    from .quant import w8a8_linear

    expected_index_query = w8a8_linear(
        qlatent,
        weights["index_wq_b"].transpose(0, 1).contiguous(),
        weights["index_wq_b_scale"],
    )
    _run("index_query_w8a8", make_linear(args.capacity, latent, 4096), [
        _spec("value", qlatent), _spec("weight", weights["index_wq_b"]), _spec("channel_scale", weights["index_wq_b_scale"]),
        _spec("output", query_for_index), _scalar(args.active),
    ], lambda v: _active(v, "output", expected_index_query, args.active), sim=args.sim, device=args.device,
        compile_only=args.compile_only, dump=dump, active=args.active, node_names=("index_query",))

    projected_weights = torch.zeros(args.capacity, 32, dtype=torch.bfloat16)
    expected_weights = torch.matmul(value.float(), weights["index_weights_proj"].float()).to(torch.bfloat16)
    _run("index_weights_proj", make_bf16_linear(args.capacity, hidden, 32), [
        _spec("value", value), _spec("weight", weights["index_weights_proj"]), _spec("output", projected_weights), _scalar(args.active),
    ], lambda v: _active(v, "output", expected_weights, args.active), sim=args.sim, device=args.device,
        compile_only=args.compile_only, dump=dump, active=args.active, node_names=("index_weights_proj",))

    compressor = weights["compressor_wkv"]
    compressor_gate = weights["compressor_wgate"]
    compressor_wkv = torch.zeros(args.capacity, 512, dtype=torch.bfloat16)
    compressor_gate_out = torch.zeros_like(compressor_wkv)
    expected_compressor = torch.matmul(value.float(), compressor.float()).to(torch.bfloat16)
    expected_gate = torch.matmul(value.float(), compressor_gate.float()).to(torch.bfloat16)
    _run("compressor_wkv", make_bf16_linear(args.capacity, hidden, 512), [
        _spec("value", value), _spec("weight", compressor), _spec("output", compressor_wkv), _scalar(args.active),
    ], lambda v: _active(v, "output", expected_compressor, args.active), sim=args.sim, device=args.device,
        compile_only=args.compile_only, dump=dump, active=args.active, node_names=("compressor_wkv",))
    _run("compressor_wgate", make_bf16_linear(args.capacity, hidden, 512), [
        _spec("value", value), _spec("weight", compressor_gate), _spec("output", compressor_gate_out), _scalar(args.active),
    ], lambda v: _active(v, "output", expected_gate, args.active), sim=args.sim, device=args.device,
        compile_only=args.compile_only, dump=dump, active=args.active, node_names=("compressor_wgate",))
    compressor_normed = torch.zeros_like(compressor_wkv)
    expected_normed = rms_norm_reference(expected_compressor, weights["compressor_norm"], 1.0e-20)
    _run("compressor_norm", make_norm(args.capacity, 512, 1.0e-20), [
        _spec("value", expected_compressor), _spec("weight", weights["compressor_norm"]), _spec("output", compressor_normed), _scalar(args.active),
    ], lambda v: _active(v, "output", expected_normed, args.active), sim=args.sim, device=args.device,
        compile_only=args.compile_only, dump=dump, active=args.active, node_names=("compressor_norm",))

    index_key = torch.zeros(args.capacity, 128, dtype=torch.bfloat16)
    expected_key = torch.matmul(expected_normed.float(), weights["index_wk"].float()).to(torch.bfloat16)
    _run("index_key_proj", make_bf16_linear(args.capacity, 512, 128), [
        _spec("value", expected_normed), _spec("weight", weights["index_wk"]), _spec("output", index_key), _scalar(args.active),
    ], lambda v: _active(v, "output", expected_key, args.active), sim=args.sim, device=args.device,
        compile_only=args.compile_only, dump=dump, active=args.active, node_names=("index_key_proj",))
    index_key_normed = torch.zeros_like(index_key)
    expected_key_normed = rms_norm_reference(expected_key, weights["index_k_norm"], 1.0e-20)
    _run("index_key_norm", make_norm(args.capacity, 128, 1.0e-20), [
        _spec("value", expected_key), _spec("weight", weights["index_k_norm"]), _spec("output", index_key_normed), _scalar(args.active),
    ], lambda v: _active(v, "output", expected_key_normed, args.active), sim=args.sim, device=args.device,
        compile_only=args.compile_only, dump=dump, active=args.active, node_names=("index_key_norm",))

    # Exercise the exact A2/A3 index-cache ABI: INT8 payload plus FP16 scale.
    index_vectors = expected_key_normed.reshape(args.capacity, 1, 128).expand(args.capacity, 32, 128).contiguous()
    payload, scales = quantize_index_cache(index_vectors)
    payload_actual = torch.zeros_like(payload)
    scales_actual = torch.zeros_like(scales)
    _run("index_cache", make_index_quantize(args.capacity, 32), [
        _spec("value", index_vectors), _spec("payload", payload_actual), _spec("scales", scales_actual), _scalar(args.active),
    ], lambda v: (v["payload"][:args.active].copy_(payload[:args.active]), v["scales"][:args.active].copy_(scales[:args.active])),
        sim=args.sim, device=args.device, compile_only=args.compile_only, dump=dump, active=args.active,
        output_names=("payload", "scales"), node_names=("index_cache_payload", "index_cache_scales"))
    reconstructed = dequantize_index_cache(payload[:args.active], scales[:args.active])
    print(f"[L2] index cache max reconstruction error={float((reconstructed - index_vectors[:args.active].float()).abs().max()):.6g}", flush=True)
    if args.actual_out is not None:
        args.actual_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"capture_kind": "pypto_a2a3_l2_components", "nodes": dump}, args.actual_out)
        print(f"[L2] actual_out={args.actual_out}", flush=True)
    print("[L2] compressor/indexer component PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
