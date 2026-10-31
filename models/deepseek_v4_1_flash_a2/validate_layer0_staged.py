"""Run the real-width L0 assembly as sequential A2/A3 kernel boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .decode_swa import swa_attention_reference
from .norm_rope import rope_reference, rms_norm_reference
from .output_proj import output_projection_reference
from .quant import w8a8_linear
from .qkv import qkv_reference
from .staged import make_grouped_wo_a, make_linear, make_norm, make_rope_kernel, make_swa, make_swiglu
from .swa_cache import gather_swa_cache_reference, write_swa_cache_reference
from .weights import load_layer0_output, load_layer0_qkv, load_layer0_shared_expert


DEFAULT_CONFIG = Path("/data/disk/deepseek-v41-flash/v41-source/config.json")
DEFAULT_QKV = Path("/data/disk/deepseek-v41-flash/v41-w8a8-layer0/w8a8-layer0.pt")
DEFAULT_OUTPUT = Path("/data/disk/deepseek-v41-flash/v41-output-layer0/a2-output-layer0.pt")
DEFAULT_SHARED = Path("/data/disk/deepseek-v41-flash/v41-moe-layer0-a2-w8a8/shared-layer0-w8a8.pt")
CAPACITY = 16
ACTIVE = 7


def _config(path: Path) -> dict:
    value = json.loads(path.read_text())
    return value.get("text_config", value)


def _capture_compare(dump, node_name, active):
    def compare(actual, expected, *, actual_outputs, **kwargs):
        del kwargs
        dump[node_name] = actual_outputs["output"].detach().cpu()[:active].clone()
        close = torch.isclose(actual.float(), expected.float(), rtol=8.0e-2, atol=8.0e-2)
        return bool(close.all()), f"{node_name} mismatch"

    return compare


def _run(name, fn, specs, golden, *, sim: bool, device: int, compile_only: bool,
         dump=None, node_name=None, active=None):
    from golden import run

    print(f"[L0] {name}: compile", flush=True)
    compare_fn = None
    if dump is not None:
        if active is None:
            raise ValueError("active is required when capturing a stage")
        compare_fn = {"output": _capture_compare(dump, node_name or name, active)}
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
    print(f"[L0] {name}: {result}", flush=True)
    if not result.passed:
        raise RuntimeError(f"L0 stage {name} failed: {result.error}")
    return result


def _spec(name, value):
    from golden import TensorSpec

    return TensorSpec(name, list(value.shape), value.dtype, init_value=value)


def _scalar(value: int):
    from golden import ScalarSpec

    return ScalarSpec("active", torch.int32, value)


def _copy_active(values, name: str, expected: torch.Tensor, active: int):
    values[name][:active].copy_(expected[:active])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--qkv", type=Path, default=DEFAULT_QKV)
    parser.add_argument("--output-weights", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shared-weights", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--capacity", type=int, default=CAPACITY)
    parser.add_argument("--active", type=int, default=ACTIVE)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--actual-out", type=Path, help="save selected active PyPTO named nodes")
    args = parser.parse_args()
    if args.capacity % 16 or not 0 <= args.active <= args.capacity:
        raise ValueError("capacity must be a multiple of 16 and active must be in range")

    config = _config(args.config)
    qkv = load_layer0_qkv(args.qkv)
    output = load_layer0_output(args.output_weights)
    shared = load_layer0_shared_expert(args.shared_weights)
    hidden = int(config["hidden_size"])
    latent = int(config["q_lora_rank"])
    heads = int(config["num_attention_heads"])
    head_dim = int(config["head_dim"])
    width = heads * head_dim
    rope_dim = int(config["qk_rope_head_dim"])
    pairs = rope_dim // 2
    groups = int(config["o_groups"])
    rank = int(config["o_lora_rank"])
    window = int(config["sliding_window"])
    generator = torch.Generator().manual_seed(410200)
    value = torch.randn(args.capacity, hidden, generator=generator).to(torch.bfloat16)
    positions = torch.arange(args.capacity, dtype=torch.int64)
    frequencies = float(config["rope_theta"]) ** (-torch.arange(0, rope_dim, 2).float() / rope_dim)
    angles = positions.float()[:, None] * frequencies[None]
    cosine, sine = angles.cos(), angles.sin()
    slots = torch.full((args.capacity, 2), -1, dtype=torch.int64)
    slots[:args.active, 0] = 0
    slots[:args.active, 1] = torch.arange(args.active, dtype=torch.int64)
    indices = torch.arange(window, dtype=torch.int32).reshape(1, window).expand(args.capacity, window).clone()
    cache = torch.zeros((2, 128, 1, head_dim), dtype=torch.bfloat16)
    sink = torch.zeros(heads, dtype=torch.float32)
    dump = {} if args.actual_out is not None else None

    qkv_weights = {name: qkv[name] for name in ("wq_a", "wq_b", "wkv", "q_norm_weight", "kv_norm_weight")}
    qkv_weights.update({name: qkv[name] for name in ("wq_a_scale", "wq_b_scale", "wkv_scale")})
    reference = qkv_reference(value, qkv_weights, cosine, sine, config)

    qa_linear = torch.zeros(args.capacity, latent, dtype=torch.bfloat16)
    query_latent = torch.zeros_like(qa_linear)
    _run("q_proj_a", make_linear(args.capacity, hidden, latent), [
        _spec("value", value), _spec("weight", qkv["wq_a"]), _spec("channel_scale", qkv["wq_a_scale"]),
        _spec("output", qa_linear), _scalar(args.active),
    ], lambda v: _copy_active(v, "output", reference["wq_a_output"], args.active), sim=args.sim, device=args.device, compile_only=args.compile_only)
    _run("q_rmsnorm", make_norm(args.capacity, latent, float(config.get("rms_norm_eps", 1e-20))), [
        _spec("value", reference["wq_a_output"]), _spec("weight", qkv["q_norm_weight"]), _spec("output", query_latent),
        _scalar(args.active),
    ], lambda v: _copy_active(v, "output", reference["query_latent"], args.active), sim=args.sim, device=args.device, compile_only=args.compile_only)

    qb_linear = torch.zeros(args.capacity, width, dtype=torch.bfloat16)
    _run("q_proj_b", make_linear(args.capacity, latent, width), [
        _spec("value", reference["query_latent"]), _spec("weight", qkv["wq_b"]), _spec("channel_scale", qkv["wq_b_scale"]),
        _spec("output", qb_linear), _scalar(args.active),
    ], lambda v: _copy_active(v, "output", reference["wq_b_output"], args.active), sim=args.sim, device=args.device, compile_only=args.compile_only)
    query = torch.zeros(args.capacity, width, dtype=torch.bfloat16)
    _run("q_rope", make_rope_kernel(args.capacity, heads, head_dim, rope_dim), [
        _spec("value", reference["wq_b_output"]), _spec("cosine", cosine), _spec("sine", sine), _spec("output", query),
        _scalar(args.active),
    ], lambda v: _copy_active(v, "output", reference["query"], args.active), sim=args.sim, device=args.device, compile_only=args.compile_only)

    kv_linear = torch.zeros(args.capacity, head_dim, dtype=torch.bfloat16)
    _run("kv_proj", make_linear(args.capacity, hidden, head_dim), [
        _spec("value", value), _spec("weight", qkv["wkv"]), _spec("channel_scale", qkv["wkv_scale"]),
        _spec("output", kv_linear), _scalar(args.active),
    ], lambda v: _copy_active(v, "output", reference["wkv_output"], args.active), sim=args.sim, device=args.device, compile_only=args.compile_only)
    kv_normed = torch.zeros_like(kv_linear)
    _run("kv_rmsnorm", make_norm(args.capacity, head_dim, float(config.get("rms_norm_eps", 1e-20))), [
        _spec("value", reference["wkv_output"]), _spec("weight", qkv["kv_norm_weight"]), _spec("output", kv_normed),
        _scalar(args.active),
    ], lambda v: _copy_active(v, "output", reference["wkv_normed_output"], args.active), sim=args.sim, device=args.device, compile_only=args.compile_only)
    window_kv = torch.zeros_like(kv_linear)
    _run("kv_rope", make_rope_kernel(args.capacity, 1, head_dim, rope_dim), [
        _spec("value", reference["wkv_normed_output"]), _spec("cosine", cosine), _spec("sine", sine), _spec("output", window_kv),
        _scalar(args.active),
    ], lambda v: _copy_active(v, "output", reference["window_kv"], args.active), sim=args.sim, device=args.device, compile_only=args.compile_only)

    attended = torch.zeros(args.capacity, width, dtype=torch.bfloat16)
    cache_after = write_swa_cache_reference(cache, reference["window_kv"], slots)
    selected = gather_swa_cache_reference(cache_after, indices)
    expected_attended = swa_attention_reference(reference["query"], selected, indices, sink, head_dim, valid_tokens=args.active)
    _run("swa_attention", make_swa(args.capacity, config), [
        _spec("query", reference["query"]), _spec("window_kv", reference["window_kv"]), _spec("slots", slots),
        _spec("indices", indices), _spec("sink", sink), _spec("cache", cache), _spec("output", attended),
        _scalar(args.active),
    ], lambda v: (_copy_active(v, "output", expected_attended, args.active), v["cache"].copy_(cache_after)), sim=args.sim, device=args.device, compile_only=args.compile_only)

    rotated = torch.zeros_like(attended)
    inverse_sine = -sine
    expected_rotated = rope_reference(expected_attended, cosine, inverse_sine, heads, head_dim, rope_dim)
    _run("inverse_rope", make_rope_kernel(args.capacity, heads, head_dim, rope_dim), [
        _spec("value", expected_attended), _spec("cosine", cosine), _spec("sine", inverse_sine), _spec("output", rotated),
        _scalar(args.active),
    ], lambda v: _copy_active(v, "output", expected_rotated, args.active), sim=args.sim, device=args.device, compile_only=args.compile_only)
    grouped = torch.zeros(args.capacity, groups * rank, dtype=torch.bfloat16)
    expected_grouped = output_projection_reference(expected_attended, output["wo_a"], output["wo_b"], output["wo_b_scale"], cosine, sine, heads, head_dim, rope_dim)["grouped"]
    _run("wo_a", make_grouped_wo_a(args.capacity, heads, head_dim, groups, rank), [
        _spec("rotated", expected_rotated), _spec("wo_a", output["wo_a"]), _spec("grouped", grouped),
        _scalar(args.active),
    ], lambda v: _copy_active(v, "grouped", expected_grouped, args.active), sim=args.sim, device=args.device, compile_only=args.compile_only)
    attention_output = torch.zeros(args.capacity, hidden, dtype=torch.bfloat16)
    expected_output = output_projection_reference(expected_attended, output["wo_a"], output["wo_b"], output["wo_b_scale"], cosine, sine, heads, head_dim, rope_dim)["output"]
    _run("wo_b", make_linear(args.capacity, groups * rank, hidden), [
        _spec("value", expected_grouped), _spec("weight", output["wo_b"]), _spec("channel_scale", output["wo_b_scale"]),
        _spec("output", attention_output), _scalar(args.active),
    ], lambda v: _copy_active(v, "output", expected_output, args.active), sim=args.sim, device=args.device, compile_only=args.compile_only)

    gate_up_expected = w8a8_linear(
        value, shared["gate_up_weight"].transpose(0, 1).contiguous(), shared["gate_up_scale"]
    )
    gate, up = gate_up_expected.chunk(2, dim=-1)
    gate = gate.clamp(max=10.0)
    up = up.clamp(-10.0, 10.0)
    sigmoid_bf16 = torch.sigmoid(gate).to(torch.bfloat16)
    silu_bf16 = gate.to(torch.bfloat16) * sigmoid_bf16
    activation_expected = silu_bf16 * up.to(torch.bfloat16)
    shared_expected = w8a8_linear(
        activation_expected, shared["down_weight"].transpose(0, 1).contiguous(), shared["down_scale"]
    )
    gate_up = torch.zeros_like(gate_up_expected)
    intermediate = int(config["moe_intermediate_size"])
    _run("shared_gate_up", make_linear(args.capacity, hidden, intermediate * 2), [
        _spec("value", value), _spec("weight", shared["gate_up_weight"]), _spec("channel_scale", shared["gate_up_scale"]),
        _spec("output", gate_up), _scalar(args.active),
    ], lambda v: _copy_active(v, "output", gate_up_expected, args.active), sim=args.sim, device=args.device,
        compile_only=args.compile_only, dump=dump, node_name="shared_gate_up", active=args.active)
    gate_up_input = gate_up_expected
    if dump is not None:
        # Chain the captured PyPTO GEMM into the following activation so the
        # named nodes represent the same intermediate tensors as vLLM.
        gate_up_input = gate_up_expected.clone()
        gate_up_input[:args.active] = dump["shared_gate_up"]
        gate_input, up_input = gate_up_input.chunk(2, dim=-1)
        gate_input = gate_input.clamp(max=10.0)
        up_input = up_input.clamp(-10.0, 10.0)
        sigmoid_input = torch.sigmoid(gate_input).to(torch.bfloat16)
        silu_input = gate_input.to(torch.bfloat16) * sigmoid_input
        activation_expected = silu_input * up_input.to(torch.bfloat16)
    activation = torch.zeros_like(activation_expected)
    _run("shared_swiglu", make_swiglu(args.capacity, intermediate), [
        _spec("gate_up", gate_up_input), _spec("output", activation), _scalar(args.active),
    ], lambda v: _copy_active(v, "output", activation_expected, args.active), sim=args.sim, device=args.device,
        compile_only=args.compile_only, dump=dump, node_name="shared_activation", active=args.active)
    # A named vLLM capture represents a chained forward.  Feed the actual
    # PyPTO SwiGLU tensor into the following INT8 re-quantization in that mode;
    # the independent path keeps the CPU reference input for leaf validation.
    activation_input = activation if dump is not None else activation_expected
    shared_stage_expected = w8a8_linear(
        activation_input,
        shared["down_weight"].transpose(0, 1).contiguous(),
        shared["down_scale"],
    )
    shared_output = torch.zeros_like(shared_expected)
    _run("shared_down", make_linear(args.capacity, intermediate, hidden), [
        _spec("value", activation_input), _spec("weight", shared["down_weight"]), _spec("channel_scale", shared["down_scale"]),
        _spec("output", shared_output), _scalar(args.active),
    ], lambda v: _copy_active(v, "output", shared_stage_expected, args.active), sim=args.sim, device=args.device,
        compile_only=args.compile_only, dump=dump, node_name="shared_output", active=args.active)
    combined = (expected_output.float() + shared_expected.float()).to(torch.bfloat16)
    print(
        f"[L0] attention/shared/attention_plus_shared: {tuple(expected_output.shape)}/"
        f"{tuple(shared_expected.shape)}/{tuple(combined.shape)}",
        flush=True,
    )
    if args.actual_out is not None:
        args.actual_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"capture_kind": "pypto_a2a3_l0_shared", "nodes": dump}, args.actual_out)
        print(f"[L0] actual_out={args.actual_out}", flush=True)
    print("[L0] staged attention + shared assembly PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
