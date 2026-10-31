"""Validate the L0 routed W8A8 expert at the host/EP boundary.

The router and expert dispatch are intentionally kept on the host side of the
single-card PyPTO validator.  The selected token slab then goes through the
same three A2/A3 kernels as a routed expert on an EP rank: W8A8 gate, W8A8 up,
BF16 SwiGLU, and W8A8 down.  Only the selected experts are read from the
converted safetensors file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .layer0 import route_reference
from .moe import routed_expert_reference
from .quant import w8a8_linear
from .staged import make_linear, make_swiglu
from .weights import open_layer0_moe


DEFAULT_WEIGHTS = Path(
    "/data/disk/deepseek-v41-flash/v41-moe-layer0-a2-w8a8/moe-layer0-w8a8.safetensors"
)
CAPACITY = 16
ACTIVE = 4
HIDDEN = 5120
INTERMEDIATE = 2304
TOP_K = 6


def _spec(name, value):
    from golden import TensorSpec

    return TensorSpec(name, list(value.shape), value.dtype, init_value=value)


def _scalar(value):
    from golden import ScalarSpec

    return ScalarSpec("active", torch.int32, value)


def _copy_active(values, name: str, expected: torch.Tensor, active: int):
    values[name][:active].copy_(expected[:active])


def _run(name, fn, specs, golden, *, sim: bool, device: int, compile_only: bool):
    from golden import run

    print(f"[L0/routed] {name}: compile", flush=True)
    result = run(
        fn=fn,
        specs=specs,
        golden_fn=golden,
        config=dict(backend_type="a2a3", execution_mode=sim, device_id=device),
        compile_only=compile_only,
        rtol=8.0e-2,
        atol=8.0e-2,
    )
    print(f"[L0/routed] {name}: {result}", flush=True)
    if not result.passed:
        raise RuntimeError(f"routed stage {name} failed: {result.error}")


def _swiglu_reference(gate_up: torch.Tensor, intermediate: int) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.float().clamp(max=10.0)
    up = up.float().clamp(-10.0, 10.0)
    sigmoid_bf16 = torch.sigmoid(gate).to(torch.bfloat16)
    silu_bf16 = (gate * sigmoid_bf16.float()).to(torch.bfloat16)
    return (silu_bf16.float() * up).to(torch.bfloat16).reshape(-1, intermediate)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--capacity", type=int, default=CAPACITY)
    parser.add_argument("--active", type=int, default=ACTIVE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--dump", type=Path, help="optional torch.save route/intermediate dump")
    args = parser.parse_args()
    if args.capacity % 16 or not 0 <= args.active <= args.capacity:
        raise ValueError("capacity must be a multiple of 16 and active must be in range")
    if args.top_k != TOP_K:
        raise ValueError("the V4.1 A2/A3 router contract uses top_k=6")

    generator = torch.Generator().manual_seed(410203)
    value = torch.randn(args.capacity, HIDDEN, generator=generator).to(torch.bfloat16)
    input_ids = torch.zeros(args.capacity, dtype=torch.int64)

    with open_layer0_moe(args.weights) as (router, provider):
        cached = {}

        def lazy_provider(expert: int):
            expert = int(expert)
            if expert not in cached:
                cached[expert] = provider(expert)
                print(f"[L0/routed] lazy-loaded expert={expert}", flush=True)
            return cached[expert]

        route = route_reference(
            value[:args.active],
            router["gate.weight"],
            router["gate.bias"],
            lazy_provider,
            top_k=args.top_k,
            input_ids=input_ids[:args.active],
            vision_bias=router.get("gate.bias_vl"),
        )
        route_ids = route["route_ids"]
        route_weights = route["route_weights"]
        counts = torch.bincount(route_ids.reshape(-1), minlength=384)
        selected_expert = int(torch.argmax(counts))
        selected_pairs = [
            (token, rank)
            for token in range(args.active)
            for rank in range(args.top_k)
            if int(route_ids[token, rank]) == selected_expert
        ]
        if not selected_pairs:
            raise RuntimeError("router produced no selected expert pair")
        if len(selected_pairs) > args.capacity:
            raise ValueError("selected expert slab exceeds the configured capacity")

        token_ids = [token for token, _ in selected_pairs]
        route_ranks = [rank for _, rank in selected_pairs]
        slab_active = len(selected_pairs)
        slab = torch.zeros(args.capacity, HIDDEN, dtype=torch.bfloat16)
        slab[:slab_active] = value[token_ids]
        selected_route_weights = torch.zeros(args.capacity, dtype=torch.float32)
        selected_route_weights[:slab_active] = route_weights[token_ids, route_ranks]
        expert = lazy_provider(selected_expert)

        gate_expected = w8a8_linear(
            slab, expert["w1"].transpose(0, 1).contiguous(), expert["w1_scale"]
        )
        up_expected = w8a8_linear(
            slab, expert["w3"].transpose(0, 1).contiguous(), expert["w3_scale"]
        )
        gate_up_expected = torch.cat((gate_expected, up_expected), dim=-1)
        activation_expected = _swiglu_reference(gate_up_expected, INTERMEDIATE)
        down_expected = w8a8_linear(
            activation_expected, expert["w2"].transpose(0, 1).contiguous(), expert["w2_scale"]
        )
        chosen_reference = routed_expert_reference(
            slab[:slab_active],
            torch.zeros((slab_active, 1), dtype=torch.int64),
            torch.ones((slab_active, 1), dtype=torch.float32),
            expert["w1"].transpose(0, 1).contiguous().unsqueeze(0), expert["w1_scale"].unsqueeze(0),
            expert["w2"].transpose(0, 1).contiguous().unsqueeze(0), expert["w2_scale"].unsqueeze(0),
            expert["w3"].transpose(0, 1).contiguous().unsqueeze(0), expert["w3_scale"].unsqueeze(0),
        )
        reference_error = (chosen_reference.float() - down_expected[:slab_active].float()).abs().max()
        if float(reference_error) > 1.0e-5:
            raise RuntimeError(f"routed expert CPU golden mismatch: {float(reference_error):.6g}")

        gate_actual = torch.zeros_like(gate_expected)
        _run("gate", make_linear(args.capacity, HIDDEN, INTERMEDIATE), [
            _spec("value", slab), _spec("weight", expert["w1"]), _spec("channel_scale", expert["w1_scale"]),
            _spec("output", gate_actual), _scalar(slab_active),
        ], lambda v: _copy_active(v, "output", gate_expected, slab_active),
            sim=args.sim, device=args.device, compile_only=args.compile_only)
        up_actual = torch.zeros_like(up_expected)
        _run("up", make_linear(args.capacity, HIDDEN, INTERMEDIATE), [
            _spec("value", slab), _spec("weight", expert["w3"]), _spec("channel_scale", expert["w3_scale"]),
            _spec("output", up_actual), _scalar(slab_active),
        ], lambda v: _copy_active(v, "output", up_expected, slab_active),
            sim=args.sim, device=args.device, compile_only=args.compile_only)
        swiglu_actual = torch.zeros_like(activation_expected)
        _run("swiglu", make_swiglu(args.capacity, INTERMEDIATE), [
            _spec("gate_up", gate_up_expected), _spec("output", swiglu_actual), _scalar(slab_active),
        ], lambda v: _copy_active(v, "output", activation_expected, slab_active),
            sim=args.sim, device=args.device, compile_only=args.compile_only)
        down_actual = torch.zeros_like(down_expected)
        _run("down", make_linear(args.capacity, INTERMEDIATE, HIDDEN), [
            _spec("value", activation_expected), _spec("weight", expert["w2"]), _spec("channel_scale", expert["w2_scale"]),
            _spec("output", down_actual), _scalar(slab_active),
        ], lambda v: _copy_active(v, "output", down_expected, slab_active),
            sim=args.sim, device=args.device, compile_only=args.compile_only)

        weighted_slab = down_expected.float() * selected_route_weights.reshape(-1, 1)
        print(
            f"[L0/routed] logits={tuple(route['logits'].shape)} topk={route_ids.tolist()} "
            f"selected_expert={selected_expert} slab_rows={slab_active} "
            f"cpu_golden_max_abs={float(reference_error):.6g}",
            flush=True,
        )
        print(
            f"[L0/routed] EP host boundary weighted contribution shape={tuple(weighted_slab.shape)} "
            f"route_weight_range=({float(selected_route_weights[:slab_active].min()):.6g},"
            f"{float(selected_route_weights[:slab_active].max()):.6g})",
            flush=True,
        )
        if args.dump is not None:
            # Keep the route metadata from the CPU router, but persist the
            # actual PyPTO stage outputs.  The previous dump wrote
            # ``down_expected`` here, which made a later vLLM comparison
            # unable to detect a device-side routed-kernel mismatch.
            actual_slab = down_actual[:slab_active].detach().cpu().clone()
            torch.save({
                "logits": route["logits"],
                "route_ids": route["route_ids"],
                "route_weights": route["route_weights"],
                "routed_output": route["output"],
                "routed_output_reference": route["output"],
                "selected_expert": selected_expert,
                "slab_token_ids": torch.tensor(token_ids, dtype=torch.int64),
                "slab_route_ranks": torch.tensor(route_ranks, dtype=torch.int64),
                "slab_gate_up": torch.cat((gate_actual, up_actual), dim=-1)[:slab_active].cpu(),
                "slab_activation": swiglu_actual[:slab_active].cpu(),
                "slab_output": actual_slab,
                "slab_output_reference": down_expected[:slab_active].cpu(),
                "slab_weighted_output": (actual_slab.float() * selected_route_weights[:slab_active, None]).to(torch.bfloat16),
            }, args.dump)
            print(f"[L0/routed] dump={args.dump}", flush=True)
    print("[L0/routed] router + lazy dispatch + routed W8A8 staged PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
