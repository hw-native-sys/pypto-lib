"""Validate the real layer-1 BF16 Engram window on A2/A3."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .engram import engram_reference, make_gate, make_lookup
from .staged import make_bf16_linear


DEFAULT_WEIGHTS = Path("/data/disk/deepseek-v41-flash/v41-engram-layer1-bf16-v1/weights.pt")
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


def _capture_compare(dump, name, active):
    def compare(actual, expected, *, actual_outputs, **kwargs):
        del kwargs
        value = actual_outputs["output"].detach().cpu().clone()
        dump[name] = value[:active].clone()
        close = torch.isclose(actual.float(), expected.float(), rtol=8.0e-2, atol=8.0e-2)
        return bool(close.all()), f"{name} mismatch"

    return compare


def _run(name, fn, specs, golden, *, sim, device, compile_only, dump=None, active=None,
         node_name=None):
    from golden import run

    print(f"[L1] {name}: compile", flush=True)
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
    print(f"[L1] {name}: {result}", flush=True)
    if not result.passed:
        raise RuntimeError(f"L1 Engram stage {name} failed: {result.error}")


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
    weights = torch.load(args.weights, map_location="cpu", weights_only=True)
    generator = torch.Generator().manual_seed(410201)
    hash_ids = torch.randint(0, weights["embed"].shape[0], (args.capacity, 24), generator=generator, dtype=torch.int32)
    hash_ids[0, :3] = torch.tensor([0, 4095, 0], dtype=torch.int32)
    hidden = torch.randn(args.capacity, 4, 5120, generator=generator).to(torch.bfloat16)
    hidden[1].zero_()
    token_mask = torch.ones(args.capacity, dtype=torch.int32)
    token_mask[2] = 0
    expected = engram_reference(
        hash_ids, weights["embed"], weights["wkv"], weights["q_weight"], weights["k_weight"], hidden, token_mask
    )
    dump = {} if args.actual_out is not None else None

    lookup = torch.zeros(args.capacity, 6144, dtype=torch.bfloat16)
    _run("lookup", make_lookup(args.capacity, int(weights["embed"].shape[0])), [
        _spec("hash_ids", hash_ids), _spec("table", weights["embed"]), _spec("output", lookup), _scalar(args.active),
    ], lambda v: _active(v, "output", expected["lookup"], args.active), sim=args.sim, device=args.device,
        compile_only=args.compile_only, dump=dump, active=args.active, node_name="lookup")

    lookup_input = expected["lookup"]
    if dump is not None:
        lookup_input = expected["lookup"].clone()
        lookup_input[:args.active] = dump["lookup"]

    kv = torch.zeros(args.capacity, 25600, dtype=torch.bfloat16)
    _run("projection", make_bf16_linear(args.capacity, 6144, 25600), [
        _spec("value", lookup_input), _spec("weight", weights["wkv"]), _spec("output", kv), _scalar(args.active),
    ], lambda v: _active(v, "output", expected["kv"], args.active), sim=args.sim, device=args.device,
        compile_only=args.compile_only, dump=dump, active=args.active, node_name="kv")

    kv_input = expected["kv"]
    if dump is not None:
        kv_input = expected["kv"].clone()
        kv_input[:args.active] = dump["kv"]

    output = torch.zeros_like(hidden)
    channel_weight = (weights["q_weight"].float() * weights["k_weight"].float()).contiguous()
    _run("masked_gate", make_gate(args.capacity), [
        _spec("hidden", hidden), _spec("kv", kv_input), _spec("channel_weight", channel_weight),
        _spec("token_mask", token_mask), _spec("output", output), _scalar(args.active),
    ], lambda v: _active(v, "output", expected["output"], args.active), sim=args.sim, device=args.device,
        compile_only=args.compile_only, dump=dump, active=args.active, node_name="output")
    if args.actual_out is not None:
        args.actual_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "capture_kind": "pypto_a2a3_l1_engram",
            "nodes": dump,
            "inputs": {"hash_ids": hash_ids[:args.active], "hidden": hidden[:args.active],
                       "token_mask": token_mask[:args.active]},
        }, args.actual_out)
        print(f"[L1] actual_out={args.actual_out}", flush=True)
    print("[L1] Engram lookup/projection/gate PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
