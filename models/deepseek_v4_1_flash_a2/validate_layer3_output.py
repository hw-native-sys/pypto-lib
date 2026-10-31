"""Align the L3 inverse-RoPE/Wo-A/Wo-B boundary with vLLM-Ascend A2/A3."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .norm_rope import rope_reference
from .output_proj import output_projection_reference
from .staged import make_grouped_wo_a, make_linear, make_rope_kernel


HEADS = 64
HEAD_DIM = 512
ROPE_DIM = 64
GROUPS = 8
RANK = 1024
HIDDEN = 5120
WIDTH = HEADS * HEAD_DIM


def _fixture(capacity: int):
    generator = torch.Generator().manual_seed(410340)
    attention = torch.randn(capacity, HEADS, HEAD_DIM, generator=generator).to(torch.bfloat16)
    wo_a = torch.randn(GROUPS, WIDTH // GROUPS, RANK, generator=generator).to(torch.bfloat16)
    wo_b = torch.randint(-127, 128, (GROUPS * RANK, HIDDEN), generator=generator, dtype=torch.int8)
    wo_b_scale = (torch.rand(HIDDEN, generator=generator) * 0.02 + 0.001).float()
    positions = torch.arange(capacity, dtype=torch.float32)
    frequencies = 10000.0 ** (-torch.arange(0, ROPE_DIM, 2).float() / ROPE_DIM)
    angles = positions[:, None] * frequencies[None]
    return {
        "attention": attention,
        "wo_a": wo_a,
        "wo_b": wo_b,
        "wo_b_scale": wo_b_scale,
        "cosine": angles.cos(),
        "sine": angles.sin(),
    }


def _vllm_output(fixture: dict, active: int, device_id: int) -> dict[str, torch.Tensor]:
    import torch_npu  # noqa: F401
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.utils import enable_custom_op

    from .capture_vllm_qkv import FrozenW8A8Config

    torch.npu.set_device(device_id)
    device = torch.device("npu", device_id)
    description = {"layers.0.attn.wo_b.weight": "W8A8_DYNAMIC"}
    quant_config = FrozenW8A8Config(description)
    config = VllmConfig()
    config.quant_config = quant_config
    with set_current_vllm_config(config), torch.inference_mode():
        init_ascend_config(config)
        enable_custom_op()
        wo_b = ReplicatedLinear(
            GROUPS * RANK,
            HIDDEN,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            prefix="layers.0.attn.wo_b",
            return_bias=False,
            disable_tp=True,
        )
        wo_b.weight.data = fixture["wo_b"].transpose(0, 1).contiguous().to(device)
        wo_b.weight_scale.data = fixture["wo_b_scale"][:, None].to(device)
        wo_b.weight_offset.data = torch.zeros_like(wo_b.weight_scale)
        wo_b.quant_method.process_weights_after_loading(wo_b)

        attention = fixture["attention"][:active].to(device)
        cosine = fixture["cosine"][:active].to(device).repeat_interleave(2, dim=-1)
        sine = fixture["sine"][:active].to(device).repeat_interleave(2, dim=-1)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            attention.unsqueeze(1),
            cosine.reshape(active, 1, 1, ROPE_DIM),
            (-sine).reshape(active, 1, 1, ROPE_DIM),
            rotary_mode="interleave",
            partial_slice=[HEAD_DIM - ROPE_DIM, HEAD_DIM],
            negate_sin=False,
        )
        rotated = attention.reshape(active, GROUPS, WIDTH // GROUPS)
        grouped = torch_npu.npu_transpose_batchmatmul(
            rotated,
            fixture["wo_a"].to(device),
            bias=None,
            scale=None,
            perm_x1=(1, 0, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
            batch_split_factor=1,
        ).reshape(active, GROUPS * RANK)
        output = wo_b(grouped)
        torch.npu.synchronize()
        return {
            "rotated": rotated.reshape(active, WIDTH).cpu(),
            "grouped": grouped.cpu(),
            "output": output.cpu(),
        }


def _spec(name: str, value: torch.Tensor):
    from golden import TensorSpec

    return TensorSpec(name, list(value.shape), value.dtype, init_value=value)


def _scalar(value: int):
    from golden import ScalarSpec

    return ScalarSpec("active", torch.int32, value)


def _run_stage(name, fn, specs, expected, active, *, sim, device, output_name="output"):
    from golden import run
    captured = {}

    def golden(values: dict):
        values[output_name][:active].copy_(expected[:active])

    def compare(actual, expected_value, *, actual_outputs, **kwargs):
        del actual, kwargs
        value = actual_outputs[output_name].detach().cpu()
        captured[output_name] = value[:active].clone()
        lhs, rhs = value[:active].float(), expected_value[:active].float()
        close = torch.allclose(lhs, rhs, rtol=1.0e-1, atol=1.0e-1)
        delta = (lhs - rhs).abs()
        return bool(close), f"{name} max_abs={float(delta.max()):.6g} mean_abs={float(delta.mean()):.6g}"

    result = run(
        fn=fn,
        specs=specs,
        golden_fn=golden,
        config=dict(backend_type="a2a3", execution_mode=sim, device_id=device),
        rtol=1.0e-1,
        atol=1.0e-1,
        compare_fn={output_name: compare},
    )
    print(f"[L3 output] {name}: {result}", flush=True)
    if not result.passed:
        raise RuntimeError(result.error)
    return result, captured


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--active", type=int, default=7)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--actual-out", type=Path)
    parser.add_argument("--vllm-out", type=Path)
    args = parser.parse_args()
    if args.capacity <= 0 or args.capacity % 16 or not 0 < args.active <= args.capacity:
        raise ValueError("capacity must be a positive multiple of 16 and active must be in range")
    capacity = args.capacity
    fixture = _fixture(capacity)
    expected = output_projection_reference(
        fixture["attention"].reshape(capacity, WIDTH), fixture["wo_a"], fixture["wo_b"], fixture["wo_b_scale"],
        fixture["cosine"], fixture["sine"], HEADS, HEAD_DIM, ROPE_DIM,
    )
    vllm = {name: expected[name] for name in ("rotated", "grouped", "output")}
    if not args.sim:
        vllm = _vllm_output(fixture, args.active, args.device)
        for name in ("rotated", "grouped", "output"):
            delta = (vllm[name][:args.active].float() - expected[name][:args.active].float()).abs()
            print(f"[L3 output] vLLM {name} vs Torch max_abs={float(delta.max()):.6g}", flush=True)
    if args.vllm_out is not None:
        args.vllm_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"capture_kind": "vllm_ascend_a2a3_l3_output_projection", "nodes": vllm}, args.vllm_out)
        print(f"[L3 output] vllm_out={args.vllm_out}", flush=True)

    inverse_sine = -fixture["sine"]
    rotated_input = fixture["attention"].reshape(capacity, WIDTH)
    rotated_fill = torch.zeros_like(rotated_input)
    _, rotated_actual = _run_stage(
        "inverse_rope",
        make_rope_kernel(capacity, HEADS, HEAD_DIM, ROPE_DIM),
        [_spec("value", rotated_input), _spec("cosine", fixture["cosine"]), _spec("sine", inverse_sine),
         _spec("output", rotated_fill), _scalar(args.active)],
        vllm["rotated"], args.active, sim=args.sim, device=args.device,
    )
    nodes = {"rotated": rotated_actual["output"]}
    delta = (nodes["rotated"].float() - vllm["rotated"][:args.active].float()).abs()
    print(f"[L3 output] PyPTO rotated vs vLLM max_abs={float(delta.max()):.6g}", flush=True)

    rotated_stage = torch.zeros(capacity, WIDTH, dtype=torch.bfloat16)
    rotated_stage[:args.active].copy_(rotated_actual["output"])
    grouped_fill = torch.zeros(capacity, GROUPS * RANK, dtype=torch.bfloat16)
    _, grouped_actual = _run_stage(
        "wo_a",
        make_grouped_wo_a(capacity, HEADS, HEAD_DIM, GROUPS, RANK),
        [_spec("rotated", rotated_stage), _spec("wo_a", fixture["wo_a"]),
         _spec("grouped", grouped_fill), _scalar(args.active)],
        vllm["grouped"], args.active, sim=args.sim, device=args.device,
        output_name="grouped",
    )
    nodes["grouped"] = grouped_actual["grouped"]
    delta = (nodes["grouped"].float() - vllm["grouped"][:args.active].float()).abs()
    print(f"[L3 output] PyPTO grouped vs vLLM max_abs={float(delta.max()):.6g}", flush=True)

    grouped_stage = torch.zeros(capacity, GROUPS * RANK, dtype=torch.bfloat16)
    grouped_stage[:args.active].copy_(grouped_actual["grouped"])
    output_fill = torch.zeros(capacity, HIDDEN, dtype=torch.bfloat16)
    _, output_actual = _run_stage(
        "wo_b",
        make_linear(capacity, GROUPS * RANK, HIDDEN),
        [_spec("value", grouped_stage), _spec("weight", fixture["wo_b"]),
         _spec("channel_scale", fixture["wo_b_scale"]), _spec("output", output_fill), _scalar(args.active)],
        vllm["output"], args.active, sim=args.sim, device=args.device,
    )
    nodes["output"] = output_actual["output"]
    delta = (nodes["output"].float() - vllm["output"][:args.active].float()).abs()
    print(f"[L3 output] PyPTO output vs vLLM max_abs={float(delta.max()):.6g}", flush=True)
    if args.actual_out is not None:
        args.actual_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"capture_kind": "pypto_a2a3_l3_output_projection", "nodes": nodes}, args.actual_out)
    print("[L3 output] inverse RoPE + Wo-A + Wo-B PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
