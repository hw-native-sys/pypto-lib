"""Validate the L3 indexer-query boundary on A2/A3.

The NPU path runs the vLLM-Ascend W8A8 query projection, partial RoPE, and
per-head INT8/FP16 quantizer on the same fixture before using those three
named tensors as PyPTO acceptance references.  This keeps the query boundary
independent of QLI metadata while still comparing real vLLM intermediates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .indexer_query import (
    INDEX_HEAD_DIM,
    INDEX_HEADS,
    INDEX_ROPE_DIM,
    indexer_query_reference,
    make_l3_indexer_query,
)


def _spec(name: str, value: torch.Tensor):
    from golden import TensorSpec

    return TensorSpec(name, list(value.shape), value.dtype, init_value=value)


def _scalar(value: int):
    from golden import ScalarSpec

    return ScalarSpec("active", torch.int32, value)


def _fixture(capacity: int, active: int):
    generator = torch.Generator().manual_seed(410330)
    q_lora = torch.randn(capacity, 1280, generator=generator).to(torch.bfloat16)
    weight = torch.randint(-127, 128, (1280, INDEX_HEADS * INDEX_HEAD_DIM), generator=generator, dtype=torch.int8)
    weight_scale = (torch.rand(INDEX_HEADS * INDEX_HEAD_DIM, generator=generator) * 0.02 + 0.001).float()
    positions = torch.arange(capacity, dtype=torch.float32)
    frequencies = 10000.0 ** (-torch.arange(0, INDEX_ROPE_DIM, 2).float() / INDEX_ROPE_DIM)
    angles = positions[:, None] * frequencies[None]
    cosine, sine = angles.cos(), angles.sin()
    return {
        "q_lora": q_lora,
        "weight": weight,
        "weight_scale": weight_scale,
        "cosine": cosine,
        "sine": sine,
        "active": active,
    }


def _vllm_quantize(query: torch.Tensor, device_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    import torch_npu  # noqa: F401

    from vllm_ascend.ops.triton.quantize_indexer_query import quantize_indexer_query
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    torch.npu.set_device(device_id)
    init_device_properties_triton()
    query_npu = query.to(torch.device("npu", device_id))
    payload, scales = quantize_indexer_query(query_npu)
    torch.npu.synchronize()
    return payload.cpu(), scales.cpu()


def _vllm_indexer_query(fixture: dict, active: int, device_id: int) -> dict[str, torch.Tensor]:
    """Run vLLM-Ascend's W8A8 projection, partial RoPE, and Q quantizer."""

    import torch_npu  # noqa: F401

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.utils import enable_custom_op

    from .capture_vllm_qkv import FrozenW8A8Config

    torch.npu.set_device(device_id)
    device = torch.device("npu", device_id)
    description = {"layers.0.indexer.wq_b.weight": "W8A8_DYNAMIC"}
    quant_config = FrozenW8A8Config(description)
    config = VllmConfig()
    config.quant_config = quant_config
    with set_current_vllm_config(config), torch.inference_mode():
        init_ascend_config(config)
        enable_custom_op()
        wq_b = ReplicatedLinear(
            1280,
            INDEX_HEADS * INDEX_HEAD_DIM,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            prefix="layers.0.indexer.wq_b",
            return_bias=False,
            disable_tp=True,
        )
        wq_b.weight.data = fixture["weight"].transpose(0, 1).contiguous().to(device)
        wq_b.weight_scale.data = fixture["weight_scale"][:, None].to(device)
        wq_b.weight_offset.data = torch.zeros_like(wq_b.weight_scale)
        wq_b.quant_method.process_weights_after_loading(wq_b)

        query = wq_b(fixture["q_lora"][:active].to(device))
        if isinstance(query, tuple):
            query = query[0]
        query = query.reshape(active, INDEX_HEADS, INDEX_HEAD_DIM)
        # The Ascend partial-RoPE op consumes a repeated adjacent-pair table.
        cosine = fixture["cosine"][:active].to(device).repeat_interleave(2, dim=-1)
        sine = fixture["sine"][:active].to(device).repeat_interleave(2, dim=-1)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            query.unsqueeze(1),
            cosine.reshape(active, 1, 1, INDEX_ROPE_DIM),
            sine.reshape(active, 1, 1, INDEX_ROPE_DIM),
            rotary_mode="interleave",
            partial_slice=[INDEX_HEAD_DIM - INDEX_ROPE_DIM, INDEX_HEAD_DIM],
            negate_sin=False,
        )
        payload, scales = _vllm_quantize(query, device_id)
        torch.npu.synchronize()
        return {"query": query.cpu(), "payload": payload, "scales": scales}


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

    fixture = _fixture(args.capacity, args.active)
    expected = indexer_query_reference(
        fixture["q_lora"], fixture["weight"], fixture["weight_scale"],
        fixture["cosine"], fixture["sine"],
    )
    vllm_nodes = {name: expected[name] for name in ("query", "payload", "scales")}
    if not args.sim:
        vllm_nodes = _vllm_indexer_query(fixture, args.active, args.device)
        payload_delta = (vllm_nodes["payload"].to(torch.int16) - expected["payload"][:args.active].to(torch.int16)).abs()
        scale_delta = (vllm_nodes["scales"].float() - expected["scales"][:args.active].float()).abs()
        query_delta = (vllm_nodes["query"].float() - expected["query"][:args.active].float()).abs()
        print(
            f"[L3] vLLM indexer query vs Torch max_abs={float(query_delta.max()):.6g}; "
            f"quant payload_max={int(payload_delta.max())} "
            f"scale_max={float(scale_delta.max()):.6g}", flush=True,
        )
    expected_query = vllm_nodes["query"] if not args.sim else expected["query"]
    expected_payload = vllm_nodes["payload"] if not args.sim else expected["payload"]
    expected_scales = vllm_nodes["scales"] if not args.sim else expected["scales"]
    if args.vllm_out is not None:
        args.vllm_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"capture_kind": "vllm_ascend_a2a3_l3_indexer_query", "nodes": vllm_nodes}, args.vllm_out)
        print(f"[L3] vllm_out={args.vllm_out}", flush=True)

    from golden import run

    output_query = torch.zeros(args.capacity, INDEX_HEADS, INDEX_HEAD_DIM, dtype=torch.bfloat16)
    output_payload = torch.zeros(args.capacity, INDEX_HEADS, INDEX_HEAD_DIM, dtype=torch.int8)
    output_scales = torch.zeros(args.capacity, INDEX_HEADS, dtype=torch.float16)
    captured: dict[str, torch.Tensor] = {}

    def golden(values: dict):
        values["query"][:args.active].copy_(expected_query[:args.active])
        values["payload"][:args.active].copy_(expected_payload[:args.active])
        values["scales"][:args.active].copy_(expected_scales[:args.active])

    def compare(name: str, exact: bool = False):
        def _compare(actual, expected_value, *, actual_outputs, **kwargs):
            del actual, kwargs
            value = actual_outputs[name].detach().cpu()
            lhs = value[:args.active]
            rhs = expected_value[:args.active]
            captured[name] = lhs.clone()
            if exact:
                ok = torch.equal(lhs, rhs)
            else:
                ok = torch.allclose(lhs.float(), rhs.float(), rtol=8.0e-2, atol=8.0e-2)
            delta = (lhs.to(torch.float32) - rhs.to(torch.float32)).abs()
            return bool(ok), f"{name} max_abs={float(delta.max()):.6g} bad={int((lhs != rhs).sum()) if exact else int((delta > 8.0e-2).sum())}"

        return _compare

    result = run(
        fn=make_l3_indexer_query(args.capacity),
        specs=[
            _spec("q_lora", fixture["q_lora"]),
            _spec("weight", fixture["weight"]),
            _spec("weight_scale", fixture["weight_scale"]),
            _spec("cosine", fixture["cosine"]),
            _spec("sine", fixture["sine"]),
            _spec("query", output_query),
            _spec("payload", output_payload),
            _spec("scales", output_scales),
            _scalar(args.active),
        ],
        golden_fn=golden,
        config=dict(backend_type="a2a3", execution_mode=args.sim, device_id=args.device),
        rtol=8.0e-2,
        atol=8.0e-2,
        compare_fn={
            "query": compare("query"),
            "payload": compare("payload", exact=True),
            "scales": compare("scales", exact=True),
        },
    )
    print(f"[L3] PyPTO indexer query vs {'Torch' if args.sim else 'vLLM-Ascend NPU'}: {result}", flush=True)
    if not result.passed:
        raise RuntimeError(result.error)
    if args.actual_out is not None:
        args.actual_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "capture_kind": "pypto_a2a3_l3_indexer_query",
            "nodes": {
                "query": captured["query"],
                "payload": captured["payload"],
                "scales": captured["scales"],
            },
        }, args.actual_out)
        print(f"[L3] actual_out={args.actual_out}", flush=True)
    print("[L3] indexer query boundary PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
