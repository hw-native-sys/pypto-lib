"""Align the QLI-to-cache index preparation boundary on A2/A3.

This validator feeds the same native vLLM-Ascend QLI output into
``prepare_indexer_indices`` and the PyPTO implementation.  Chronological
INT32 indices are compared exactly; the logical valid lengths are checked
between vLLM-Ascend and the CPU reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .indexer_prepare import make_indexer_prepare, prepare_indexer_indices_reference
from .indexer_select import indexer_select_reference
from .validate_layer3_indexer_select import KEY_CAPACITY, TOPK, _fixture, _vllm_qli


def _spec(name: str, value: torch.Tensor):
    from golden import TensorSpec

    return TensorSpec(name, list(value.shape), value.dtype, init_value=value)


def _scalar(value: int):
    from golden import ScalarSpec

    return ScalarSpec("active", torch.int32, value)


def _vllm_prepare(selected: torch.Tensor, positions: torch.Tensor, device_id: int):
    import torch_npu  # noqa: F401

    from vllm_ascend.ops.triton.prepare_indexer_indices import prepare_indexer_indices
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    device = torch.device("npu", device_id)
    torch.npu.set_device(device_id)
    init_device_properties_triton()
    selected_npu = selected.to(device)
    positions_npu = positions.to(device)
    lengths = torch.empty((selected.shape[0], 1), dtype=torch.int32, device=device)
    output = prepare_indexer_indices(
        selected_npu,
        positions_npu,
        2,
        lengths_output=lengths,
    )
    torch.npu.synchronize()
    return output.cpu(), lengths.cpu()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--key-capacity", type=int, default=KEY_CAPACITY)
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--actual-out", type=Path)
    parser.add_argument("--vllm-out", type=Path)
    parser.add_argument(
        "--selected-indices",
        type=Path,
        help="load the QLI INT32 output from a previous validation stage",
    )
    args = parser.parse_args()
    if args.capacity <= 0 or args.capacity % 16:
        raise ValueError("capacity must be a positive multiple of 16")
    if args.key_capacity != KEY_CAPACITY or args.topk != TOPK:
        raise ValueError("the A2/A3 preparation ABI uses key_capacity=512 and topk=512")

    fixture = _fixture(args.capacity, args.key_capacity)
    # QLI output is the input to this boundary.  A chain run may provide the
    # previous stage's dump; otherwise retain the standalone validator's
    # native vLLM/CPU fixture generation.
    selected = None
    if args.selected_indices is not None:
        payload = torch.load(args.selected_indices, map_location="cpu", weights_only=False)
        nodes = payload.get("nodes", payload)
        if "indices" not in nodes:
            raise ValueError(f"{args.selected_indices} does not contain nodes['indices']")
        selected = nodes["indices"].to(dtype=torch.int32, device="cpu")
        expected_shape = (args.capacity, args.topk)
        if tuple(selected.shape) != expected_shape:
            raise ValueError(
                f"selected indices from {args.selected_indices} have shape {tuple(selected.shape)}, "
                f"expected {expected_shape}"
            )
        print(f"[L3 prepare] using selected indices: {args.selected_indices}", flush=True)
    elif not args.sim:
        selected = _vllm_qli(fixture, args.device)
    qli_cpu = indexer_select_reference(
        fixture["query"], fixture["query_scale"], fixture["key"], fixture["key_scale"],
        fixture["weights"], fixture["positions"], topk=args.topk, compress_ratio=2,
    )["indices"]
    selected_for_reference = selected if selected is not None else qli_cpu
    expected_cpu = prepare_indexer_indices_reference(
        selected_for_reference,
        fixture["positions"],
        compress_ratio=2,
    )
    if args.sim:
        expected = expected_cpu
    else:
        selected = selected_for_reference.contiguous()
        prepared, lengths = _vllm_prepare(selected, fixture["positions"], args.device)
        expected = {"indices": prepared, "lengths": lengths}
        if not torch.equal(expected_cpu["indices"], prepared) or not torch.equal(expected_cpu["lengths"], lengths):
            raise RuntimeError("vLLM prepare_indexer_indices disagrees with CPU reference")
        print("[L3 prepare] vLLM indices/lengths vs CPU: PASS", flush=True)
        if args.vllm_out is not None:
            args.vllm_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"capture_kind": "vllm_ascend_a2a3_l3_indexer_prepare",
                        "nodes": expected}, args.vllm_out)
            print(f"[L3 prepare] vllm_out={args.vllm_out}", flush=True)

    from golden import run

    raw = selected if selected is not None else expected_cpu["indices"]
    output = torch.full_like(raw, -1)
    captured: dict[str, torch.Tensor] = {}

    def golden(values: dict):
        values["indices"].copy_(expected["indices"])

    def compare(name: str):
        def _compare(actual, expected_value, *, actual_outputs, **kwargs):
            del actual, kwargs
            value = actual_outputs[name].detach().cpu()
            captured[name] = value.clone()
            ok = torch.equal(value, expected_value)
            return bool(ok), f"{name} exact={ok}"
        return _compare

    result = run(
        fn=make_indexer_prepare(args.capacity, args.topk, compress_ratio=2),
        specs=[
            _spec("raw_selected", raw),
            _spec("positions", fixture["positions"]),
            _spec("indices", output),
            _scalar(args.capacity),
        ],
        golden_fn=golden,
        config=dict(backend_type="a2a3", execution_mode=args.sim, device_id=args.device),
        compare_fn={"indices": compare("indices")},
    )
    print(f"[L3 prepare] PyPTO vs {'Torch' if args.sim else 'vLLM-Ascend NPU'}: {result}", flush=True)
    if not result.passed:
        raise RuntimeError(result.error)
    if args.actual_out is not None:
        args.actual_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"capture_kind": "pypto_a2a3_l3_indexer_prepare", "nodes": captured}, args.actual_out)
        print(f"[L3 prepare] actual_out={args.actual_out}", flush=True)
    print("[L3 prepare] visible filter + chronological sort PASS (length reference checked)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
