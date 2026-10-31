"""Align the A2/A3 PyPTO QLI score/Top-K boundary with vLLM-Ascend.

The native QLI operator is allowed to return the selected positions in any
score-equivalent order.  vLLM sorts them in a later ``prepare_indexer_indices``
step, so this validator compares the valid index sets row by row and checks
the PyPTO score matrix against the independent INT32/FP32 CPU oracle.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .indexer_select import indexer_select_reference, make_indexer_select


HEADS = 32
HEAD_DIM = 128
TOPK = 512
KEY_CAPACITY = 512
COMPRESS_RATIO = 2


def _fixture(capacity: int, key_capacity: int):
    generator = torch.Generator().manual_seed(410350)
    query = torch.randint(-127, 128, (capacity, HEADS, HEAD_DIM), generator=generator, dtype=torch.int8)
    query_scale = torch.rand(capacity, HEADS, generator=generator, dtype=torch.float16) * 0.02
    key = torch.randint(-127, 128, (key_capacity, HEAD_DIM), generator=generator, dtype=torch.int8)
    key_scale = torch.rand(key_capacity, generator=generator, dtype=torch.float16) * 0.02
    weights = torch.rand(capacity, HEADS, generator=generator, dtype=torch.float16)
    # Exercise full and partially visible compressed prefixes while retaining
    # the fixed key-cache allocation used by the PyPTO signature.
    visible = torch.full((capacity,), key_capacity, dtype=torch.int32)
    if capacity >= 4:
        visible[1] = key_capacity - key_capacity // 16
        visible[2] = key_capacity // 2
        visible[3] = key_capacity // 8
    positions = visible * COMPRESS_RATIO - 1
    return {
        "query": query,
        "query_scale": query_scale,
        "key": key,
        "key_scale": key_scale,
        "weights": weights,
        "positions": positions,
        "visible": visible,
    }


def _vllm_qli(fixture: dict[str, torch.Tensor], device_id: int) -> torch.Tensor:
    """Run vLLM-Ascend's A2/A3 ``npu_quant_lightning_indexer_v3``."""

    import torch_npu  # noqa: F401
    from vllm_ascend.utils import enable_custom_op

    enable_custom_op()
    torch.npu.set_device(device_id)
    device = torch.device("npu", device_id)
    capacity = fixture["query"].shape[0]
    key_capacity = fixture["key"].shape[0]
    query = fixture["query"].to(device)
    query_scale = fixture["query_scale"].to(device)
    key = fixture["key"].reshape(key_capacity // 128, 128, 1, HEAD_DIM).to(device)
    key_scale = fixture["key_scale"].reshape(key_capacity // 128, 128, 1).to(device)
    weights = fixture["weights"].to(device)
    lengths = fixture["visible"].to(device)
    starts = torch.arange(capacity + 1, dtype=torch.int32, device=device)
    residual = torch.remainder(lengths, COMPRESS_RATIO)
    block_table = torch.arange(key_capacity // 128, dtype=torch.int32, device=device).reshape(1, -1)
    block_table = block_table.expand(capacity, -1).contiguous()
    common = dict(
        cu_seqlens_q=starts,
        seqused_k=lengths,
        cmp_residual_k=residual,
        max_seqlen_q=1,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=COMPRESS_RATIO,
    )
    metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
        HEADS,
        1,
        HEAD_DIM,
        TOPK,
        2,
        batch_size=capacity,
        max_seqlen_k=key_capacity,
        device=str(device),
        **common,
    )
    selected, _, _ = torch.ops._C_ascend.npu_quant_lightning_indexer_v3(
        query,
        key,
        weights,
        query_scale,
        key_scale,
        TOPK,
        2,
        block_table=block_table,
        metadata=metadata,
        candidate_topk_index=None,
        candidate_mode=3,
        candidate_topk_blocks=2048,
        candidate_block_size=8,
        **common,
    )
    torch.npu.synchronize()
    return selected.squeeze(1).cpu()


def _spec(name: str, value: torch.Tensor):
    from golden import TensorSpec

    return TensorSpec(name, list(value.shape), value.dtype, init_value=value)


def _scalar(value: int):
    from golden import ScalarSpec

    return ScalarSpec("active", torch.int32, value)


def _compare_indices(actual: torch.Tensor, expected: torch.Tensor, visible: torch.Tensor):
    messages = []
    for row in range(actual.shape[0]):
        count = int(visible[row])
        row_actual = actual[row]
        # QLI sorts the fixed 512-entry candidate row.  Entries beyond the
        # request-visible prefix are legal here and are filtered by the next
        # prepare_indexer_indices boundary; only malformed negative values are
        # rejected at this stage.
        invalid = row_actual < -1
        if bool(invalid.any()):
            messages.append(
                f"row={row} visible={count} invalid_indices={int(invalid.sum())} "
                f"sample={row_actual[invalid][:8].tolist()}"
            )
            continue
        lhs = row_actual[(row_actual >= 0) & (row_actual < count)].sort().values
        rhs_row = expected[row]
        rhs = rhs_row[(rhs_row >= 0) & (rhs_row < count)].sort().values
        if not torch.equal(lhs, rhs):
            missing = rhs[~torch.isin(rhs, lhs)]
            extra = lhs[~torch.isin(lhs, rhs)]
            messages.append(
                f"row={row} visible={count} actual={lhs[:8].tolist()} expected={rhs[:8].tolist()} "
                f"unique_actual={int(torch.unique(lhs).numel())} missing={missing[:8].tolist()} extra={extra[:8].tolist()}"
            )
    return not messages, "; ".join(messages[:2])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--key-capacity", type=int, default=KEY_CAPACITY)
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--actual-out", type=Path)
    parser.add_argument("--vllm-out", type=Path)
    args = parser.parse_args()
    if args.capacity <= 0 or args.capacity % 16:
        raise ValueError("capacity must be a positive multiple of 16")
    if args.key_capacity != KEY_CAPACITY or args.topk != TOPK:
        raise ValueError("the A2/A3 QLI validation ABI uses key_capacity=512 and topk=512")

    fixture = _fixture(args.capacity, args.key_capacity)
    expected = indexer_select_reference(
        fixture["query"], fixture["query_scale"], fixture["key"], fixture["key_scale"],
        fixture["weights"], fixture["positions"], topk=args.topk, compress_ratio=COMPRESS_RATIO,
    )
    vllm_indices = expected["indices"]
    if not args.sim:
        vllm_indices = _vllm_qli(fixture, args.device)
        ok, detail = _compare_indices(vllm_indices, expected["indices"], fixture["visible"])
        print(f"[L3 QLI] vLLM indices vs CPU set: {'PASS' if ok else 'FAIL'} {detail}", flush=True)
        if not ok:
            raise RuntimeError(detail)
        if args.vllm_out is not None:
            args.vllm_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"capture_kind": "vllm_ascend_a2a3_l3_qli", "nodes": {"indices": vllm_indices}},
                args.vllm_out,
            )
            print(f"[L3 QLI] vllm_out={args.vllm_out}", flush=True)

    from golden import run

    score_output = torch.zeros(args.capacity, args.key_capacity, dtype=torch.float32)
    index_output = torch.full((args.capacity, args.topk), -1, dtype=torch.int32)
    captured: dict[str, torch.Tensor] = {}

    def golden(values: dict):
        values["scores"][:].copy_(expected["scores"])
        values["indices"][:].copy_(vllm_indices if not args.sim else expected["indices"])

    def compare_scores(actual, expected_value, *, actual_outputs, **kwargs):
        del actual, kwargs
        value = actual_outputs["scores"].detach().cpu()
        captured["scores"] = value.clone()
        delta = (value - expected_value).abs()
        close = torch.allclose(value, expected_value, rtol=2.0e-2, atol=2.0e-2)
        return bool(close), f"scores max_abs={float(delta.max()):.6g} mean_abs={float(delta.mean()):.6g}"

    def compare_indices(actual, expected_value, *, actual_outputs, **kwargs):
        del actual, kwargs
        value = actual_outputs["indices"].detach().cpu()
        captured["indices"] = value.clone()
        ok, detail = _compare_indices(value, expected_value, fixture["visible"])
        return bool(ok), detail or "valid index sets exact"

    result = run(
        fn=make_indexer_select(args.capacity, args.key_capacity, topk=args.topk),
        specs=[
            _spec("query", fixture["query"]),
            _spec("query_scale", fixture["query_scale"]),
            _spec("key", fixture["key"]),
            _spec("key_scale", fixture["key_scale"]),
            _spec("weights", fixture["weights"]),
            _spec("positions", fixture["positions"]),
            _spec("scores", score_output),
            _spec("indices", index_output),
            _scalar(args.capacity),
        ],
        golden_fn=golden,
        config=dict(backend_type="a2a3", execution_mode=args.sim, device_id=args.device),
        rtol=2.0e-2,
        atol=2.0e-2,
        compare_fn={"scores": compare_scores, "indices": compare_indices},
    )
    print(f"[L3 QLI] PyPTO vs {'Torch' if args.sim else 'vLLM-Ascend NPU'}: {result}", flush=True)
    if not result.passed:
        raise RuntimeError(result.error)
    if "indices" in captured:
        reference_indices = expected["indices"] if args.sim else vllm_indices
        captured_ok, captured_detail = _compare_indices(
            captured["indices"], reference_indices, fixture["visible"]
        )
        if not captured_ok:
            raise RuntimeError(f"captured QLI indices mismatch: {captured_detail}")
    if args.actual_out is not None:
        args.actual_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"capture_kind": "pypto_a2a3_l3_qli", "nodes": captured}, args.actual_out)
        print(f"[L3 QLI] actual_out={args.actual_out}", flush=True)
    print("[L3 QLI] INT8 score + Top-K boundary PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
