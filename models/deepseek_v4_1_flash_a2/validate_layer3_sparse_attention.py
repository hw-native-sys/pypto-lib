"""Validate the L3 ratio-2 sparse attention against vLLM-Ascend on A2/A3.

The fixture deliberately uses the A2/A3 cache ABI: BF16 paged window and
compressed KV, INT32 sparse indices, and FP32 attention sinks.  vLLM's native
``npu_sparse_flash_mla`` consumes the paged form while PyPTO consumes the
same rows after a deterministic host-side gather.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .sparse_attention import make_sparse_attention, sparse_attention_reference


CAPACITY = 16
ACTIVE = 7
HEADS = 64
HEAD_DIM = 512
WINDOW = 128
COMPRESSED_TOPK = 512
COMPRESSED_PAGES = 4
VALUE_PADDING = 16
STRICT_RTOL = 8.0e-2
STRICT_ATOL = 8.0e-2
NPU_BASELINE_RTOL = 1.0e-1
NPU_BASELINE_ATOL = 1.0e-1


def _fixture(capacity: int, active: int, compressed_indices: torch.Tensor | None = None):
    generator = torch.Generator().manual_seed(410300)
    query = torch.randn(capacity, HEADS, HEAD_DIM, generator=generator).to(torch.bfloat16)
    window_cache = torch.randn(1, WINDOW, 1, HEAD_DIM, generator=generator).to(torch.bfloat16)
    compressed_cache = torch.randn(COMPRESSED_PAGES, WINDOW, 1, HEAD_DIM, generator=generator).to(torch.bfloat16)
    window_indices = torch.arange(WINDOW, dtype=torch.int32).reshape(1, WINDOW).expand(capacity, WINDOW).clone()
    if compressed_indices is None:
        compressed_indices = torch.arange(COMPRESSED_TOPK, dtype=torch.int32).reshape(1, COMPRESSED_TOPK).expand(capacity, COMPRESSED_TOPK).clone()
    else:
        if tuple(compressed_indices.shape) != (capacity, COMPRESSED_TOPK) or compressed_indices.dtype != torch.int32:
            raise ValueError(f"compressed_indices must be INT32 [{capacity}, {COMPRESSED_TOPK}]")
        compressed_indices = compressed_indices.clone()
    sink = torch.randn(HEADS, generator=generator, dtype=torch.float32)
    window_rows = window_cache.reshape(-1, HEAD_DIM)[window_indices.to(torch.long)]
    compressed_valid = compressed_indices.ge(0)
    compressed_rows = compressed_cache.reshape(-1, HEAD_DIM)[compressed_indices.clamp_min(0).to(torch.long)]
    compressed_rows.masked_fill_(~compressed_valid.unsqueeze(-1), 0)
    zero_window = torch.zeros((*window_rows.shape[:-1], VALUE_PADDING), dtype=torch.bfloat16)
    zero_compressed = torch.zeros((*compressed_rows.shape[:-1], VALUE_PADDING), dtype=torch.bfloat16)
    return {
        "query": query,
        "window_cache": window_cache,
        "compressed_cache": compressed_cache,
        "window_indices": window_indices,
        "compressed_indices": compressed_indices,
        "compressed_lengths": compressed_valid.sum(dim=-1, dtype=torch.int32),
        "window_rows": window_rows,
        "compressed_rows": compressed_rows,
        "window_rows_padded": torch.cat((zero_window, window_rows), dim=-1),
        "compressed_rows_padded": torch.cat((zero_compressed, compressed_rows), dim=-1),
        "sink": sink,
        "active": active,
    }


def _vllm_sparse_attention(fixture: dict, device_id: int) -> torch.Tensor:
    """Run the exact vLLM-Ascend A2/A3 sparse attention operator."""

    import torch_npu  # noqa: F401
    from vllm_ascend.utils import enable_custom_op

    enable_custom_op()
    torch.npu.set_device(device_id)
    device = torch.device("npu", device_id)
    active = int(fixture["active"])
    query = fixture["query"][:active].to(device)
    window_cache = fixture["window_cache"].to(device)
    compressed_cache = fixture["compressed_cache"].to(device)
    sink = fixture["sink"].to(device)
    # Decode batches are one request per active token.  Treating the active
    # rows as one multi-token request changes the band/causal interpretation
    # inside SparseFlashMla and does not match vLLM-Ascend's decode layout.
    starts = torch.arange(active + 1, dtype=torch.int32, device=device)
    ori_lengths = torch.full((active,), WINDOW, dtype=torch.int32, device=device)
    cmp_lengths = fixture["compressed_lengths"][:active].to(device)
    # Ratio-2 compressed mode requires one residual count per request.
    cmp_residual = torch.remainder(cmp_lengths, 2)
    cmp_indices = fixture["compressed_indices"][:active].unsqueeze(1).to(device)
    ori_blocks = torch.zeros((active, 1), dtype=torch.int32, device=device)
    cmp_blocks = torch.arange(COMPRESSED_PAGES, dtype=torch.int32, device=device).reshape(1, -1)
    cmp_blocks = cmp_blocks.expand(active, -1).contiguous()
    metadata = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
        HEADS,
        1,
        HEAD_DIM,
        cu_seqlens_q=starts,
        seqused_q=None,
        seqused_ori_kv=ori_lengths,
        seqused_cmp_kv=cmp_lengths,
        batch_size=active,
        max_seqlen_q=1,
        max_seqlen_ori_kv=WINDOW,
        max_seqlen_cmp_kv=COMPRESSED_TOPK,
        # A2/A3 reserves ori_topk for the compressed path; it must be zero
        # when both cache planes are enabled.
        ori_topk=0,
        cmp_topk=COMPRESSED_TOPK,
        cmp_ratio=2,
        ori_topk_length=None,
        cmp_topk_length=None,
        cmp_residual_kv=cmp_residual,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=WINDOW - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
        has_ori_kv=True,
        has_cmp_kv=True,
    )
    output, _ = torch.ops._C_ascend.npu_sparse_flash_mla(
        query,
        ori_kv=window_cache,
        cmp_kv=compressed_cache,
        # A2/A3 CSA derives the contiguous SWA plane from its block table;
        # passing ori_sparse_indices would select the unsupported ORI+CMP
        # sparse template.
        ori_sparse_indices=None,
        cmp_sparse_indices=cmp_indices,
        ori_block_table=ori_blocks,
        cmp_block_table=cmp_blocks,
        cu_seqlens_q=starts,
        seqused_q=None,
        seqused_ori_kv=ori_lengths,
        seqused_cmp_kv=cmp_lengths,
        cmp_residual_kv=cmp_residual,
        ori_topk_length=None,
        cmp_topk_length=None,
        sinks=sink,
        metadata=metadata,
        softmax_scale=HEAD_DIM**-0.5,
        cmp_ratio=2,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=WINDOW - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
        topk_value_mode=1,
        return_softmax_lse=False,
    )
    torch.npu.synchronize()
    return output.detach().cpu()


def _specs(fixture: dict):
    from golden import ScalarSpec, TensorSpec

    capacity = fixture["query"].shape[0]
    return [
        TensorSpec("query", [capacity, HEADS, HEAD_DIM], torch.bfloat16, init_value=fixture["query"]),
        TensorSpec("window_rows", [capacity, WINDOW, HEAD_DIM + VALUE_PADDING], torch.bfloat16, init_value=fixture["window_rows_padded"]),
        TensorSpec("window_indices", [capacity, WINDOW], torch.int32, init_value=fixture["window_indices"]),
        TensorSpec("compressed_rows", [capacity, COMPRESSED_TOPK, HEAD_DIM + VALUE_PADDING], torch.bfloat16, init_value=fixture["compressed_rows_padded"]),
        TensorSpec("compressed_indices", [capacity, COMPRESSED_TOPK], torch.int32, init_value=fixture["compressed_indices"]),
        TensorSpec("sink", [HEADS], torch.float32, init_value=fixture["sink"]),
        TensorSpec(
            "output",
            [capacity, HEADS, HEAD_DIM + VALUE_PADDING],
            torch.bfloat16,
            init_value=torch.zeros((capacity, HEADS, HEAD_DIM + VALUE_PADDING), dtype=torch.bfloat16),
        ),
        ScalarSpec("active", torch.int32, int(fixture["active"])),
    ]


def _copy_expected(values: dict, expected: torch.Tensor, active: int):
    values["output"].zero_()
    values["output"][:active].copy_(expected[:active])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=int, default=CAPACITY)
    parser.add_argument("--active", type=int, default=ACTIVE)
    parser.add_argument("--sim", action="store_true", help="run PyPTO simulator and compare to Torch reference")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--actual-out", type=Path)
    parser.add_argument("--vllm-out", type=Path)
    parser.add_argument(
        "--compressed-indices",
        type=Path,
        help="load prepared INT32 [capacity, 512] indices from a previous L3 stage",
    )
    args = parser.parse_args()
    if args.capacity % 16 or not 0 < args.active <= args.capacity:
        raise ValueError("capacity must be a positive multiple of 16 and active must be in range")

    compressed_indices = None
    if args.compressed_indices is not None:
        payload = torch.load(args.compressed_indices, map_location="cpu", weights_only=False)
        nodes = payload.get("nodes", payload)
        if "indices" not in nodes:
            raise ValueError(f"{args.compressed_indices} does not contain nodes['indices']")
        compressed_indices = nodes["indices"].to(dtype=torch.int32, device="cpu")
        print(f"[L3] using prepared compressed indices: {args.compressed_indices}", flush=True)
    fixture = _fixture(args.capacity, args.active, compressed_indices)
    torch_expected = sparse_attention_reference(
        fixture["query"], fixture["window_rows"], fixture["window_indices"],
        fixture["compressed_rows"], fixture["compressed_indices"], fixture["sink"],
        active=args.active,
    )
    expected_padded = torch.cat(
        (torch.zeros((*torch_expected.shape[:-1], VALUE_PADDING), dtype=torch.bfloat16), torch_expected),
        dim=-1,
    )
    vllm_expected = torch_expected
    if not args.sim:
        vllm_expected = _vllm_sparse_attention(fixture, args.device)
        diff = (vllm_expected[:args.active].float() - torch_expected[:args.active].float()).abs()
        strict_close = torch.isclose(
            vllm_expected[:args.active].float(), torch_expected[:args.active].float(),
            rtol=STRICT_RTOL, atol=STRICT_ATOL,
        )
        print(
            f"[L3] vLLM vs Torch max_abs={float(diff.max()):.6g} "
            f"mean_abs={float(diff.mean()):.6g} strict_bad={int((~strict_close).sum())}",
            flush=True,
        )
        if args.vllm_out is not None:
            args.vllm_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"capture_kind": "vllm_ascend_a2a3_l3_sparse_attention",
                        "nodes": {"output": vllm_expected}, "output": vllm_expected,
                        "inputs": {k: v for k, v in fixture.items() if isinstance(v, torch.Tensor)}}, args.vllm_out)

    from golden import run

    captured = {}

    def compare(actual, expected, *, actual_outputs, **kwargs):
        del kwargs
        value = actual_outputs["output"].detach().cpu().clone()
        captured["output"] = value
        delta = (value[:args.active].float() - expected[:args.active].float()).abs()
        rtol = STRICT_RTOL if args.sim else NPU_BASELINE_RTOL
        atol = STRICT_ATOL if args.sim else NPU_BASELINE_ATOL
        close = torch.isclose(value[:args.active].float(), expected[:args.active].float(), rtol=rtol, atol=atol)
        return bool(close.all()), (
            f"max_abs={float(delta.max()):.6g} mean_abs={float(delta.mean()):.6g} "
            f"bad={int((~close).sum())} gate_rtol={rtol:.3g} gate_atol={atol:.3g}"
        )

    def golden(values: dict):
        _copy_expected(values, expected_padded if args.sim else torch.cat(
            (torch.zeros((*vllm_expected.shape[:-1], VALUE_PADDING), dtype=torch.bfloat16), vllm_expected), dim=-1
        ), args.active)

    result = run(
        fn=make_sparse_attention(
            args.capacity, HEADS, HEAD_DIM, WINDOW, COMPRESSED_TOPK, value_padding=VALUE_PADDING
        ),
        specs=_specs(fixture),
        golden_fn=golden,
        config=dict(backend_type="a2a3", execution_mode=args.sim, device_id=args.device),
        rtol=STRICT_RTOL,
        atol=STRICT_ATOL,
        compare_fn={"output": compare},
    )
    if not args.sim and "output" in captured:
        # Keep the vLLM comparison as the acceptance gate, but also expose
        # the independent CPU distance so a shared vLLM/PyPTO drift is visible.
        pypto_value = captured["output"][:args.active, ..., VALUE_PADDING:].float()
        cpu_delta = (pypto_value - torch_expected[:args.active].float()).abs()
        print(
            f"[L3] PyPTO NPU vs Torch max_abs={float(cpu_delta.max()):.6g} "
            f"mean_abs={float(cpu_delta.mean()):.6g} "
            f"strict_bad={int((cpu_delta > STRICT_ATOL).sum())}",
            flush=True,
        )
    print(f"[L3] PyPTO vs {'Torch' if args.sim else 'vLLM-Ascend NPU'}: {result}", flush=True)
    if args.actual_out is not None and "output" in captured:
        args.actual_out.parent.mkdir(parents=True, exist_ok=True)
        # Persist the same active, unpadded ABI that vLLM returns.  The
        # runtime buffer includes capacity rows and a 16-column alignment pad;
        # those are staging details, not part of the compared boundary.
        saved_nodes = {"output": captured["output"][:args.active, ..., VALUE_PADDING:].clone()}
        torch.save({"capture_kind": "pypto_a2a3_l3_sparse_attention", "nodes": saved_nodes,
                    "inputs": {k: v for k, v in fixture.items() if isinstance(v, torch.Tensor)}}, args.actual_out)
        print(f"[L3] actual_out={args.actual_out}", flush=True)
    if not result.passed:
        raise RuntimeError(result.error)
    print("[L3] ratio-2 BF16 sparse attention PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
