"""CPU/simulator contract check for the A2/A3 layer-15 reuse path."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .layer14_15 import LAYER15, layer15_reuse_reference, validate_layer14_15_contract
from .precision import compare_tensors


def _fixture(seed: int = 1515, tokens: int = 16, heads: int = 4, head_dim: int = 8):
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(tokens, heads, head_dim, generator=generator).to(torch.bfloat16)
    window = torch.randn(tokens, 4, head_dim, generator=generator).to(torch.bfloat16)
    compressed = torch.randn(tokens, 8, head_dim, generator=generator).to(torch.bfloat16)
    window_indices = torch.full((tokens, 4), -1, dtype=torch.int32)
    compressed_indices = torch.full((tokens, 8), -1, dtype=torch.int32)
    for token in range(tokens):
        window_indices[token, : min(4, token + 1)] = torch.arange(min(4, token + 1), dtype=torch.int32)
        visible = min(8, token // 2 + 1)
        compressed_indices[token, :visible] = torch.arange(visible, dtype=torch.int32)
    sink = torch.randn(heads, generator=generator, dtype=torch.float32)
    return query, window, window_indices, compressed, compressed_indices, sink


def _load_nodes(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and isinstance(payload.get("nodes"), dict):
        payload = payload["nodes"]
    if not isinstance(payload, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in payload.items()):
        raise TypeError(f"{path} must contain a tensor mapping or a 'nodes' tensor mapping")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", action="store_true", help="accepted for parity with NPU validators")
    parser.add_argument("--seed", type=int, default=1515)
    parser.add_argument("--actual-out", type=Path)
    parser.add_argument("--vllm-dump", type=Path)
    args = parser.parse_args()

    validate_layer14_15_contract()
    query, window, window_indices, compressed, compressed_indices, sink = _fixture(args.seed)
    output = layer15_reuse_reference(
        query,
        window,
        window_indices,
        compressed,
        compressed_indices,
        sink,
        active=query.shape[0],
    )
    nodes = {
        "query": query,
        "compressed_indices": compressed_indices,
        "output": output,
    }
    assert nodes["query"].dtype == torch.bfloat16
    assert nodes["compressed_indices"].dtype == torch.int32
    assert nodes["output"].dtype == torch.bfloat16
    if args.actual_out is not None:
        args.actual_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"capture_kind": "a2a3_layer15_reuse", "nodes": nodes}, args.actual_out)
        print(f"[L15] actual_out={args.actual_out}")
    if args.vllm_dump is not None:
        expected = _load_nodes(args.vllm_dump)
        for name, value in nodes.items():
            if name not in expected:
                raise AssertionError(f"vLLM dump is missing L15 node {name}")
            delta = compare_tensors(value, expected[name])
            print(f"[L15] {name}: max_abs={delta.max_abs:.6g} mean_abs={delta.mean_abs:.6g}")
    print(
        f"[L15] reuse PASS: source_layer={LAYER15.reuse_source_layer_id}, "
        f"active={query.shape[0]}, output_dtype={output.dtype}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
