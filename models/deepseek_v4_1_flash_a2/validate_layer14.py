"""CPU/simulator contract check for the A2/A3 layer-14 source path.

The repository does not contain converted L14 checkpoint tensors.  This
validator therefore uses a deterministic frozen fixture and emits named
intermediates in the same format used by the vLLM dump comparison helpers.
Pass ``--vllm-dump`` when a vLLM-Ascend capture is available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .layer14_15 import (
    LAYER14,
    index_key_reference,
    layer14_source_reference,
    ratio2_pool_reference,
    validate_layer14_15_contract,
)
from .indexer_prepare import prepare_indexer_indices_reference
from .precision import compare_tensors


def _fixture(seed: int = 1415, tokens: int = 16, head_dim: int = 32):
    generator = torch.Generator().manual_seed(seed)
    kv = torch.randn(tokens, head_dim, generator=generator, dtype=torch.float32)
    score = torch.randn(tokens, head_dim, generator=generator, dtype=torch.float32)
    previous_kv = torch.randn(tokens, head_dim, generator=generator, dtype=torch.float32)
    previous_score = torch.randn(tokens, head_dim, generator=generator, dtype=torch.float32)
    position_ids = torch.arange(tokens, dtype=torch.int32)
    norm_weight = torch.randn(head_dim, generator=generator).to(torch.bfloat16)
    index_wk = torch.randn(head_dim, 128, generator=generator, dtype=torch.float32)
    index_norm = torch.randn(128, generator=generator).to(torch.bfloat16)
    query = torch.randint(-127, 128, (tokens, 32, 128), generator=generator, dtype=torch.int8)
    query_scale = (torch.rand(tokens, 32, generator=generator) * 0.02 + 0.001).to(torch.float16)
    weights = (torch.rand(tokens, 32, generator=generator) * 0.5 + 0.5).to(torch.float16)
    return {
        "kv_proj": kv,
        "score_proj": score,
        "previous_kv": previous_kv,
        "previous_score": previous_score,
        "norm_weight": norm_weight,
        "position_ids": position_ids,
        "index_wk": index_wk,
        "index_norm_weight": index_norm,
        "qli_query": query,
        "qli_query_scale": query_scale,
        "qli_weights": weights,
    }


def _load_nodes(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and isinstance(payload.get("nodes"), dict):
        payload = payload["nodes"]
    if not isinstance(payload, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in payload.items()):
        raise TypeError(f"{path} must contain a tensor mapping or a 'nodes' tensor mapping")
    return payload


def _compare(nodes: dict[str, torch.Tensor], path: Path) -> None:
    expected = _load_nodes(path)
    missing = sorted(set(nodes) - set(expected))
    if missing:
        raise AssertionError(f"vLLM dump is missing L14 nodes: {missing}")
    for name, value in nodes.items():
        delta = compare_tensors(value, expected[name])
        print(f"[L14] {name}: max_abs={delta.max_abs:.6g} mean_abs={delta.mean_abs:.6g}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", action="store_true", help="accepted for parity with NPU validators")
    parser.add_argument("--seed", type=int, default=1415)
    parser.add_argument("--actual-out", type=Path)
    parser.add_argument("--vllm-dump", type=Path)
    args = parser.parse_args()

    validate_layer14_15_contract()
    fixture = _fixture(args.seed)
    result = layer14_source_reference(**fixture, topk=16)
    nodes = {
        "compressed": result["compressed"],
        "publish": result["publish"],
        "index_key": result["index_key"],
        "index_key_scale": result["index_key_scale"],
        "qli_scores": result["qli_scores"],
        "qli_indices": result["qli_indices"],
        "prepared_indices": result["prepared_indices"],
        "prepared_lengths": result["prepared_lengths"],
    }
    assert nodes["compressed"].dtype == torch.bfloat16
    assert nodes["index_key"].dtype == torch.int8
    assert nodes["index_key_scale"].dtype == torch.float16
    assert nodes["qli_indices"].dtype == torch.int32
    assert nodes["prepared_indices"].dtype == torch.int32
    if args.actual_out is not None:
        args.actual_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"capture_kind": "a2a3_layer14_source", "nodes": nodes}, args.actual_out)
        print(f"[L14] actual_out={args.actual_out}")
    if args.vllm_dump is not None:
        _compare(nodes, args.vllm_dump)
    print(
        f"[L14] source PASS: ratio={LAYER14.compress_ratio}, "
        f"published={int(nodes['publish'].sum())}, topk={nodes['qli_indices'].shape[-1]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
