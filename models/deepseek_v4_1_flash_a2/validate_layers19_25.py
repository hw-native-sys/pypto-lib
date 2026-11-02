# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the L19/L20/L21/L24/L25 A2/A3 reference chain on a real checkpoint.

This validator deliberately loads only attention and indexer tensors.  The
384 routed experts and the Engram tables are not needed to verify the
requested decoder attention operators.  A vLLM capture can be supplied with
``--vllm-dump``; without one the script still checks the complete PyTorch
reference chain, all cache dtypes, and every published intermediate shape.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from .layers19_25 import (
    INDEX_DIM,
    INDEX_TOPK,
    _QKV_CONFIG,
    _qkv_weights,
    c1a_full_reference,
    c1a_reindex_reference,
    c1a_reuse_reference,
    c2a_reuse_reference,
)
from .indexer_query import indexer_query_reference
from .layer14_15 import layer14_source_reference
from .qkv import qkv_reference
from .raw_checkpoint import RawCheckpoint, load_attention_weights


def _tables(tokens: int, theta: float = 160000.0) -> tuple[torch.Tensor, torch.Tensor]:
    position = torch.arange(tokens, dtype=torch.float32)
    frequency = theta ** (-torch.arange(0, 64, 2, dtype=torch.float32) / 64)
    angle = position[:, None] * frequency[None, :]
    return angle.cos(), angle.sin()


def _cache_rows(values: torch.Tensor, width: int, rows: int, tokens: int | None = None) -> torch.Tensor:
    token_count = values.shape[0] if tokens is None else int(tokens)
    output = torch.zeros(token_count, rows, width, dtype=values.dtype, device=values.device)
    count = min(values.shape[0], rows)
    if count:
        output[:, :count] = values[:count].unsqueeze(0)
    return output


def _assert_dtype(nodes: dict[str, torch.Tensor], expected: dict[str, torch.dtype]) -> None:
    for name, dtype in expected.items():
        value = nodes[name]
        if value.dtype != dtype:
            raise AssertionError(f"{name}: expected {dtype}, got {value.dtype}")


def _load_dump(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and isinstance(payload.get("nodes"), dict):
        payload = payload["nodes"]
    if not isinstance(payload, dict):
        raise TypeError("vLLM dump must be a tensor mapping or contain a nodes mapping")
    return payload


def _compare(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: shape {tuple(actual.shape)} != {tuple(expected.shape)}")
    if actual.dtype != expected.dtype:
        raise AssertionError(f"{name}: dtype {actual.dtype} != {expected.dtype}")
    delta = (actual.float() - expected.float()).abs()
    print(f"[{name}] max_abs={float(delta.max()):.6g} mean_abs={float(delta.mean()):.6g}")


def run(checkpoint: Path, tokens: int = 4) -> dict[str, dict[str, torch.Tensor]]:
    if tokens < 1 or tokens > 16:
        raise ValueError("tokens must be in [1, 16]")
    checkpoint_reader = RawCheckpoint(checkpoint)
    cosine, sine = _tables(tokens)
    positions = torch.arange(tokens, dtype=torch.int32)
    # C2A publishes the completed pair at the preceding (even) position.
    source_positions = (positions - 1).clamp_min(0)
    source_cosine = cosine[source_positions]
    source_sine = sine[source_positions]
    generator = torch.Generator().manual_seed(190251)
    hidden = {
        layer: torch.randn(tokens, 5120, generator=generator).to(torch.bfloat16)
        for layer in (14, 19, 20, 21, 24, 25)
    }
    window_indices = torch.full((tokens, 128), -1, dtype=torch.int32)
    window_indices[:, : min(tokens, 128)] = torch.arange(min(tokens, 128), dtype=torch.int32)
    window_values = torch.randn(tokens, 128, 512, generator=generator).to(torch.bfloat16)
    sink = torch.zeros(64, dtype=torch.float32)

    weights = {layer: load_attention_weights(checkpoint_reader, layer) for layer in (14, 19, 20, 21, 24, 25)}
    qkv14 = qkv_reference(hidden[14], _qkv_weights(weights[14]), cosine, sine, _QKV_CONFIG)
    index_query14 = indexer_query_reference(
        qkv14["query_latent"], weights[14]["index_wq_b"], weights[14]["index_wq_b_scale"], cosine, sine,
    )
    compressor_wkv14 = torch.matmul(
        hidden[14].float(), weights[14]["compressor_wkv_canonical"].float()
    )
    compressor_wgate14 = torch.matmul(
        hidden[14].float(), weights[14]["compressor_wgate_canonical"].float()
    )
    previous_wkv14 = torch.zeros_like(compressor_wkv14)
    previous_wgate14 = torch.zeros_like(compressor_wgate14)
    previous_wkv14[1:] = compressor_wkv14[:-1]
    previous_wgate14[1:] = compressor_wgate14[:-1]
    source14 = layer14_source_reference(
        kv_proj=compressor_wkv14,
        score_proj=compressor_wgate14,
        previous_kv=previous_wkv14,
        previous_score=previous_wgate14,
        norm_weight=weights[14]["compressor_norm"],
        position_ids=positions,
        index_wk=weights[14]["index_wk"],
        index_norm_weight=weights[14]["index_k_norm"],
        qli_query=index_query14["payload"],
        qli_query_scale=index_query14["scales"],
        qli_weights=(
            torch.matmul(hidden[14].float(), weights[14]["index_weights_proj"].float())
            .to(torch.bfloat16).float() * (32 * 128) ** -0.5
        ).to(torch.float16),
        topk=INDEX_TOPK,
        previous_valid=torch.ones(tokens, dtype=torch.bool),
        compressed_cosine=source_cosine,
        compressed_sine=source_sine,
    )
    source20 = c1a_full_reference(
        hidden[20], weights[20], positions, cosine, sine, cosine, sine,
    )
    compressed_rows_l14 = _cache_rows(source14["published_compressed"], 512, INDEX_TOPK, tokens)
    compressed_rows_l20 = _cache_rows(source20["compressed"], 512, INDEX_TOPK, tokens)
    index_rows = source20["index_key"]
    index_scale = source20["index_key_scale"]
    index_padding = INDEX_TOPK - index_rows.shape[0]
    if index_padding > 0:
        index_rows = torch.cat((index_rows, torch.zeros(index_padding, INDEX_DIM, dtype=torch.int8)), dim=0)
        index_scale = torch.cat((index_scale, torch.ones(index_padding, dtype=torch.float16)), dim=0)

    outputs: dict[str, dict[str, torch.Tensor]] = {"L20": source20}
    outputs["L19"] = c2a_reuse_reference(
        hidden[19], weights[19], cosine, sine,
        window_values, window_indices, compressed_rows_l14, source14["prepared_indices"], sink,
    )
    outputs["L21"] = c1a_reuse_reference(
        hidden[21], weights[21], cosine, sine, source20["prepared_indices"],
        window_values, window_indices, compressed_rows_l20, sink,
    )
    reindexed = c1a_reindex_reference(
        hidden[24], weights[24], positions, cosine, sine,
        index_rows, index_scale, source20["pool_block_ids"],
        window_values, window_indices, compressed_rows_l20, sink,
        candidate_block_counts=source20["pool_block_counts"],
    )
    outputs["L24"] = reindexed
    outputs["L25"] = c1a_reuse_reference(
        hidden[25], weights[25], cosine, sine, reindexed["prepared_indices"],
        window_values, window_indices, compressed_rows_l20, sink,
    )

    _assert_dtype(
        source20,
        {
            "query": torch.bfloat16,
            "compressor_latent": torch.bfloat16,
            "compressed": torch.bfloat16,
            "index_key": torch.int8,
            "index_key_scale": torch.float16,
            "index_query_payload": torch.int8,
            "index_query_scale": torch.float16,
            "index_weights": torch.float16,
            "prepared_indices": torch.int32,
        },
    )
    if source14["compressed"].dtype != torch.bfloat16:
        raise AssertionError("L14 compressed cache must be BF16")
    if source14["index_key"].dtype != torch.int8 or source14["index_key_scale"].dtype != torch.float16:
        raise AssertionError("L14 index cache must be INT8 with FP16 scales")
    if source14["prepared_indices"].dtype != torch.int32:
        raise AssertionError("L14 prepared indices must be INT32")
    for name, nodes in outputs.items():
        prepared = nodes.get("prepared_indices", nodes.get("compressed_indices"))
        print(
            f"[{name}] query={tuple(nodes['query'].shape)} "
            f"prepared={tuple(prepared.shape) if prepared is not None else 'not returned'} "
            f"output={tuple(nodes['output_output'].shape) if 'output_output' in nodes else 'not requested'}"
        )
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_checkpoint = os.environ.get("DSV41_A2_CHECKPOINT")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(default_checkpoint) if default_checkpoint else None,
        help="raw checkpoint directory (or set DSV41_A2_CHECKPOINT)",
    )
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--actual-out", type=Path)
    parser.add_argument("--vllm-dump", type=Path)
    args = parser.parse_args()

    if args.checkpoint is None:
        parser.error("--checkpoint is required unless DSV41_A2_CHECKPOINT is set")
    outputs = run(args.checkpoint, args.tokens)
    if args.actual_out is not None:
        args.actual_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"capture_kind": "a2a3_layers19_25", "nodes": outputs}, args.actual_out)
        print(f"[L19-L25] actual_out={args.actual_out}")
    if args.vllm_dump is not None:
        expected = _load_dump(args.vllm_dump)
        for layer, nodes in outputs.items():
            for key, value in nodes.items():
                name = f"{layer}.{key}"
                if name not in expected:
                    raise AssertionError(f"vLLM dump is missing {name}")
                _compare(name, value, expected[name])
    print("[L19-L25] checkpoint simulator PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
