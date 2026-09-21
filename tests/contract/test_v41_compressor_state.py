# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Request-owned compressor state regressions and optional two-call kernel validation."""

import argparse

import pytest
import torch

from models.deepseek_v4_1_flash.config import FLASH
from models.deepseek_v4_1_flash.golden import compressor_ratio2, compressor_ratio2_paged
from models.deepseek_v4_1_flash.metadata import build_forward_metadata


def _weights(width=8):
    """Create deterministic compressor projection and normalization weights."""
    generator = torch.Generator().manual_seed(12060)
    return (
        torch.randn(width, width, generator=generator),
        torch.randn(width, width, generator=generator),
        torch.ones(width, dtype=torch.bfloat16),
    )


def _run(chunks, prefixes, blocks, cache, weights):
    """Pack request chunks and evaluate their shared physical state cache."""
    lengths = torch.tensor([len(chunk) for chunk in chunks], dtype=torch.int32)
    starts = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).int()))
    positions = torch.cat([torch.arange(p, p + len(x)) for p, x in zip(prefixes, chunks)]).int()
    requests = torch.repeat_interleave(torch.arange(len(chunks)), lengths).int()
    return compressor_ratio2_paged(
        torch.cat(chunks), starts, positions, requests, torch.tensor(blocks, dtype=torch.int32)[:, None],
        cache, *weights,
    )


@pytest.mark.parametrize("capacity", [1, 2, 4, 7])
@pytest.mark.parametrize("splits", [[1, 10, 2], [2, 3, 8], [5, 1, 1, 6]])
def test_chunked_ring_matches_unchunked(capacity, splits):
    """Compare chunked compression with an unchunked sequence across ring capacities."""
    torch.manual_seed(12)
    x = torch.randn(sum(splits), 8).bfloat16()
    weights = _weights()
    expected, _, _ = compressor_ratio2(x, *weights)
    cache = torch.randn(5, capacity, 16)
    initial = cache.clone()
    actual = []
    offset = 0
    for length in splits:
        latent, publish = _run([x[offset:offset + length]], [offset], [3], cache, weights)
        actual.append(latent[publish])
        offset += length
    torch.testing.assert_close(torch.cat(actual), expected, rtol=0, atol=0)
    torch.testing.assert_close(cache[[0, 1, 2, 4]], initial[[0, 1, 2, 4]], rtol=0, atol=0)
    kv, score = x.float() @ weights[0], x.float() @ weights[1]
    for pos in range(max(0, len(x) - capacity), len(x)):
        torch.testing.assert_close(cache[3, pos % capacity], torch.cat((kv[pos], score[pos])))


def test_reorder_departure_admission_and_block_reuse():
    """Preserve request state through batch reordering and released-block reuse."""
    torch.manual_seed(1260)
    a, b, c = (torch.randn(12, 8).bfloat16() for _ in range(3))
    weights = _weights()
    cache = torch.randn(5, 4, 16)
    initial = cache.clone()
    _run([a[:3], b[:1]], [0, 0], [3, 1], cache, weights)
    # A departs with a pending pair; C reuses its block, while B moves to batch row zero.
    latent, publish = _run([b[1:10], c[:3]], [1, 0], [1, 3], cache, weights)
    b_full, _, _ = compressor_ratio2(b[:10], *weights)
    c_full, _, _ = compressor_ratio2(c[:3], *weights)
    torch.testing.assert_close(latent[:9][publish[:9]], b_full, rtol=0, atol=0)
    torch.testing.assert_close(latent[9:][publish[9:]], c_full, rtol=0, atol=0)
    latent, publish = _run([c[3:4], b[10:12]], [3, 10], [3, 1], cache, weights)
    c_full, _, _ = compressor_ratio2(c[:4], *weights)
    b_full, _, _ = compressor_ratio2(b, *weights)
    torch.testing.assert_close(latent[publish], torch.stack((c_full[-1], b_full[-1])), rtol=0, atol=0)
    torch.testing.assert_close(cache[[0, 2, 4]], initial[[0, 2, 4]], rtol=0, atol=0)


def test_padding_inactive_empty_and_restored_prefix():
    """Check inactive inputs and continuation from a restored predecessor."""
    weights = _weights()
    cache = torch.randn(3, 4, 16)
    initial = cache.clone()
    x = torch.randn(5, 8).bfloat16()
    latent, publish = compressor_ratio2_paged(
        x, torch.tensor([0, 0, 2, 3]), torch.tensor([1, 2, 1, -999, -999]),
        torch.tensor([1, 1, 2, 999, -1]), torch.tensor([[-1], [-1], [99]]), cache, *weights,
    )
    assert not publish.any()
    assert not latent.any()
    assert torch.equal(cache, initial)
    for starts, table in [(torch.tensor([0]), torch.empty(0, 1).int()), (torch.tensor([0, 0]), torch.tensor([[-1]]))]:
        latent, publish = compressor_ratio2_paged(
            x, starts, torch.full((5,), -1), torch.full((5,), -1), table, cache, *weights,
        )
        assert not publish.any()
        assert torch.equal(cache, initial)
    # An odd-position continuation is valid only with the matching predecessor restored.
    predecessor = torch.randn(1, 8).bfloat16()
    cache[2, 0, :8] = predecessor.float() @ weights[0]
    cache[2, 0, 8:] = predecessor.float() @ weights[1]
    actual, publish = _run([x[:1]], [1], [2], cache, weights)
    expected, _, _ = compressor_ratio2(torch.cat((predecessor, x[:1])), *weights)
    torch.testing.assert_close(actual[publish], expected, rtol=0, atol=0)


def test_metadata_preserves_engine_state_ownership():
    """Preserve physical block mappings and reject duplicate live ownership."""
    tables = {s: torch.tensor([[0], [1]], dtype=torch.int32) for s in FLASH.kv_source_layer_ids}
    state_tables = {s: torch.tensor([[4], [2]], dtype=torch.int32)
                    for s in FLASH.kv_source_layer_ids if FLASH.compress_ratios[s] == 2}
    metadata = build_forward_metadata(torch.tensor([0, 1, 3]), torch.tensor([1, 0]), tables[2], tables, state_tables)
    assert metadata.token_to_req_indices.tolist() == [0, 1, 1]
    assert set(metadata.state_block_tables) == {2, 8, 14}
    for source, table in state_tables.items():
        assert torch.equal(metadata.state_block_tables[source], table)
    state_tables[2] = torch.tensor([[4], [4]], dtype=torch.int32)
    with pytest.raises(ValueError, match="distinct"):
        build_forward_metadata(torch.tensor([0, 1, 3]), torch.tensor([1, 0]), tables[2], tables, state_tables)



@pytest.mark.parametrize("batch", [0, 2])
def test_metadata_empty_and_inactive_requests(batch):
    """Suppress cache publication for empty or inactive packed requests."""
    tables = {source: torch.zeros(batch, 1, dtype=torch.int32) for source in FLASH.kv_source_layer_ids}
    state_tables = {source: torch.full((batch, 1), -1, dtype=torch.int32)
                    for source in FLASH.kv_source_layer_ids if FLASH.compress_ratios[source] == 2}
    starts = torch.zeros(batch + 1, dtype=torch.int32)
    if batch:
        starts[-1] = 2
    metadata = build_forward_metadata(starts, torch.zeros(batch, dtype=torch.int32), tables[2], tables, state_tables)
    for source in state_tables:
        assert bool((metadata.compressed_slots[source] == -1).all())
        assert bool((metadata.index_slots[source] == -1).all())
    assert metadata.query_lens.tolist() == ([0, 2] if batch else [])

def _kernel_inputs(empty=False):
    """Build two calls with reordering, ring wraparound and invalid block indices."""
    from models.deepseek_v4_1_flash import config as C

    gen = torch.Generator().manual_seed(1260)
    tokens = 16
    # Call one: [A, B, inactive]. Call two: [B, new C, inactive], C reuses A's block.
    lengths = [[3, 1, 1], [9, 3, 1]]
    prefixes = [[0, 0, 0], [1, 0, 0]]
    starts = torch.zeros(2, 4, dtype=torch.int32)
    positions = torch.full((2, tokens), -999, dtype=torch.int32)
    requests = torch.full((2, tokens), 999, dtype=torch.int32)
    for step in range(2):
        starts[step, 1:] = torch.tensor(lengths[step]).cumsum(0).int()
        for request in range(3):
            start, end = starts[step, request:request + 2].tolist()
            positions[step, start:end] = torch.arange(prefixes[step][request], prefixes[step][request] + end - start)
            requests[step, start:end] = request
    if empty:
        starts.zero_()
        requests.fill_(-1)
    return {
        "kv": torch.randn(2, tokens, C.HEAD_DIM, generator=gen),
        "score": torch.randn(2, tokens, C.HEAD_DIM, generator=gen),
        "starts": starts,
        "positions": positions,
        "requests": requests,
        "table": torch.tensor([[[3], [1], [-1]], [[1], [3], [99]]], dtype=torch.int32),
        "cache": torch.randn(5, C.STATE_CAPACITY, C.STATE_WIDTH, generator=gen),
        "norm": torch.ones(C.HEAD_DIM, dtype=torch.bfloat16),
        "first_latent": torch.full((tokens, C.HEAD_DIM), -19, dtype=torch.bfloat16),
        "second_latent": torch.full((tokens, C.HEAD_DIM), -19, dtype=torch.bfloat16),
    }


def _kernel_golden(values):
    """Evaluate tokens sequentially, updating the ring before each following token."""
    values = dict(values)
    for key in ["kv", "score", "starts", "positions", "requests", "table"]:
        values[key] = torch.stack((values[f"first_{key}"], values[f"second_{key}"]))
    width = values["kv"].shape[-1]
    capacity = values["cache"].shape[1]
    for step in range(2):
        count = int(values["starts"][step, -1])
        latent = values["first_latent" if step == 0 else "second_latent"]
        latent[:count] = 0
        for token in range(count):
            request = int(values["requests"][step, token])
            block = int(values["table"][step, request, 0])
            if not 0 <= block < values["cache"].shape[0]:
                continue
            pos = int(values["positions"][step, token])
            if pos % 2:
                previous = values["cache"][block, (pos - 1) % capacity]
                pair_kv = torch.stack((previous[:width], values["kv"][step, token]))
                pair_score = torch.stack((previous[width:], values["score"][step, token]))
                pooled = (pair_kv * pair_score.softmax(dim=0)).sum(dim=0).bfloat16().float()
                normalized = pooled * torch.rsqrt(pooled.square().mean() + FLASH.rms_norm_eps)
                latent[token] = (normalized * values["norm"].float()).bfloat16()
            values["cache"][block, pos % capacity, :width] = values["kv"][step, token]
            values["cache"][block, pos % capacity, width:] = values["score"][step, token]


def run_kernel_cases(platform, device):
    """Validate two compressor calls and an empty workload on the selected platform."""
    import pypto.language as pl

    from golden import ScalarSpec, TensorSpec, run
    from models.deepseek_v4_1_flash import config as C
    from models.deepseek_v4_1_flash.compressor import compressor_ratio2 as kernel

    @pl.jit
    def two_calls(
        first_kv: pl.Tensor[[C.T_DYN, C.HEAD_DIM], pl.FP32],
        first_score: pl.Tensor[[C.T_DYN, C.HEAD_DIM], pl.FP32],
        first_starts: pl.Tensor[[C.Q_START_DYN], pl.INT32],
        first_positions: pl.Tensor[[C.T_DYN], pl.INT32],
        first_requests: pl.Tensor[[C.T_DYN], pl.INT32],
        first_table: pl.Tensor[[C.B_DYN, 1], pl.INT32],
        second_kv: pl.Tensor[[C.T_DYN, C.HEAD_DIM], pl.FP32],
        second_score: pl.Tensor[[C.T_DYN, C.HEAD_DIM], pl.FP32],
        second_starts: pl.Tensor[[C.Q_START_DYN], pl.INT32],
        second_positions: pl.Tensor[[C.T_DYN], pl.INT32],
        second_requests: pl.Tensor[[C.T_DYN], pl.INT32],
        second_table: pl.Tensor[[C.B_DYN, 1], pl.INT32],
        cache: pl.InOut[pl.Tensor[[C.STATE_BLOCKS_DYN, C.STATE_CAPACITY, C.STATE_WIDTH], pl.FP32]],
        norm: pl.Tensor[[C.HEAD_DIM], pl.BF16],
        first_latent: pl.InOut[pl.Tensor[[C.T_DYN, C.HEAD_DIM], pl.BF16]],
        second_latent: pl.InOut[pl.Tensor[[C.T_DYN, C.HEAD_DIM], pl.BF16]],
        first_count: pl.Scalar[pl.INT32],
        second_count: pl.Scalar[pl.INT32],
    ):
        """Execute consecutive compressor calls with an explicit state dependency."""
        first_kv.bind_dynamic(0, C.T_DYN)
        first_starts.bind_dynamic(0, C.Q_START_DYN)
        first_table.bind_dynamic(0, C.B_DYN)
        second_kv.bind_dynamic(0, C.T_DYN)
        second_starts.bind_dynamic(0, C.Q_START_DYN)
        second_table.bind_dynamic(0, C.B_DYN)
        cache.bind_dynamic(0, C.STATE_BLOCKS_DYN)
        ready = pl.system.task_dummy(deps=[])
        first_pool, first_state = kernel(
            first_kv, first_score, first_starts, first_positions, first_requests, first_table, cache, norm,
            first_latent, first_count, ready,
        )
        kernel(
            second_kv, second_score, second_starts, second_positions, second_requests, second_table, cache, norm,
            second_latent, second_count, first_state,
        )
        return cache, first_latent, second_latent

    for empty in [False, True]:
        staged = _kernel_inputs(empty)
        inputs = {f"{step}_{key}": staged[key][index].clone()
                  for index, step in enumerate(["first", "second"])
                  for key in ["kv", "score", "starts", "positions", "requests", "table"]}
        inputs.update({key: staged[key] for key in ["cache", "norm", "first_latent", "second_latent"]})
        specs = [TensorSpec(name, list(value.shape), value.dtype, init_value=value) for name, value in inputs.items()]
        specs += [ScalarSpec("first_count", torch.int32, int(inputs["first_starts"][-1]), compile_runtime=True),
                  ScalarSpec("second_count", torch.int32, int(inputs["second_starts"][-1]), compile_runtime=True)]
        result = run(fn=two_calls, specs=specs, golden_fn=_kernel_golden,
                     config={"platform": platform, "device_id": device}, rtol=0.01, atol=0.01,
                     compare_fn={"cache": lambda actual, expected, **_: (torch.equal(actual, expected), "exact ring state")})
        if not result.passed:
            raise RuntimeError(result.error or "compressor kernel validation failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate two compressor calls sharing a state cache")
    parser.add_argument("-p", "--platform", choices=["a5", "a5sim"], default="a5sim")
    parser.add_argument("-d", "--device", type=int, default=0)
    args = parser.parse_args()
    run_kernel_cases(args.platform, args.device)
