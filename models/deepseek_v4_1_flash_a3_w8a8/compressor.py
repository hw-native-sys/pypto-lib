# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Per-layer KV compressor: C1A single-row projection and C2A paired pooling.

C2A replaces every pair of consecutive token rows by one compressed row.  The
pair weight is a per-channel softmax over the two gate scores; the pooled row is
rounded to BF16 and RMS-normalized.  Positions whose index is even do not emit a
row, so the first token of a chunk needs the previous token's projections from
the carried state.
"""

import pypto.language as pl
import torch

from config import B_DYN, FLASH as M, STATE_BLOCKS_DYN, STATE_CAPACITY, T_DYN
from rmsnorm import rms_norm


# tiling and scheduling
POOL_WORKERS = 16

# model config
D = M.hidden_size
HEAD_DIM = M.head_dim
STATE_WIDTH = 2 * HEAD_DIM
EPS = M.rms_norm_eps

# tiling
T_TILE = 16
N_TILE = 128
K_TILE = 256


@pl.jit.inline(auto_scope=False)
def bf16_project(
    x: pl.Tensor,
    weight: pl.Tensor,
    output: pl.Tensor,
    width: pl.constexpr,
    output_width: pl.constexpr,
    output_fp32: pl.constexpr,
):
    """Project BF16 rows with a logical ``[out, in]`` weight and optional BF16 rounding."""
    t_dim = pl.tensor.dim(x, 0)
    # A5-provable token schedule: one SPMD dispatch over token-tile x n-tile
    # blocks (dynamic dim outermost, constant divisors) instead of a sequential
    # dynamic range, which cannot compose with an inline MoE.
    n_tiles = output_width // N_TILE
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    for block in pl.spmd(token_tiles * n_tiles, name_hint="compressor_project"):
        t_block = block // n_tiles
        n_block = block % n_tiles
        t0 = t_block * T_TILE
        n0 = n_block * N_TILE
        rows = pl.min(T_TILE, t_dim - t0)
        acc = pl.create_tensor([T_TILE, N_TILE], dtype=pl.FP32)
        for k0 in pl.pipeline(0, width, K_TILE, stage=2):
            a = pl.slice(x, [T_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
            a = pl.set_validshape(pl.fillpad(a, pad_value=pl.PadValue.zero), T_TILE, K_TILE)
            b = weight[n0 : n0 + N_TILE, k0 : k0 + K_TILE]
            acc = pl.matmul_acc(acc, a, b, b_trans=True, init_cond=(k0 == 0))
        if output_fp32:
            output[t0 : t0 + T_TILE, n0 : n0 + N_TILE] = pl.set_validshape(acc, rows, N_TILE)
        else:
            rounded = pl.cast(acc, target_type=pl.BF16, mode="rint")
            output[t0 : t0 + T_TILE, n0 : n0 + N_TILE] = pl.set_validshape(rounded, rows, N_TILE)
    return output


@pl.jit.inline(auto_scope=False)
def c2a_compressor(
    x: pl.Tensor,
    wkv: pl.Tensor,
    wgate: pl.Tensor,
    norm_w: pl.Tensor,
    position_ids: pl.Tensor,
    request_ids: pl.Tensor,
    state_block_table: pl.Tensor,
    state_cache: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    """Pool consecutive row pairs per request and carry the trailing projection.

    ``state_cache[block, position % 2, :]`` holds the ``(kv, score)``
    projections of the request's most recent token, so a chunk can start on an
    odd position.  The state is written after every pooling read completes.
    """
    t_dim = pl.tensor.dim(x, 0)
    kv_proj = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.FP32)
    score_proj = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.FP32)
    bf16_project(x, wkv, kv_proj, width=D, output_width=HEAD_DIM, output_fp32=True)
    bf16_project(x, wgate, score_proj, width=D, output_width=HEAD_DIM, output_fp32=True)
    latent = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.BF16)
    blocks = pl.tensor.dim(state_cache, 0)
    flat = pl.reshape(state_cache, [blocks * STATE_CAPACITY, STATE_WIDTH])
    tokens = pl.tensor.dim(x, 0)
    workers = pl.max(1, pl.min(num_tokens, POOL_WORKERS))
    step = workers
    with pl.spmd(workers, name_hint="c2a_pair_pool") as pool_tid:
        worker = pl.tile.get_block_idx()
        for token in pl.range(worker, num_tokens, step):
            position = pl.read(position_ids, [token])
            if position % 2 == 1:
                request = pl.cast(pl.read(request_ids, [token]), pl.INDEX)
                block = pl.cast(pl.read(state_block_table, [request, 0]), pl.INDEX)
                current_kv = kv_proj[token : token + 1, 0:HEAD_DIM]
                current_score = score_proj[token : token + 1, 0:HEAD_DIM]
                at_chunk_start = pl.cast(0, pl.INT32)
                if token == 0:
                    at_chunk_start = pl.cast(1, pl.INT32)
                else:
                    if pl.read(request_ids, [token - 1]) != pl.read(request_ids, [token]):
                        at_chunk_start = pl.cast(1, pl.INT32)
                if at_chunk_start == 1:
                    slot = block * STATE_CAPACITY + (position - 1) % STATE_CAPACITY
                    previous_kv = flat[slot : slot + 1, 0:HEAD_DIM]
                    previous_score = flat[slot : slot + 1, HEAD_DIM:STATE_WIDTH]
                else:
                    previous_kv = kv_proj[token - 1 : token, 0:HEAD_DIM]
                    previous_score = score_proj[token - 1 : token, 0:HEAD_DIM]
                highest = pl.maximum(previous_score, current_score)
                previous_weight = pl.exp(pl.sub(previous_score, highest))
                current_weight = pl.exp(pl.sub(current_score, highest))
                weighted = pl.add(
                    pl.mul(previous_kv, previous_weight),
                    pl.mul(current_kv, current_weight),
                )
                pooled = pl.div(weighted, pl.add(previous_weight, current_weight))
                latent[token : token + 1, 0:HEAD_DIM] = pl.cast(pooled, target_type=pl.BF16, mode="rint")
    # One writer: a per-worker round-robin store can drop an even-position
    # token's final state row when that worker's pool body is empty, so the
    # single-instance loop writes every row after the pool dependency.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c2a_state_write", deps=[pool_tid]) as state_tid:
        # Read anchors: order the state write after the projection writes even
        # when the token's pool path was skipped (even position), which would
        # otherwise let the composed scheduler store zeros for that slot.
        _kv_anchor = pl.read(kv_proj, [0, 0])
        _score_anchor = pl.read(score_proj, [0, 0])
        for token in pl.range(num_tokens):
            position = pl.read(position_ids, [token])
            last_of_request = pl.cast(0, pl.INT32)
            if token + 1 == num_tokens:
                last_of_request = pl.cast(1, pl.INT32)
            else:
                if pl.read(request_ids, [token + 1]) != pl.read(request_ids, [token]):
                    last_of_request = pl.cast(1, pl.INT32)
            if last_of_request == 1:
                if position >= 0:
                    request = pl.cast(pl.read(request_ids, [token]), pl.INDEX)
                    block = pl.cast(pl.read(state_block_table, [request, 0]), pl.INDEX)
                    slot = block * STATE_CAPACITY + position % STATE_CAPACITY
                    flat[slot : slot + 1, 0:HEAD_DIM] = kv_proj[token : token + 1, 0:HEAD_DIM]
                    flat[slot : slot + 1, HEAD_DIM:STATE_WIDTH] = score_proj[token : token + 1, 0:HEAD_DIM]
    output = rms_norm(latent, norm_w, output, width=HEAD_DIM, output_fp32=False, column_tile=256)
    return output


@pl.jit.inline(auto_scope=False)
def c1a_compressor(
    x: pl.Tensor,
    wkv: pl.Tensor,
    norm_w: pl.Tensor,
    output: pl.Tensor,
):
    """Project every token row and normalize it; one compressed row per token."""
    t_dim = pl.tensor.dim(x, 0)
    latent = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.BF16)
    bf16_project(x, wkv, latent, width=D, output_width=HEAD_DIM, output_fp32=False)
    rms_norm(latent, norm_w, output, width=HEAD_DIM, output_fp32=False, column_tile=256)
    return output



@pl.jit
def c1a_compressor_test(
    x: pl.Tensor[[T_DYN, 5120], pl.BF16],
    wkv: pl.Tensor[[512, 5120], pl.BF16],
    norm_w: pl.Tensor[[512], pl.BF16],
    output: pl.Out[pl.Tensor[[T_DYN, 512], pl.BF16]],
):
    x.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    c1a_compressor(x, wkv, norm_w, output)
    return output


@pl.jit
def c2a_compressor_test(
    x: pl.Tensor[[T_DYN, 5120], pl.BF16],
    wkv: pl.Tensor[[512, 5120], pl.BF16],
    wgate: pl.Tensor[[512, 5120], pl.BF16],
    norm_w: pl.Tensor[[512], pl.BF16],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.InOut[pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, 2 * 512], pl.FP32]],
    output: pl.Out[pl.Tensor[[T_DYN, 512], pl.BF16]],
):
    x.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    request_ids.bind_dynamic(0, T_DYN)
    state_block_table.bind_dynamic(0, B_DYN)
    state_cache.bind_dynamic(0, STATE_BLOCKS_DYN)
    output.bind_dynamic(0, T_DYN)
    c2a_compressor(
        x, wkv, wgate, norm_w, position_ids, request_ids, state_block_table, state_cache,
        output, pl.tensor.dim(x, 0),
    )
    return output, state_cache


def _golden_rms(value: torch.Tensor, norm_w: torch.Tensor) -> torch.Tensor:
    fp32 = value.float()
    inverse = torch.rsqrt(fp32.square().mean(dim=-1, keepdim=True) + EPS)
    return (fp32 * inverse * norm_w.float()).to(torch.bfloat16)


def golden_c1a_compressor(tensors):
    """Torch reference: BF16 projection followed by RMSNorm."""
    projected = torch.matmul(tensors["x"].float(), tensors["wkv"].float().t()).to(torch.bfloat16)
    tensors["output"][:] = _golden_rms(projected, tensors["norm_w"])


def golden_c2a_compressor(tensors):
    """Torch reference: per-request pair softmax pooling, BF16 rounding, RMSNorm."""
    x = tensors["x"].float()
    # Evaluate the BF16 dot in FP64, then restore the FP32 projection boundary.
    kv = torch.matmul(x.double(), tensors["wkv"].double().t()).float()
    score = torch.matmul(x.double(), tensors["wgate"].double().t()).float()
    positions = tensors["position_ids"]
    requests = tensors["request_ids"]
    state_block_table = tensors["state_block_table"]
    state_cache = tensors["state_cache"]
    tokens = x.shape[0]
    width = kv.shape[-1]
    latent = torch.zeros(tokens, width, dtype=torch.float32)
    for t in range(tokens):
        if int(positions[t]) % 2 != 1:
            continue
        request = int(requests[t])
        at_start = t == 0 or int(requests[t - 1]) != request
        if not at_start:
            previous_kv, previous_score = kv[t - 1], score[t - 1]
        else:
            block = int(state_block_table[request, 0])
            slot = (int(positions[t]) - 1) % 2
            previous_kv = state_cache[block, slot, 0:width]
            previous_score = state_cache[block, slot, width : 2 * width]
        pair_kv = torch.stack((previous_kv, kv[t]))
        pair_score = torch.stack((previous_score, score[t]))
        pooled = (pair_kv * pair_score.softmax(dim=0)).sum(dim=0)
        latent[t] = pooled.to(torch.bfloat16).float()
    tensors["output"][:] = _golden_rms(latent.to(torch.bfloat16), tensors["norm_w"])
    for t in range(tokens):
        last = t + 1 == tokens or int(requests[t + 1]) != int(requests[t])
        if last and int(positions[t]) >= 0:
            block = int(state_block_table[int(requests[t]), 0])
            slot = int(positions[t]) % 2
            state_cache[block, slot, 0:width] = kv[t]
            state_cache[block, slot, width : 2 * width] = score[t]


def validate_c2a_reference_precision():
    """Check correlated BF16 dots and carried state against analytic projections."""
    factors = torch.tensor([0.002, 0.003, 0.004, 0.005], dtype=torch.bfloat16)
    kv_weight = factors.repeat_interleave(HEAD_DIM // 4)
    score_weight = -kv_weight.flip(0)
    row_values = torch.tensor([-0.984375, 0.75], dtype=torch.bfloat16)
    state = torch.full((2, STATE_CAPACITY, STATE_WIDTH), 7.0, dtype=torch.float32)
    values = {
        "x": row_values[:, None].expand(2, D).clone(),
        "wkv": kv_weight[:, None].expand(HEAD_DIM, D).clone(),
        "wgate": score_weight[:, None].expand(HEAD_DIM, D).clone(),
        "norm_w": torch.ones(HEAD_DIM, dtype=torch.bfloat16),
        "position_ids": torch.tensor([0, 2], dtype=torch.int32),
        "request_ids": torch.tensor([0, 1], dtype=torch.int32),
        "state_block_table": torch.tensor([[1], [0]], dtype=torch.int32),
        "state_cache": state,
        "output": torch.zeros(2, HEAD_DIM, dtype=torch.bfloat16),
    }
    weights = torch.cat((kv_weight, score_weight)).double()
    projected = (D * row_values.double()[:, None] * weights[None, :]).float()
    expected_state = state.clone()
    expected_state[1, 0] = projected[0]
    expected_state[0, 0] = projected[1]
    golden_c2a_compressor(values)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)

    next_values = torch.tensor([0.875, -0.5], dtype=torch.bfloat16)
    values["x"] = next_values[:, None].expand(2, D).clone()
    values["position_ids"] += 1
    next_projected = (D * next_values.double()[:, None] * weights[None, :]).float()
    previous_kv = projected[:, :HEAD_DIM]
    previous_score = projected[:, HEAD_DIM:]
    current_kv = next_projected[:, :HEAD_DIM]
    current_score = next_projected[:, HEAD_DIM:]
    highest = torch.maximum(previous_score, current_score)
    previous_weight = torch.exp(previous_score - highest)
    current_weight = torch.exp(current_score - highest)
    numerator = previous_kv * previous_weight + current_kv * current_weight
    pooled = (numerator / (previous_weight + current_weight)).to(torch.bfloat16).float()
    inverse = torch.rsqrt(pooled.square().mean(dim=-1, keepdim=True) + EPS)
    expected_output = (pooled * inverse).to(torch.bfloat16)
    expected_state[1, 1] = next_projected[0]
    expected_state[0, 1] = next_projected[1]
    golden_c2a_compressor(values)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
    torch.testing.assert_close(values["output"], expected_output, rtol=0, atol=0)


def _position_ids(tokens: int, start: int) -> torch.Tensor:
    return torch.arange(start, start + tokens, dtype=torch.int32)


def build_c1a_specs(tokens: int = 16):
    from golden import TensorSpec

    return [
        TensorSpec("x", [tokens, D], torch.bfloat16, init_value=torch.randn),
        TensorSpec("wkv", [HEAD_DIM, D], torch.bfloat16, init_value=lambda: torch.randn(HEAD_DIM, D) * 0.02),
        TensorSpec("norm_w", [HEAD_DIM], torch.bfloat16, init_value=lambda: torch.randn(HEAD_DIM) * 0.05 + 1.0),
        TensorSpec("output", [tokens, HEAD_DIM], torch.bfloat16),
    ]


def build_c2a_specs(tokens: int = 8, seed: int = 0):
    from golden import TensorSpec
    from utils import build_request_plan

    generator = torch.Generator().manual_seed(seed)
    plan = build_request_plan([(1, tokens // 2), (tokens + 3, tokens - tokens // 2)], ratio=2)
    state_blocks = plan["state_block_table"].shape[0]
    initial = torch.randn(state_blocks, STATE_CAPACITY, 2 * HEAD_DIM, generator=generator)
    return [
        TensorSpec("x", [tokens, D], torch.bfloat16, init_value=torch.randn),
        TensorSpec("wkv", [HEAD_DIM, D], torch.bfloat16, init_value=lambda: torch.randn(HEAD_DIM, D) * 0.02),
        TensorSpec("wgate", [HEAD_DIM, D], torch.bfloat16, init_value=lambda: torch.randn(HEAD_DIM, D) * 0.02),
        TensorSpec("norm_w", [HEAD_DIM], torch.bfloat16, init_value=lambda: torch.randn(HEAD_DIM) * 0.05 + 1.0),
        TensorSpec("position_ids", [tokens], torch.int32, init_value=plan["position_ids"]),
        TensorSpec("request_ids", [tokens], torch.int32, init_value=plan["request_ids"]),
        TensorSpec("state_block_table", [state_blocks, 1], torch.int32, init_value=plan["state_block_table"]),
        TensorSpec("state_cache", [state_blocks, STATE_CAPACITY, 2 * HEAD_DIM], torch.float32, init_value=initial),
        TensorSpec("output", [tokens, HEAD_DIM], torch.bfloat16),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    from config import add_parallel_arguments
    add_parallel_arguments(parser)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    validate_c2a_reference_precision()
    torch.manual_seed(args.seed)
    output_compare = ratio_allclose(atol=2e-2, rtol=1e-2, max_error_ratio=5e-3)
    failures = []

    result = run(
        fn=c1a_compressor_test,
        specs=build_c1a_specs(args.tokens),
        golden_fn=golden_c1a_compressor,
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        rtol=2e-2,
        atol=2e-2,
        compare_fn={"output": output_compare},
        compile_only=args.compile_only,
    )
    if not result.passed:
        failures.append(f"c1a: {result.error or 'validation failed'}")

    # The second case starts on an odd position so the first pair comes from carried state.
    result = run(
        fn=c2a_compressor_test,
        specs=build_c2a_specs(args.tokens, seed=args.seed),
        golden_fn=golden_c2a_compressor,
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        rtol=2e-2,
        atol=2e-2,
        compare_fn={"output": output_compare, "state_cache": ratio_allclose(atol=1e-4, rtol=1e-4)},
        compile_only=args.compile_only,
    )
    if not result.passed:
        failures.append(f"c2a: {result.error or 'validation failed'}")

    if failures:
        for failure in failures:
            print(failure)
        raise SystemExit(1)
