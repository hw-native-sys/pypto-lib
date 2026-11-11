# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""DSpark sliding-window attention over history and all five query rows."""

from pathlib import Path
from pypto.runtime import pto_isa_include_dir
from utils import packed_cache_compare

import pypto.language as pl

from quant import pack_quantized_cache, unpack_quantized_cache


from config import (
    DSPARK_QUERY_WIDTH, DSPARK_SWA_INDEX_WIDTH, FLASH, MOE_TOKENS, ORI_BLOCKS_DYN, TP_SIZE,
    WINDOW_CACHE_BYTES,
)
from o_proj import o_proj
from qkv_proj_rope import quantize_projection_input, q_proj_rope, kv_proj, window_publish


D = FLASH.hidden_size
HEAD_DIM = FLASH.head_dim
LOCAL_H = FLASH.num_attention_heads // TP_SIZE
LOCAL_O_GROUPS = FLASH.o_groups // TP_SIZE
LOCAL_O_WIDTH = LOCAL_O_GROUPS * FLASH.o_lora_rank
O_GROUP_IN = FLASH.num_attention_heads * HEAD_DIM // FLASH.o_groups
O_LORA = FLASH.o_lora_rank
Q_LORA = FLASH.q_lora_rank
ROPE_DIM = FLASH.qk_rope_head_dim


QUERY_HEAD_TILE = max(32, LOCAL_H)
SOFTMAX_SCALE = HEAD_DIM**-0.5


@pl.jit.extern(core_type="aiv", source=Path(__file__).parent / "kernels" / "dspark_gather.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _gather_dspark_window_native(
    cache: pl.Tensor,
    indices: pl.Tensor,
    selected: pl.Out[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
) -> pl.Tensor:
    """Gather and dequantize DSpark windows with guarded query reuse."""
    ...


@pl.jit.inline(auto_scope=False)
def gather_dspark_window(
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8],
    indices: pl.Tensor[[MOE_TOKENS, DSPARK_SWA_INDEX_WIDTH], pl.INT32],
    selected: pl.Tensor[[MOE_TOKENS, DSPARK_SWA_INDEX_WIDTH * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    groups = (num_tokens + DSPARK_QUERY_WIDTH - 1) // DSPARK_QUERY_WIDTH
    workers = pl.max(1, pl.min(groups * (DSPARK_SWA_INDEX_WIDTH // 32), 64))
    with pl.spmd(workers, name_hint="dspark_cache_gather"):
        selected = _gather_dspark_window_native(cache, indices, selected, num_tokens)
    return selected


@pl.jit.extern(core_type="mixed", aic_source=Path(__file__).parent / "kernels" / "dspark_attention_core.cpp",
               aiv_source=Path(__file__).parent / "kernels" / "dspark_attention_core.cpp",
               include_dirs=(pto_isa_include_dir(),), dual_aiv_dispatch=True)
def _attend_dspark_native(
    query: pl.Tensor,
    selected: pl.Tensor,
    indices: pl.Tensor,
    sink: pl.Tensor,
    output: pl.Out[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
    heads: pl.Scalar[pl.INT32],
) -> pl.Tensor:
    """Attend selected K64 tiles with paired ownership and BF16 probabilities."""
    ...


@pl.jit.inline(auto_scope=False)
def attend_dspark_window(
    query: pl.Tensor[[MOE_TOKENS, LOCAL_H * HEAD_DIM], pl.BF16],
    selected: pl.Tensor[[MOE_TOKENS, DSPARK_SWA_INDEX_WIDTH * HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[MOE_TOKENS, DSPARK_SWA_INDEX_WIDTH], pl.INT32],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[MOE_TOKENS, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    workers = pl.max(1, pl.min(num_tokens, 32))
    if LOCAL_H < 32:
        padded_query = pl.create_tensor([MOE_TOKENS, QUERY_HEAD_TILE * HEAD_DIM], dtype=pl.BF16)
        for row in pl.spmd(num_tokens, name_hint="dspark_pad_heads"):
            padded = pl.full([1, QUERY_HEAD_TILE * HEAD_DIM], dtype=pl.BF16, value=0.0)
            active_query = query[row : row + 1, :]
            padded = pl.assemble(padded, active_query, [0, 0])
            padded_query[row : row + 1, :] = padded
        with pl.spmd(workers, name_hint="dspark_online_attention"):
            output = _attend_dspark_native(padded_query, selected, indices, sink, output, num_tokens, LOCAL_H)
    else:
        with pl.spmd(workers, name_hint="dspark_online_attention"):
            output = _attend_dspark_native(query, selected, indices, sink, output, num_tokens, LOCAL_H)
    return output


@pl.jit.inline(auto_scope=False)
def dspark_attention(
    x: pl.Tensor,
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    window_slots: pl.Tensor,
    window_indices: pl.Tensor,
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8],
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    qr = pl.create_tensor([MOE_TOKENS, Q_LORA], dtype=pl.BF16)
    q = pl.create_tensor([MOE_TOKENS, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    proj_payload, proj_scales, proj_payload_padded_rows = quantize_projection_input(x, num_tokens, row_tile=80)
    q_proj_result = q_proj_rope(
        proj_payload, proj_scales,
        wq_a, wq_a_scale, q_norm_weight,
        wq_b, wq_b_scale,
        rope_cos, rope_sin,
        qr, q,
        num_tokens, qa_worker_limit=32, qb_worker_limit=32,
        padded_rows=proj_payload_padded_rows,
        qa_row_tile=80, qb_row_tile=80,
    )
    kv_projection = pl.create_tensor([MOE_TOKENS, HEAD_DIM], dtype=pl.BF16)
    window_input_ready = pl.system.task_dummy(deps=[])
    kv_projection = kv_proj(
        proj_payload, proj_scales,
        wkv, wkv_scale,
        kv_projection, num_tokens, worker_limit=32, row_tile=80,
    )
    window_publish(
        kv_projection, kv_norm_weight, rope_cos, rope_sin,
        window_slots, window_cache,
        num_tokens, window_input_ready,
    )
    selected = pl.create_tensor([MOE_TOKENS, DSPARK_SWA_INDEX_WIDTH * HEAD_DIM], dtype=pl.BF16)
    selected = gather_dspark_window(window_cache, window_indices, selected, num_tokens)
    attended = pl.create_tensor([MOE_TOKENS, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    attended = attend_dspark_window(q, selected, window_indices, attn_sink, attended, num_tokens)
    o_proj_result = o_proj(
        attended, wo_a, wo_a_scale, wo_b, wo_b_scale,
        rope_cos, rope_sin, output, num_tokens, worker_limit=32, oa_row_tile=80, ob_row_tile=80,
    )
    return output, window_cache


@pl.jit
def dspark_attention_test(
    x: pl.Tensor[[MOE_TOKENS, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[MOE_TOKENS, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[MOE_TOKENS, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[MOE_TOKENS], pl.INT64],
    window_indices: pl.Tensor[[MOE_TOKENS, DSPARK_SWA_INDEX_WIDTH], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    output: pl.Out[pl.Tensor[[MOE_TOKENS, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    return dspark_attention(
        x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
        kv_norm_weight, attn_sink, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        window_slots, window_indices, window_cache, output, num_tokens,
    )


def make_attention_inputs(seed: int = 43):
    """Prepare five queries sharing history and all current KV rows."""
    import torch

    from sparse_attention import make_swa_inputs

    values = make_swa_inputs(batch=DSPARK_QUERY_WIDTH, seed=seed)
    positions = torch.arange(2, 2 + DSPARK_QUERY_WIDTH)
    angles = positions.float()[:, None] * (
        10000.0 ** (-torch.arange(0, ROPE_DIM, 2).float() / ROPE_DIM)
    )
    values["rope_cos"] = angles.cos()
    values["rope_sin"] = angles.sin()
    values["window_slots"] = 128 + positions
    indices = torch.full((DSPARK_QUERY_WIDTH, DSPARK_SWA_INDEX_WIDTH), -1, dtype=torch.int32)
    indices[:, : DSPARK_QUERY_WIDTH + 2] = torch.arange(
        128, 130 + DSPARK_QUERY_WIDTH, dtype=torch.int32
    )
    values["window_indices"] = indices
    for name in ("x", "rope_cos", "rope_sin", "window_slots", "window_indices"):
        current = values[name]
        fill = -1 if name in ("window_slots", "window_indices") else 0
        padded = torch.full((MOE_TOKENS, *current.shape[1:]), fill, dtype=current.dtype)
        padded[:DSPARK_QUERY_WIDTH] = current
        values[name] = padded
    return values


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="V4.1 DSpark non-causal SWA validation")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.execution_mode:
        parser.error("DSpark native attention requires A5 device execution; simulator support is unavailable")
    import torch

    from golden import ScalarSpec, TensorSpec, ratio_allclose, run
    from sparse_attention import SWA_INPUT_NAMES, swa_official_reference

    values = make_attention_inputs()

    def golden(tensors):
        payload, factors = unpack_quantized_cache(tensors["window_cache"], False)
        fields = dict(tensors, window_cache=payload, window_cache_scale=factors)
        row_inputs = ("x", "rope_cos", "rope_sin", "window_slots", "window_indices")
        reference_inputs = {
            name: fields[name][:DSPARK_QUERY_WIDTH] if name in row_inputs else fields[name]
            for name in SWA_INPUT_NAMES
        }
        output, cache, cache_scale = swa_official_reference(reference_inputs, key_tile=64)
        tensors["output"].zero_()
        tensors["output"][:DSPARK_QUERY_WIDTH].copy_(output)
        tensors["window_cache"].copy_(pack_quantized_cache(cache, cache_scale))

    def compare_payload(actual, expected, **_kwargs):
        return bool(torch.allclose(actual.float(), expected.float(), rtol=0.05, atol=0.05)), "FP8 payload"

    values["window_cache"] = pack_quantized_cache(values["window_cache"], values.pop("window_cache_scale"))
    specs = [TensorSpec(name, list(values[name].shape), values[name].dtype, init_value=values[name])
             for name in SWA_INPUT_NAMES if name != "window_cache_scale"]
    specs += [
        TensorSpec("output", [MOE_TOKENS, D], torch.bfloat16),
        ScalarSpec("num_tokens", torch.int32, DSPARK_QUERY_WIDTH, compile_runtime=True),
    ]
    result = run(
        fn=dspark_attention_test,
        specs=specs,
        golden_fn=golden,
        compare_fn={
            "window_cache": packed_cache_compare(
                compare_payload,
                ratio_allclose(max_error_ratio=0.0),
                fp4=False,
            ),
            "output": ratio_allclose(atol=0.03, rtol=0.03, max_error_ratio=0.001),
        },
        config={"backend_type": args.backend_type, "device_id": args.device,
                "execution_mode": args.execution_mode},
        compile_only=args.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(1)
