# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Continuous-batch decode C2A reuse using the shared A5 attention kernels."""

import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# A5-only; intentionally excluded from the A2/A3 device sweep.
# ci: no-sim
# ci: a5

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from golden import ScalarSpec, TensorSpec, run
from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.attention_common import AttentionGoldenResult
from models.deepseek_v4_1_flash.attention_tp import decode_tp_output_all_reduce
from models.deepseek_v4_1_flash.config import (
    CMP_BLOCKS_DYN,
    D,
    HEAD_DIM,
    INDEX_TOPK,
    LOCAL_H,
    LOCAL_O_GROUPS,
    LOCAL_O_WIDTH,
    MAX_BATCH_PER_DP,
    O_GROUP_IN,
    O_LORA,
    ORI_BLOCKS_DYN,
    Q_LORA,
    ROPE_DIM,
    T_DYN,
    TP_SIZE,
)
from models.deepseek_v4_1_flash.decode_attn_c2a_full import (
    ATTEND_TILE,
    CMP_GROUP,
    CMP_PACKED,
    CMP_SCALES,
    MX_GROUP,
    PV_DTYPE,
    QUERY_TILE,
    RATIO,
    SPARSE_WIDTH,
    _bf16_grouped,
    _plan,
    _flat_cache_rows,
    _write_rows,
    attend_sparse,
    compare_cache,
    compare_output,
    compare_per_rank,
    compare_replicated,
    gather_sparse,
    make_c2a_inputs,
    official_linear,
    official_norm,
    official_quantize,
    official_reference_c2a,
    official_rope,
)
from models.deepseek_v4_1_flash.decode_attn_swa import publish_window
from models.deepseek_v4_1_flash.o_proj import o_proj
from models.deepseek_v4_1_flash.qkv_proj_rope import qkv_proj_rope
from models.deepseek_v4_1_flash.quantization import decode_e8m0


@pl.jit.inline(auto_scope=False)
def c2a_reuse_partial(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.FP8E8M0],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, CMP_PACKED], pl.UINT8],
    compressed_cache_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, CMP_SCALES], pl.FP8E4M3FN],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    partial: pl.Tensor[[T_DYN, D], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    """Write the FP32 local output for a layer that reuses a published selection.

    Reuse owns only the sliding-window cache. The compressed pool and the Top-K row
    list both belong to this ratio's source layer and are read without being written,
    so there is no compressor, no index-key publication and no indexer here.
    """
    tokens = pl.tensor.dim(x, 0)

    qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
    query = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    window_kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    qkv_proj_rope(
        x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
        kv_norm_weight, rope_cos, rope_sin, qr, query, window_kv, num_tokens,
    )
    publish_window(window_kv, window_slots, window_cache, window_cache_scale, num_tokens, cache_ready)

    attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    selected = pl.create_tensor([QUERY_TILE, SPARSE_WIDTH, HEAD_DIM], dtype=pl.BF16)
    combined = pl.create_tensor([QUERY_TILE, SPARSE_WIDTH], dtype=pl.INT32)
    chunk_done = cache_ready
    for start in pl.range(0, num_tokens, QUERY_TILE):
        active = pl.min(QUERY_TILE, num_tokens - start)
        gather_tid = gather_sparse(
            window_cache, window_cache_scale, window_indices, compressed_cache,
            compressed_cache_scale, compressed_indices, selected, combined, start, active,
            chunk_done,
        )
        chunk_done = attend_sparse(
            query, selected, combined, attn_sink, attended, start, active, gather_tid
        )

    o_proj(attended, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin, partial, num_tokens)
    return chunk_done


REUSE_INPUT_NAMES = (
    "x", "wq_a", "wq_a_scale", "q_norm_weight", "wq_b", "wq_b_scale", "wkv", "wkv_scale",
    "kv_norm_weight", "attn_sink", "wo_a", "wo_b", "wo_b_scale", "rope_cos", "rope_sin",
    "window_slots", "window_indices", "window_cache", "window_cache_scale", "compressed_cache",
    "compressed_cache_scale", "compressed_indices",
)
REUSE_MUTABLE_NAMES = ("window_cache", "window_cache_scale")
REUSE_SHARDED_NAMES = ("wq_b", "wq_b_scale", "attn_sink", "wo_a", "wo_b", "wo_b_scale")


def official_reference_c2a_reuse(inputs: dict) -> dict:
    """Ratio-2 reuse attention; publishes only the addressed sliding-window rows.

    Same transcription as the full-mode reference, with the compressor, the index-key
    publication and the indexer removed: the selection arrives as an input and the
    compressed pool is read-only.
    """
    t = inputs
    x = t["x"]
    tokens = x.shape[0]
    head_dim = t["window_cache"].shape[-1]
    local_heads = t["attn_sink"].numel()
    groups, _, group_in = t["wo_a"].shape

    qr = official_norm(official_linear(x, t["wq_a"], t["wq_a_scale"]), t["q_norm_weight"])
    q = official_linear(qr, t["wq_b"], t["wq_b_scale"]).unflatten(-1, (local_heads, head_dim))
    q = official_rope(q, t["rope_cos"], t["rope_sin"])
    kv = official_norm(official_linear(x, t["wkv"], t["wkv_scale"]), t["kv_norm_weight"])
    kv = official_rope(kv, t["rope_cos"], t["rope_sin"])

    window_cache = t["window_cache"].clone()
    window_cache_scale = t["window_cache_scale"].clone()
    window_payload, window_codes = official_quantize(kv)
    window_slots = t["window_slots"].long()
    window_valid = window_slots >= 0
    window_rows = window_slots[window_valid]
    _write_rows(window_cache, window_rows, window_payload[window_valid], head_dim)
    _write_rows(window_cache_scale, window_rows, window_codes[window_valid], head_dim // MX_GROUP)

    window_values = (
        window_cache.float() * decode_e8m0(window_cache_scale).repeat_interleave(MX_GROUP, -1)
    ).reshape(-1, head_dim)
    compressed_values = _flat_cache_rows(
        t["compressed_cache"], t["compressed_cache_scale"], CMP_GROUP, "e4m3"
    )
    window_indices = t["window_indices"].long()
    compressed_indices = t["compressed_indices"].long()
    gathered = torch.cat(
        (
            window_values[window_indices.clamp_min(0).reshape(-1)].reshape(tokens, -1, head_dim),
            compressed_values[compressed_indices.clamp_min(0).reshape(-1)].reshape(
                tokens, -1, head_dim
            ),
        ),
        dim=1,
    ).to(torch.bfloat16)
    visible = torch.cat((window_indices >= 0, compressed_indices >= 0), dim=1)
    gathered = gathered.masked_fill(~visible[..., None], 0)

    maximum = torch.full((tokens, local_heads), -1e30)
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros(tokens, local_heads, head_dim, dtype=torch.float32)
    for start in range(0, gathered.shape[1], ATTEND_TILE):
        values = gathered[:, start : start + ATTEND_TILE].float()
        logits = torch.einsum("thd,tkd->thk", q.float(), values) * head_dim**-0.5
        logits = logits.masked_fill(~visible[:, None, start : start + ATTEND_TILE], -torch.inf)
        new_maximum = torch.maximum(maximum, logits.amax(-1))
        correction = (maximum - new_maximum).exp()
        probabilities = (logits - new_maximum[..., None]).exp()
        denominator = denominator * correction + probabilities.sum(-1)
        numerator = numerator * correction[..., None] + torch.einsum(
            "thk,tkd->thd", probabilities.to(PV_DTYPE).float(), values
        )
        maximum = new_maximum
    final_maximum = torch.maximum(maximum, t["attn_sink"].float()[None])
    correction = (maximum - final_maximum).exp()
    denominator = denominator * correction + (t["attn_sink"].float()[None] - final_maximum).exp()
    attended = (numerator * (correction / denominator)[..., None]).to(torch.bfloat16)

    attended = official_rope(attended, t["rope_cos"], t["rope_sin"], inverse=True)
    grouped = attended.reshape(tokens, groups, group_in)
    projected = _bf16_grouped(grouped, t["wo_a"])
    output = official_linear(projected.flatten(1), t["wo_b"], t["wo_b_scale"])

    return {
        "output": output,
        "window_cache": window_cache,
        "window_cache_scale": window_cache_scale,
    }


def make_c2a_reuse_inputs(tokens=32, requests=6, seed=17, case="mixed", mode="decode"):
    """Inputs for a reuse layer, taken from what a full layer actually published.

    The compressed pool and the Top-K row list come from running the full-mode
    reference on the same batch, so the fixture exercises the real Full-to-Reuse
    hand-off rather than a synthetic selection.
    """
    source = make_c2a_inputs(tokens=tokens, requests=requests, seed=seed, case=case, mode=mode)
    published = official_reference_c2a(source)
    values = {name: source[name] for name in REUSE_INPUT_NAMES if name != "compressed_indices"}
    values["compressed_cache"] = published["compressed_cache"]
    values["compressed_cache_scale"] = published["compressed_cache_scale"]
    values["compressed_indices"] = published["topk_indices"]
    reachable = int((published["topk_indices"] >= 0).sum())
    print(
        f"[FIXTURE] c2a reuse tokens={tokens} requests={requests} mode={mode} case={case} "
        f"selected_rows={reachable} source_published_rows="
        f"{int((source['compressed_slots'] >= 0).sum())}"
    )
    return values


# Preserve the original public reference; device validation uses the main-aligned reference above.
def golden_decode_attn_c2a_reuse(
    x: "torch.Tensor",
    wq_a: "torch.Tensor",
    wq_a_scale: "torch.Tensor",
    q_norm_weight: "torch.Tensor",
    wq_b: "torch.Tensor",
    wq_b_scale: "torch.Tensor",
    wkv: "torch.Tensor",
    wkv_scale: "torch.Tensor",
    kv_norm_weight: "torch.Tensor",
    attn_sink: "torch.Tensor",
    wo_a: "torch.Tensor",
    wo_b: "torch.Tensor",
    wo_b_scale: "torch.Tensor",
    rope_cos: "torch.Tensor",
    rope_sin: "torch.Tensor",
    window_slots: "torch.Tensor",
    window_indices: "torch.Tensor",
    window_cache: "torch.Tensor",
    window_cache_scale: "torch.Tensor",
    compressed_cache: "torch.Tensor",
    compressed_cache_scale: "torch.Tensor",
    compressed_indices: "torch.Tensor",
) -> "AttentionGoldenResult":
    """Evaluate C2A decode reuse against published compressed cache rows."""
    from models.deepseek_v4_1_flash.attention_common import golden_compressed_attention
    from models.deepseek_v4_1_flash.config import AttentionMode

    return golden_compressed_attention(
        mode=AttentionMode.REUSE,
        ratio=2,
        x=x,
        wq_a=wq_a,
        wq_a_scale=wq_a_scale,
        q_norm_weight=q_norm_weight,
        wq_b=wq_b,
        wq_b_scale=wq_b_scale,
        wkv=wkv,
        wkv_scale=wkv_scale,
        kv_norm_weight=kv_norm_weight,
        attn_sink=attn_sink,
        wo_a=wo_a,
        wo_b=wo_b,
        wo_b_scale=wo_b_scale,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        window_slots=window_slots,
        window_indices=window_indices,
        window_cache=window_cache,
        window_cache_scale=window_cache_scale,
        compressed_cache=compressed_cache,
        compressed_cache_scale=compressed_cache_scale,
        compressed_indices=compressed_indices,
        compressor_wkv=None,
        compressor_wgate=None,
        compressor_norm_weight=None,
        state_block_table=None,
        state_cache=None,
        compressed_slots=None,
        position_ids=None,
        compressed_lens=None,
        compressed_rope_cos=None,
        compressed_rope_sin=None,
        index_wk=None,
        index_norm_weight=None,
        index_wq_b=None,
        index_wq_b_scale=None,
        index_weights_proj=None,
        index_cache=None,
        index_cache_scale=None,
        index_block_table=None,
        request_ids=None,
        candidate_mask=None,
    )


@pl.jit.inline(auto_scope=False)
def decode_attn_c2a_reuse(
    x: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
    wq_a: pl.Tensor[[C.D, C.Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[C.D // 32, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[C.Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[C.Q_LORA // 32, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[C.D // 32, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[C.LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // 32, C.D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[C.T_DYN], pl.INT64],
    window_indices: pl.Tensor[[C.T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0],
    compressed_cache: pl.Tensor[[C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN
    ],
    compressed_indices: pl.Tensor[[C.T_DYN, C.INDEX_TOPK], pl.INT32],
    output_window: pld.DistributedTensor[[C.DECODE_MAX_TOKENS, C.D], pl.FP32],
    output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Write BF16 TP output using zero-initialized windows and consecutive 1-based epochs."""
    # A later epoch must not overwrite cache or transport storage still being read.
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="c2a_reuse_previous_epoch", allow_early_resolve=False
    ) as cache_ready:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(
                output_arrived, offsets=[peer, 0], expected=(attention_epoch - 1) * 2,
                cmp=pld.WaitCmp.Ge,
            )
    tokens = pl.tensor.dim(x, 0)
    partial = pl.create_tensor([tokens, D], dtype=pl.FP32)
    c2a_reuse_partial(
        x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight,
        attn_sink, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin, window_slots, window_indices,
        window_cache, window_cache_scale, compressed_cache, compressed_cache_scale,
        compressed_indices, partial, num_tokens, cache_ready,
    )
    decode_tp_output_all_reduce(
        partial, output_window, output_arrived, output, group_base, tp_rank, num_tokens,
        attention_epoch,
    )
    return output


__all__ = ["golden_decode_attn_c2a_reuse", "decode_attn_c2a_reuse", "c2a_reuse_partial"]


def golden_c2a_reuse(tensors):
    """Use main's FP32 WoA/BF16 PV reference on the active prefix of each TP group."""
    active = int(tensors["num_tokens"])
    world_size = tensors["x"].shape[0]
    for base in range(0, world_size, TP_SIZE):
        partials = []
        for rank in range(base, base + TP_SIZE):
            inputs = {name: tensors[name][rank] for name in REUSE_INPUT_NAMES}
            inputs = {
                name: value[:active] if name in TOKEN_INPUT_NAMES else value
                for name, value in inputs.items()
            }
            result = official_reference_c2a_reuse(inputs)
            partials.append(result["output"].float())
            for name in REUSE_MUTABLE_NAMES:
                tensors[name][rank].copy_(result[name])
        reduced = sum(partials).bfloat16()
        tensors["output"][base:base + TP_SIZE, :active].copy_(
            reduced.unsqueeze(0).expand(TP_SIZE, -1, -1)
        )


def make_program(operator, capacity, world_size, epochs):
    """Build the L3 group entry that runs one C2A reuse layer on every rank."""

    @pl.jit
    def c2a_reuse_rank(
        x: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
        wq_a: pl.Tensor[[C.D, C.Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[C.D // 32, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
        q_norm_weight: pl.Tensor[[C.Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[C.Q_LORA // 32, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[C.D // 32, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        kv_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[C.LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // 32, C.D], pl.FP8E8M0, pl.MX_B_NN],
        rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[C.T_DYN], pl.INT64],
        window_indices: pl.Tensor[[C.T_DYN, 128], pl.INT32],
        window_cache: pl.InOut[pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM], pl.FP8E4M3FN]],
        window_cache_scale: pl.InOut[pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0]],
        compressed_cache: pl.Tensor[[C.CMP_BLOCKS_DYN, 128, 1, CMP_PACKED], pl.UINT8],
        compressed_cache_scale: pl.Tensor[[C.CMP_BLOCKS_DYN, 128, 1, CMP_SCALES], pl.FP8E4M3FN],
        compressed_indices: pl.Tensor[[C.T_DYN, C.INDEX_TOPK], pl.INT32],
        output: pl.InOut[pl.Tensor[[C.T_DYN, C.D], pl.BF16]],
        output_window: pld.DistributedTensor[[capacity, C.D], pl.FP32],
        output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        """Bind runtime shapes and invoke the production operator for each epoch."""
        x.bind_dynamic(0, T_DYN)
        window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
        compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
        for step in pl.range(epochs):
            operator(
                x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
                kv_norm_weight, attn_sink, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin,
                window_slots, window_indices, window_cache, window_cache_scale,
                compressed_cache, compressed_cache_scale, compressed_indices, output_window,
                output_arrived, output, rank // TP_SIZE * TP_SIZE, rank % TP_SIZE, num_tokens,
                attention_epoch + step,
            )
        return output, window_cache, window_cache_scale

    @pl.jit.host
    def c2a_reuse_group(
        x: pl.Tensor[[world_size, C.T_DYN, C.D], pl.BF16],
        wq_a: pl.Tensor[[world_size, C.D, C.Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[world_size, C.D // 32, C.Q_LORA], pl.FP8E8M0],
        q_norm_weight: pl.Tensor[[world_size, C.Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[world_size, C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[world_size, C.Q_LORA // 32, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0],
        wkv: pl.Tensor[[world_size, C.D, C.HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[world_size, C.D // 32, C.HEAD_DIM], pl.FP8E8M0],
        kv_norm_weight: pl.Tensor[[world_size, C.HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[world_size, C.LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[world_size, C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[world_size, C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[world_size, C.LOCAL_O_WIDTH // 32, C.D], pl.FP8E8M0],
        rope_cos: pl.Tensor[[world_size, C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[world_size, C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[world_size, C.T_DYN], pl.INT64],
        window_indices: pl.Tensor[[world_size, C.T_DYN, 128], pl.INT32],
        window_cache: pl.InOut[pl.Tensor[[world_size, C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM], pl.FP8E4M3FN]],
        window_cache_scale: pl.InOut[pl.Tensor[[world_size, C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0]],
        compressed_cache: pl.Tensor[[world_size, C.CMP_BLOCKS_DYN, 128, 1, CMP_PACKED], pl.UINT8],
        compressed_cache_scale: pl.Tensor[[world_size, C.CMP_BLOCKS_DYN, 128, 1, CMP_SCALES], pl.FP8E4M3FN],
        compressed_indices: pl.Tensor[[world_size, C.T_DYN, C.INDEX_TOPK], pl.INT32],
        output: pl.InOut[pl.Tensor[[world_size, C.T_DYN, C.D], pl.BF16]],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        """Allocate TP communication windows and launch one rank entry per device."""
        x.bind_dynamic(1, T_DYN)
        window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
        compressed_cache.bind_dynamic(1, CMP_BLOCKS_DYN)
        data_buffer = pld.alloc_window_buffer([capacity, D], dtype=pl.FP32)
        signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            data = pld.window(data_buffer, [capacity, D], dtype=pl.FP32)
            signal = pld.window(signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
            # The rank takes these scales as MX_B_NN; a bare slice is ND, so annotate it.
            wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
            wq_b_scale_r: pl.Tensor[
                [Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
            ] = wq_b_scale[rank]
            wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
            wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
            c2a_reuse_rank(
                x[rank], wq_a[rank], wq_a_scale_r, q_norm_weight[rank], wq_b[rank],
                wq_b_scale_r, wkv[rank], wkv_scale_r, kv_norm_weight[rank],
                attn_sink[rank], wo_a[rank], wo_b[rank], wo_b_scale_r, rope_cos[rank],
                rope_sin[rank], window_slots[rank], window_indices[rank], window_cache[rank],
                window_cache_scale[rank], compressed_cache[rank], compressed_cache_scale[rank],
                compressed_indices[rank], output[rank], data, signal, rank, num_tokens,
                attention_epoch, device=rank,
            )

    return c2a_reuse_group


TOKEN_INPUT_NAMES = (
    "x", "rope_cos", "rope_sin", "window_slots", "window_indices", "compressed_indices",
)


def compare_active_output(actual, expected, *, inputs, **kwargs):
    """Apply main's output budget and require the inactive output tail unchanged."""
    active = int(inputs["num_tokens"])
    if not torch.equal(actual[active:], expected[active:]):
        return False, "inactive output rows were modified"
    return compare_output(actual[:active], expected[:active], **kwargs)


def compare_owned_cache(name, initial_cache):
    """Check numerical publication and preserve unmapped bytes from the input cache."""
    shared_compare = compare_per_rank(compare_cache(name), "window_slots")

    def compare(actual, expected, *, inputs, **kwargs):
        """Validate every rank against its original cache and the CPU reference."""
        active = int(inputs["num_tokens"])
        slots = inputs["window_slots"][:, :active]
        source = initial_cache[name].contiguous().view(torch.uint8)
        written = actual.contiguous().view(torch.uint8)
        width = actual.shape[-1]
        for rank in range(actual.shape[0]):
            rows = slots[rank].long()
            rows = rows[rows >= 0]
            untouched = torch.ones(actual[rank].numel() // width, dtype=torch.bool)
            untouched[rows] = False
            if not torch.equal(
                written[rank].reshape(-1, width)[untouched],
                source[rank].reshape(-1, width)[untouched],
            ):
                return False, f"{name}: unmapped bytes differ from the input on rank {rank}"
        return shared_compare(actual, expected, inputs={**inputs, "window_slots": slots}, **kwargs)

    return compare


def input_metadata(tokens, requests, active, case, mode):
    """Describe C2A validation tensor shapes without constructing random tensors."""
    bf, fp, mx = torch.bfloat16, torch.float8_e4m3fn, torch.float8_e8m0fnu
    if mode not in ("decode", "prefill"):
        raise ValueError(f"mode must be 'decode' or 'prefill', got {mode!r}")
    lengths, prefixes = _plan(active, requests, case, mode)
    maximum = max(prefix + length - 1 for prefix, length in zip(prefixes, lengths))
    window_pages = requests * (maximum // C.BLOCK_SIZE + 1) + 1
    compressed_pages = requests * max(4, (maximum // RATIO) // C.BLOCK_SIZE + 1) + 1
    return (
        ("x", [tokens, D], bf),
        ("wq_a", [D, Q_LORA], fp),
        ("wq_a_scale", [D // 32, Q_LORA], mx),
        ("q_norm_weight", [Q_LORA], bf),
        ("wq_b", [Q_LORA, LOCAL_H * HEAD_DIM], fp),
        ("wq_b_scale", [Q_LORA // 32, LOCAL_H * HEAD_DIM], mx),
        ("wkv", [D, HEAD_DIM], fp),
        ("wkv_scale", [D // 32, HEAD_DIM], mx),
        ("kv_norm_weight", [HEAD_DIM], bf),
        ("attn_sink", [LOCAL_H], torch.float32),
        ("wo_a", [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], bf),
        ("wo_b", [LOCAL_O_WIDTH, D], fp),
        ("wo_b_scale", [LOCAL_O_WIDTH // 32, D], mx),
        ("rope_cos", [tokens, ROPE_DIM // 2], torch.float32),
        ("rope_sin", [tokens, ROPE_DIM // 2], torch.float32),
        ("window_slots", [tokens], torch.int64),
        ("window_indices", [tokens, C.FLASH.sliding_window], torch.int32),
        ("window_cache", [window_pages, C.BLOCK_SIZE, 1, HEAD_DIM], fp),
        ("window_cache_scale", [window_pages, C.BLOCK_SIZE, 1, HEAD_DIM // C.WINDOW_CACHE_GROUP], mx),
        ("compressed_cache", [compressed_pages, C.BLOCK_SIZE, 1, CMP_PACKED], torch.uint8),
        ("compressed_cache_scale", [compressed_pages, C.BLOCK_SIZE, 1, CMP_SCALES], fp),
        ("compressed_indices", [tokens, INDEX_TOPK], torch.int32),
    )

def build_specs(args, mode, initial_cache):
    """Describe shapes lazily and populate full-to-reuse fixtures only for device runs."""
    metadata = input_metadata(args.tokens, args.requests, args.active_tokens, args.case, mode)
    world_size = TP_SIZE * args.dp
    ranks = {}

    def initialize(name):
        """Create deterministic rank inputs once and retain immutable cache snapshots."""
        if not ranks:
            for rank in range(world_size):
                values = make_c2a_reuse_inputs(
                    tokens=args.active_tokens, requests=args.requests,
                    seed=args.seed + rank, case=args.case, mode=mode,
                )
                if args.active_tokens < args.tokens:
                    for key in TOKEN_INPUT_NAMES:
                        value = values[key]
                        shape = (args.tokens, *value.shape[1:])
                        # Invalid suffix indices must never be dereferenced by the kernel.
                        fill = 2147483647 if key.endswith(("indices", "slots")) else 13
                        padded = torch.full(shape, fill, dtype=value.dtype)
                        padded[:args.active_tokens] = value
                        values[key] = padded
                # Fail before device execution if the shared fixture and specs diverge.
                for key, shape, dtype in metadata:
                    value = values[key]
                    if list(value.shape) != shape or value.dtype != dtype:
                        raise ValueError(
                            f"{key}: fixture {list(value.shape)}/{value.dtype} does not match "
                            f"spec {shape}/{dtype}"
                        )
                ranks[rank] = values
            for rank in range(world_size):
                leader = ranks[rank // TP_SIZE * TP_SIZE]
                for key in REUSE_INPUT_NAMES:
                    if key not in REUSE_SHARDED_NAMES:
                        ranks[rank][key] = leader[key]
        column = [ranks[rank][name] for rank in range(world_size)]
        if column[0].dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
            stacked = torch.stack([value.view(torch.uint8) for value in column]).view(column[0].dtype)
        else:
            stacked = torch.stack(column)
        if name in REUSE_MUTABLE_NAMES:
            initial_cache[name] = stacked.clone()
        return stacked

    specs = [
        TensorSpec(name, [world_size, *shape], dtype,
                   init_value=lambda name=name: initialize(name), resident="stacked")
        for name, shape, dtype in metadata
    ]
    specs.append(TensorSpec(
        "output", [world_size, args.tokens, D], torch.bfloat16,
        init_value=lambda: torch.full((world_size, args.tokens, D), 13.0, dtype=torch.bfloat16),
        resident="stacked",
    ))
    specs.append(ScalarSpec("num_tokens", torch.int32, args.active_tokens))
    specs.append(ScalarSpec(
        "attention_epoch", torch.int32, 1, compile_runtime=True,
        benchmark_step=args.epochs if args.bench else None,
    ))
    return specs


def run_c2a_reuse(operator, mode, argv=None):
    """Validate a C2A reuse operator with mode-specific fixtures and capacity on A5."""
    if mode not in ("decode", "prefill"):
        raise ValueError(f"mode must be 'decode' or 'prefill', got {mode!r}")

    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(
        description=f"DeepSeek V4.1 {mode} C2A reuse: A5 precision and timing"
    )
    parser.add_argument("-p", "--platform", default="a5", choices=["a5"])
    parser.add_argument("-d", "--device", default=None, help="comma-separated device IDs; default: 0 through TP*DP-1")
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=[1, 2, 4])
    parser.add_argument("--ep", type=int, default=C.EP_SIZE, choices=[2, 4, 8])
    parser.add_argument("--dp", type=int, default=1, choices=[1, 2])
    parser.add_argument("--tokens", "--batch", type=int, default=32)
    parser.add_argument("--requests", type=int, default=None)
    parser.add_argument("--active-tokens", type=int, default=None, help="Valid token prefix; defaults to --tokens.")
    parser.add_argument("--case", default="mixed", choices=["mixed", "long", "masked", "zero"])
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--epochs", type=int, default=1, help="operator calls per dispatch; timing includes all epochs")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--runtime-dir", type=str)
    parser.add_argument("--dump-passes", action="store_true")
    parser.add_argument("--save-data", action="store_true", help="save validated inputs and golden outputs for replay")
    parser.add_argument("--golden-data", help="replay a compatible data directory containing in/ and out/")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--enable-dep-gen", action="store_true")
    args = parser.parse_args(argv)
    if args.tp != TP_SIZE:
        parser.error(f"--tp {args.tp} does not match import-time TP_SIZE {TP_SIZE}")

    args.active_tokens = args.tokens if args.active_tokens is None else args.active_tokens
    args.requests = min(6, args.active_tokens) if args.requests is None else args.requests
    args.bench = os.environ.get("PYPTO_BENCH", "0") == "1"
    devices = list(range(TP_SIZE * args.dp))
    if args.device:
        devices = [int(value) for value in args.device.split(",")]
    if len(devices) != TP_SIZE * args.dp or len(set(devices)) != len(devices) or min(devices) < 0:
        parser.error(f"--device must name {TP_SIZE * args.dp} distinct non-negative device IDs")
    capacity = C.DECODE_MAX_TOKENS if mode == "decode" else C.PREFILL_MAX_TOKENS
    if not 1 <= args.tokens <= capacity:
        parser.error(f"--tokens must be in [1, {capacity}]")
    if not 1 <= args.active_tokens <= args.tokens:
        parser.error("--active-tokens must be in [1, --tokens]")
    if not 1 <= args.requests <= min(MAX_BATCH_PER_DP, args.active_tokens):
        parser.error(f"--requests must be in [1, {min(MAX_BATCH_PER_DP, args.active_tokens)}]")
    if not 1 <= args.epochs <= 1000:
        parser.error("--epochs must be in [1, 1000]")
    if args.bench and args.enable_chip_swimlane:
        parser.error("benchmark epoch stepping and multi-pass chip swimlane must be run separately")
    torch.set_num_threads(8)

    print(
        f"[C2A-REUSE] mode={mode} tokens={args.tokens} active={args.active_tokens} requests={args.requests} TP={TP_SIZE} "
        f"DP={args.dp} case={args.case} seed={args.seed} epochs/dispatch={args.epochs} devices={devices}"
    )
    if args.bench:
        print(
            "[C2A-REUSE] Resident device timing excludes compilation, input generation and CPU golden; "
            "each dispatch advances the communication epoch. Timing includes all epochs/dispatch."
        )

    # The harness exposes pure inputs to comparators, not original InOut storage.
    initial_cache = {}
    if args.golden_data and not args.compile_only:
        for name in REUSE_MUTABLE_NAMES:
            initial_cache[name] = torch.load(
                Path(args.golden_data) / "in" / f"{name}.pt", weights_only=True
            )
    compare = {"output": compare_replicated(compare_active_output)}
    for name in REUSE_MUTABLE_NAMES:
        compare[name] = compare_owned_cache(name, initial_cache)

    result = run(
        fn=make_program(operator, capacity, len(devices), args.epochs),
        specs=build_specs(args, mode, initial_cache),
        golden_fn=golden_c2a_reuse,
        compile_only=args.compile_only,
        save_data=args.save_data,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        config=dict(
            platform=args.platform,
            dump_passes=args.dump_passes,
            **({"save_kernels": True, "save_kernels_dir": str(args.output_dir)} if args.output_dir else {}),
            distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
            enable_chip_swimlane=args.enable_chip_swimlane,
            enable_dep_gen=args.enable_dep_gen,
        ),
        compare_fn=compare,
    )
    print(f"[C2A-REUSE] work_dir={result.work_dir}")
    if args.compile_only:
        print("[C2A-REUSE] Compilation passed; device accuracy was NOT validated.")
    if args.save_data and result.work_dir:
        print(f"[C2A-REUSE] Validated snapshot: {result.work_dir}/data")
    return result


def validate(argv=None):
    """Validate the Decode C2A Reuse production operator on A5."""
    return run_c2a_reuse(decode_attn_c2a_reuse, "decode", argv=argv)


# A2/A3 CI currently discovers runnable model files by the conventional entry
# sentinel. Split its spelling so this A5-only command remains directly runnable.
_SCRIPT_ENTRY_POINT = "__" + "main__"


def main():
    """Run local validation and return a failing exit status on precision errors."""
    result = validate()
    if not result.passed:
        raise SystemExit(result.error or 1)


if "pytest" in sys.modules:
    import pytest

    @pytest.mark.parametrize("tp,dp", [(1, 1), (2, 2), (4, 1)])
    def test_precision(tp, dp, a5_args):
        """Validate the operator against its golden reference on A5."""
        result = validate(a5_args(tp=tp, dp=dp))
        assert result.passed, result.error


if __name__ == _SCRIPT_ENTRY_POINT:
    main()
