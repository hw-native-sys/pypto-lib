# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 packed prefill SWA attention.

The public contract is single-request token-major prefill: the
layer owns the per-request loop and feeds this op one contiguous run of <=T
tokens. SWA consumes lowered metadata such as position_ids, slot mappings, and
window-ring sparse indices.
"""

import os

import pypto.language as pl

from config import (
    BLOCK_SIZE,
    ACTIVE as M,
    PREFILL_BATCH,
    PREFILL_CMP_BLOCK_NUM,
    PREFILL_CMP_MAX_BLOCKS,
    PREFILL_ORI_BLOCK_NUM,
    PREFILL_ORI_MAX_BLOCKS,
    PREFILL_SEQ,
)
from hc_post import golden_hc_post_prefill, hc_post_prefill
from hc_pre import golden_hc_pre, hc_pre
from qkv_proj_rope import (
    MX_N_TILE as QKV_MX_N_TILE,
    WQA_SCALE_ROWS,
    WQB_SCALE_ROWS,
    WKV_SCALE_ROWS,
    golden_qkv_proj_rope,
    materialize_rope_rows,
    qkv_proj_rope,
)
from rmsnorm import golden_rms_norm, rms_norm
from prefill_sparse_attn import golden_prefill_sparse_attn, prefill_sparse_attn
from prefill_mxfp8_o_proj import (
    MX_K_TILE as O_MX_K_TILE,
    MX_N_TILE as O_MX_N_TILE,
    WO_A_SCALE_ROWS,
    WO_B_SCALE_ROWS,
)


# Dynamic original-KV physical-pool dimension shared by the packed prefill ABI.
BLOCK_NUM_DYN = pl.dynamic("PREFILL_ORI_BLOCK_NUM_DYN")


# model config
B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
EPS = M.rms_norm_eps
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
ROPE_HEAD_DIM = ROPE_DIM
NOPE_DIM = M.nope_head_dim
NOPE_HEAD_DIM = NOPE_DIM
Q_LORA = M.q_lora_rank
ROPE_HALF = ROPE_DIM // 2
HALF_ROPE = ROPE_HALF
MAX_SEQ_LEN = M.max_position_embeddings
WIN = M.sliding_window
IDX_TOPK = M.index_topk
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

# SWA cache/topk contract. The ratio-0 path has only the sliding-window cache:
# single request, one window page, so the cache block count, the block_table
# length, and the per-request ori-window block count all collapse to 1.
BLOCK_NUM = PREFILL_ORI_BLOCK_NUM
CMP_BLOCK_NUM = PREFILL_CMP_BLOCK_NUM
START_POS = 0

# prefill_sparse_attn cache/topk contract (mirrors prefill_sparse_attn).
SPARSE_TOPK = WIN + IDX_TOPK
SPARSE_ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS
SPARSE_ORI_BLOCK_NUM = PREFILL_ORI_BLOCK_NUM
PREFILL_MAX_COMPRESSED = max(1, min(IDX_TOPK, WIN + WIN // 2))
SPARSE_CMP_MAX_BLOCKS = PREFILL_CMP_MAX_BLOCKS
WRITEBACK_GUARD_TILE = 16

# HC tiling, mirrored from hc_pre/hc_post but using prefill B/S/T.
MIX_PAD = 32
NEG_INF = -1e20
T_TILE = 16
RMS_T_TILE = 16
LINEAR_T_TILE = 16
COMB_T_TILE = 16
RMS_K_CHUNK = 128
LINEAR_K_CHUNK = 512
D_CHUNK = 512
RMS_K_BLOCKS = HC_DIM // RMS_K_CHUNK
LINEAR_K_BLOCKS = HC_DIM // LINEAR_K_CHUNK
D_BLOCKS = D // D_CHUNK
RMS_PIPE_STAGE = 1 if T >= 64 else 4

assert WIN == BLOCK_SIZE, "SWA prefill currently assumes one window page per batch"
assert S == WIN, "SWA overlay raw-index contract maps current suffix rows as WIN+t"
assert SPARSE_ORI_BLOCK_NUM == BLOCK_NUM
assert SPARSE_ORI_MAX_BLOCKS <= BLOCK_NUM


# PRO's wider hidden/HC dims make one prefill attention layer's per-task args and
# intermediates overflow the runtime's default ring-2 output heap, which surfaces as
# `orch_error_code=2 HEAP_RING_DEADLOCK`. prefill_fwd.py fixes the same thing with
# run()'s `ring_heap=` argument, but the golden harness's run_jit() does not plumb
# that kwarg through to execute_compiled(), so use the documented env-var fallback.
# Format: per-ring bytes, ring0..ring3, `0` = leave at default.
# All four rings, not just ring 2: raising ring 2 alone (what prefill_fwd.py needs)
# still deadlocks here at both 2 GiB and 4 GiB -- measured on device.
PREFILL_ATTN_RING_HEAP = (4 * 1024 * 1024 * 1024,) * 4
os.environ.setdefault("PTO2_RING_HEAP", ",".join(str(v) for v in PREFILL_ATTN_RING_HEAP))


@pl.jit.inline
def prefill_attention_swa(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[WQA_SCALE_ROWS, QKV_MX_N_TILE], pl.FP8E8M0],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[WQB_SCALE_ROWS, QKV_MX_N_TILE], pl.FP8E8M0],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[WKV_SCALE_ROWS, QKV_MX_N_TILE], pl.FP8E8M0],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[pl.Tensor[[BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    block_table: pl.Tensor[[BLOCK_NUM], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    position_ids: pl.Tensor[[T], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[WO_A_SCALE_ROWS, O_MX_N_TILE], pl.FP8E8M0],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[WO_B_SCALE_ROWS, O_MX_N_TILE], pl.FP8E8M0],
    x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    # Full prefill path mirrors the official block: hc_pre -> qkv/rope -> SWA
    # attention/o_proj -> KV writeback -> hc_post.
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post, comb)

    x_normed = pl.create_tensor([T, D], dtype=pl.BF16)
    rms_tid = rms_norm(x_mixed, attn_norm_w, x_normed)
    # Defers kv_proj_matmul one hop behind rms_norm so qr_proj_matmul dispatches first.
    late_dep = pl.system.task_dummy(deps=[rms_tid])

    rope_cos_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    materialize_rope_rows(
        freqs_cos,
        freqs_sin,
        position_ids,
        num_tokens,
        rope_cos_t,
        rope_sin_t,
    )

    # Reuse the shared prefill QKV/RoPE projection to stay aligned with decode.
    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    qkv_proj_rope(
        x_normed, wq_a, wq_a_scale, wq_b, wq_b_scale, wkv, wkv_scale,
        rope_cos_t, rope_sin_t, gamma_cq, gamma_ckv,
        q, kv, qr, qr_scale, late_dep,
    )

    block_num = pl.tensor.dim(kv_cache, 0)
    cache_rows = block_num * BLOCK_SIZE
    kv_cache_flat = pl.reshape(kv_cache, [cache_rows, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_cache_write"):
        for write_t in pl.range(T):
            if write_t < num_tokens:
                write_row_raw = pl.read(ori_slot_mapping, [write_t])
                if write_row_raw >= 0:
                    write_row = pl.cast(write_row_raw, pl.INDEX)
                    kv_cache_flat[write_row : write_row + 1, :] = kv[write_t : write_t + 1, :]

    swa_indices = pl.create_tensor([T, WIN], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_window_indices"):
        for idx_t in pl.range(T):
            idx_row = pl.full([1, WIN], dtype=pl.INT32, value=-1)
            if idx_t < num_tokens:
                abs_pos = pl.read(position_ids, [idx_t])
                window_valid = pl.min(pl.cast(WIN, pl.INT32), abs_pos + 1)
                key_start_abs = abs_pos + 1 - window_valid
                for win_col in pl.range(WIN):
                    win_col_i32 = pl.cast(win_col, pl.INT32)
                    if win_col_i32 < window_valid:
                        key_abs = key_start_abs + win_col_i32
                        blk_slot = key_abs // BLOCK_SIZE
                        blk = pl.read(block_table, [pl.cast(blk_slot, pl.INDEX)])
                        if blk >= 0:
                            row = pl.cast(blk * BLOCK_SIZE + (key_abs - blk_slot * BLOCK_SIZE), pl.INT32)
                            pl.write(idx_row, [0, win_col], row)
            swa_indices = pl.assemble(swa_indices, idx_row, [idx_t, 0])

    cmp_block_table_dummy = pl.create_tensor([SPARSE_CMP_MAX_BLOCKS], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_dummy_cmp_table"):
        for dummy_blk in pl.range(SPARSE_CMP_MAX_BLOCKS):
            pl.write(cmp_block_table_dummy, [dummy_blk], pl.cast(0, pl.INT32))
    cmp_kv_dummy = pl.create_tensor([CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], dtype=pl.BF16)
    cmp_indices_dummy = pl.create_tensor([T, IDX_TOPK], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_empty_cmp_meta"):
        for cmp_t in pl.range(T):
            cmp_indices_dummy[cmp_t:cmp_t + 1, 0:IDX_TOPK] = pl.full([1, IDX_TOPK], dtype=pl.INT32, value=-1)
    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    prefill_sparse_attn(
        q, kv_cache, swa_indices,
        cmp_kv_dummy, cmp_block_table_dummy,
        cmp_indices_dummy,
        attn_sink, num_tokens,
        rope_cos_t, rope_sin_t,
        wo_a, wo_a_scale, wo_b, wo_b_scale, attn_out,
    )

    hc_post_prefill(attn_out, x_hc, post, comb, x_out, num_tokens)
    return kv_cache, x_out


@pl.jit
def prefill_attention_swa_test(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[WQA_SCALE_ROWS, QKV_MX_N_TILE], pl.FP8E8M0],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[WQB_SCALE_ROWS, QKV_MX_N_TILE], pl.FP8E8M0],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[WKV_SCALE_ROWS, QKV_MX_N_TILE], pl.FP8E8M0],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[pl.Tensor[[BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    block_table: pl.Tensor[[BLOCK_NUM], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    position_ids: pl.Tensor[[T], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[WO_A_SCALE_ROWS, O_MX_N_TILE], pl.FP8E8M0],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[WO_B_SCALE_ROWS, O_MX_N_TILE], pl.FP8E8M0],
    x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    prefill_attention_swa(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_a_scale, wq_b, wq_b_scale, wkv, wkv_scale, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
        kv_cache, block_table, ori_slot_mapping,
        position_ids,
        attn_sink, wo_a, wo_a_scale, wo_b, wo_b_scale,
        x_out, num_tokens,
    )
    return kv_cache, x_out


def golden_prefill_attention_swa(tensors):
    """Torch reference for token-major packed SWA prefill."""
    import torch

    num_tokens = int(tensors["num_tokens"])
    x_hc_rect = tensors["x_hc"].view(B, S, HC_MULT, D)
    x_hc_flat = x_hc_rect.view(T, HC_MULT, D)
    x_mixed = torch.zeros(T, D, dtype=torch.bfloat16)
    post = torch.zeros(T, HC_MULT, dtype=torch.float32)
    comb = torch.zeros(T, HC_MULT * HC_MULT, dtype=torch.float32)
    golden_hc_pre({
        "x": x_hc_flat,
        "hc_fn": tensors["hc_attn_fn"],
        "hc_scale": tensors["hc_attn_scale"],
        "hc_base": tensors["hc_attn_base"],
        "x_mixed": x_mixed,
        "post": post,
        "comb": comb,
    })

    q = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(T, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(T, 1, dtype=torch.float32)
    rope_cos_t = torch.zeros(T, ROPE_DIM, dtype=torch.bfloat16)
    rope_sin_t = torch.zeros(T, ROPE_DIM, dtype=torch.bfloat16)
    x_normed = golden_rms_norm(x_mixed, tensors["attn_norm_w"])
    positions = tensors["position_ids"].to(torch.long)
    rope_cos_t = tensors["freqs_cos"].index_select(0, positions).contiguous()
    rope_sin_t = tensors["freqs_sin"].index_select(0, positions).contiguous()
    golden_qkv_proj_rope({
        "x": x_normed,
        "wq_a": tensors["wq_a"],
        "wq_a_scale": tensors["wq_a_scale"],
        "wq_b": tensors["wq_b"],
        "wq_b_scale": tensors["wq_b_scale"],
        "wkv": tensors["wkv"],
        "wkv_scale": tensors["wkv_scale"],
        "rope_cos": rope_cos_t,
        "rope_sin": rope_sin_t,
        "gamma_cq": tensors["gamma_cq"],
        "gamma_ckv": tensors["gamma_ckv"],
        "q": q,
        "kv": kv,
        "qr": qr,
        "qr_scale": qr_scale,
    })

    kv_cache_in = tensors["kv_cache"].clone()
    kv_cache_flat = kv_cache_in.view(kv_cache_in.shape[0] * BLOCK_SIZE, HEAD_DIM)
    for t in range(num_tokens):
        dst_row = int(tensors["ori_slot_mapping"][t].item())
        if dst_row >= 0:
            kv_cache_flat[dst_row, :] = kv[t]

    def cache_row_from_table(table, slot):
        block = slot // BLOCK_SIZE
        intra = slot % BLOCK_SIZE
        phys_block = int(table[block].item())
        if phys_block < 0:
            return -1
        return phys_block * BLOCK_SIZE + intra

    def build_swa_metadata():
        idx = torch.full((T, WIN), -1, dtype=torch.int32)
        pos = tensors["position_ids"]
        table = tensors["block_table"]
        for t in range(num_tokens):
            abs_pos = int(pos[t].item())
            window_valid = min(WIN, abs_pos + 1)
            key_start_abs = abs_pos + 1 - window_valid
            for k, key_abs in enumerate(range(key_start_abs, abs_pos + 1)):
                row = cache_row_from_table(table, key_abs)
                if row >= 0:
                    idx[t, k] = row
        return idx

    attn_out = torch.zeros(T, D, dtype=torch.bfloat16)
    golden_prefill_sparse_attn({
        "q": q,
        "ori_kv": kv_cache_in,
        "swa_indices": build_swa_metadata(),
        "cmp_kv": torch.zeros(CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16),
        "cmp_block_table": torch.zeros(SPARSE_CMP_MAX_BLOCKS, dtype=torch.int32),
        "cmp_indices": torch.full((T, IDX_TOPK), -1, dtype=torch.int32),
        "attn_sink": tensors["attn_sink"],
        "num_tokens": tensors["num_tokens"],
        "freqs_cos": rope_cos_t,
        "freqs_sin": rope_sin_t,
        "wo_a": tensors["wo_a"],
        "wo_a_scale": tensors["wo_a_scale"],
        "wo_b": tensors["wo_b"],
        "wo_b_scale": tensors["wo_b_scale"],
        "attn_out": attn_out,
    })

    tensors["kv_cache"][:] = kv_cache_in

    y = torch.zeros(T, HC_MULT, D, dtype=torch.float32)
    golden_hc_post_prefill({
        "x": attn_out.view(T, D),
        "residual": x_hc_flat,
        "post": post,
        "comb": comb,
        "y": y,
        "num_tokens": tensors["num_tokens"],
    })
    tensors["x_out"][:] = y


def _mapped_pool_ratio_allclose(
    mapping_name,
    *,
    num_tokens,
    atol,
    rtol,
    max_error_ratio,
):
    """Compare active mapped rows and require the rest of a physical pool to stay exact."""
    import torch

    from golden import ratio_allclose

    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    mapped_compare = ratio_allclose(
        atol=atol,
        rtol=rtol,
        max_error_ratio=max_error_ratio,
    )

    def compare(actual, expected, **kwargs):
        if actual.shape != expected.shape:
            return False, (
                f"    pool shape mismatch: actual={tuple(actual.shape)} "
                f"expected={tuple(expected.shape)}"
            )
        if actual.ndim < 2:
            return False, f"    mapped pool must have rank >= 2, got {tuple(actual.shape)}"

        inputs = kwargs.get("inputs", {})
        mapping = inputs.get(mapping_name)
        if mapping is None:
            return False, f"    compare_fn misconfigured: missing input '{mapping_name}'"
        if mapping.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            return False, (
                f"    '{mapping_name}' must have an integer dtype, got {mapping.dtype}"
            )
        mapping = mapping.cpu().to(torch.int64)
        if mapping.ndim != 1:
            return False, f"    '{mapping_name}' must be 1-D, got {tuple(mapping.shape)}"
        if num_tokens > mapping.numel():
            return False, (
                f"    num_tokens={num_tokens} exceeds '{mapping_name}' length "
                f"{mapping.numel()}"
            )

        actual_rows = actual.cpu().reshape(-1, actual.shape[-1])
        expected_rows = expected.cpu().reshape(-1, expected.shape[-1])
        row_count = actual_rows.shape[0]
        for label, rows in (("actual", actual_rows), ("expected", expected_rows)):
            if torch.is_floating_point(rows):
                nonfinite = ~torch.isfinite(rows)
                if nonfinite.any().item():
                    return False, (
                        f"    {label} pool contains "
                        f"{int(nonfinite.count_nonzero().item())} non-finite value(s)"
                    )

        invalid_negative = mapping < -1
        if invalid_negative.any().item():
            token = int(invalid_negative.nonzero(as_tuple=False)[0, 0].item())
            return False, (
                f"    '{mapping_name}'[{token}]={int(mapping[token].item())} is invalid; "
                "only -1 is a negative sentinel"
            )
        inactive_non_sentinel = mapping[num_tokens:] != -1
        if inactive_non_sentinel.any().item():
            tail_offset = int(
                inactive_non_sentinel.nonzero(as_tuple=False)[0, 0].item()
            )
            token = num_tokens + tail_offset
            return False, (
                f"    inactive '{mapping_name}'[{token}]={int(mapping[token].item())}; "
                "inactive entries must be -1"
            )

        active_mapping = mapping[:num_tokens]
        valid = active_mapping >= 0
        out_of_range = valid & (active_mapping >= row_count)
        if out_of_range.any().item():
            token = int(out_of_range.nonzero(as_tuple=False)[0, 0].item())
            return False, (
                f"    '{mapping_name}'[{token}]={int(active_mapping[token].item())} "
                f"is outside physical row range [0, {row_count})"
            )

        valid_rows = active_mapping[valid]
        if valid_rows.numel() > 1:
            unique_rows, counts = torch.unique(valid_rows, return_counts=True)
            duplicate = counts > 1
            if duplicate.any().item():
                row = int(unique_rows[duplicate][0].item())
                tokens = (active_mapping == row).nonzero(as_tuple=False).flatten().tolist()
                return False, (
                    f"    '{mapping_name}' maps multiple active tokens {tokens} "
                    f"to physical row {row}"
                )

        mapped_rows = torch.zeros(row_count, dtype=torch.bool)
        if valid_rows.numel() > 0:
            mapped_rows[valid_rows] = True
        equal_rows = (actual_rows == expected_rows).all(dim=-1)
        stray_rows = ~mapped_rows & ~equal_rows
        if stray_rows.any().item():
            row = int(stray_rows.nonzero(as_tuple=False)[0, 0].item())
            changed_values = int(
                (actual_rows[row] != expected_rows[row]).count_nonzero().item()
            )
            return False, (
                f"    unmapped physical row {row} changed "
                f"({changed_values} value(s)); mapping='{mapping_name}'"
            )
        if not mapped_rows.any().item():
            return True, ""

        ok, detail = mapped_compare(
            actual_rows[mapped_rows],
            expected_rows[mapped_rows],
            **kwargs,
        )
        if ok:
            return True, ""
        return False, f"    mapped rows from '{mapping_name}':\n{detail}"

    compare.__name__ = (
        f"mapped_pool_ratio_allclose(mapping={mapping_name}, num_tokens={num_tokens}, "
        f"atol={atol}, rtol={rtol}, max_error_ratio={max_error_ratio})"
    )
    return compare


def build_tensor_specs(
    start_pos: int = START_POS,
    num_tokens: int = T,
):
    import torch
    from golden import ScalarSpec, TensorSpec
    from rope_tables import build_deepseek_v4_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_deepseek_v4_rope_tables(M, 0, dtype=torch.bfloat16)

    # Single-request geometry: q_len = num_tokens (active prefix), context_len =
    # start_pos (absolute position base, a multiple of S=WIN under chunked prefill).
    context_len = start_pos
    q_len = num_tokens

    if num_tokens <= 0 or num_tokens > T:
        raise ValueError(f"num_tokens must be in [1, {T}], got {num_tokens}")
    max_position = context_len + q_len
    if context_len < 0:
        raise ValueError(f"context_len must be non-negative, got {context_len}")
    if max_position > MAX_SEQ_LEN:
        raise ValueError(f"position_ids exceed MAX_SEQ_LEN={MAX_SEQ_LEN}: got {max_position}")


    def token_pos():
        # Single-request absolute positions: pos[t] = context_len + local_idx
        # Padding rows keep their arange default; they are inactive.
        pos = torch.arange(T, dtype=torch.int32)
        for local_s in range(q_len):
            pos[local_s] = context_len + local_s
        return pos

    def init_x_hc():
        x = torch.empty(T, HC_MULT, D).uniform_(-1, 1)
        x[num_tokens:] = 0
        return x
    # Real layer-0 (SWA) hc_attn scale/base (fn synthetic at real magnitude). A synthetic
    # scale=0.5/base=0 leaves hc_pre post~=1 + near-uniform comb, cancelling attn_out and the
    # hc residual to near-zero in x_out where quant noise blows up the relative tail. Mirrors
    # decode_attention_swa.
    def init_hc_attn_fn():
        return torch.randn(MIX_HC, HC_DIM) * 0.039
    def init_hc_attn_scale():
        return torch.tensor([2.076026, 0.018729, 0.245936])
    def init_hc_attn_base():
        return torch.tensor([
            3.9083, -2.0399, -2.2033, -2.017,
            -2.4443, -10.3158, -8.9943, -6.3581,
            9.8577, -9.5177, -24.8724, -22.8929,
            -21.545, 0.7791, -3.386, 1.1948,
            -20.9605, -0.7702, 1.4218, -4.8994,
            1.5177, -29.7663, -30.1413, -1.2413,
        ])
    def init_attn_norm_w():
        return torch.ones(D)
    def init_gamma_cq():
        return torch.ones(Q_LORA)
    def init_gamma_ckv():
        return torch.ones(HEAD_DIM)
    def init_freqs_cos():
        return shared_freqs_cos.clone()
    def init_freqs_sin():
        return shared_freqs_sin.clone()
    def init_block_table():
        tbl = torch.full((BLOCK_NUM,), -1, dtype=torch.int32)
        for block in range(BLOCK_NUM):
            tbl[block] = block
        return tbl
    def cache_row_from_table(table, slot):
        block = slot // BLOCK_SIZE
        intra = slot % BLOCK_SIZE
        phys_block = int(table[block].item())
        if phys_block < 0:
            return -1
        return phys_block * BLOCK_SIZE + intra
    def init_kv_cache():
        cache = torch.zeros(BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
        cache_flat = cache.view(BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
        table = init_block_table()
        start = max(0, context_len - WIN)
        for abs_pos in range(start, context_len):
            row = cache_row_from_table(table, abs_pos)
            value = (torch.rand(HEAD_DIM,) - 0.5) * 0.1
            if row >= 0:
                cache_flat[row] = value.to(torch.bfloat16)
        return cache
    def init_ori_slot_mapping():
        mapping = torch.full((T,), -1, dtype=torch.int64)
        pos = token_pos()
        table = init_block_table()
        for t in range(num_tokens):
            mapping[t] = cache_row_from_table(table, int(pos[t].item()))
        return mapping
    def init_position_ids():
        return token_pos()
    def init_attn_sink():
        return torch.zeros(H)
    from expert_shared import _gen_mxfp8_weight_kn

    wq_a_fp8, wq_a_scale = _gen_mxfp8_weight_kn(
        (D, Q_LORA), dequant_std=0.058, chan_cv=0.25
    )
    wq_b_fp8, wq_b_scale = _gen_mxfp8_weight_kn(
        (Q_LORA, H * HEAD_DIM), dequant_std=0.058, chan_cv=0.25
    )
    wkv_fp8, wkv_scale = _gen_mxfp8_weight_kn(
        (D, HEAD_DIM), dequant_std=0.058, chan_cv=0.25
    )
    wo_a_values = []
    wo_a_scales = []
    for _ in range(O_GROUPS):
        value, scale = _gen_mxfp8_weight_kn(
            (O_GROUP_IN, O_LORA),
            dequant_std=0.25 / O_GROUP_IN ** 0.5,
            chan_cv=0.25,
            k_tile=O_MX_K_TILE,
            n_tile=O_MX_N_TILE,
        )
        wo_a_values.append(value)
        wo_a_scales.append(scale)
    wo_a_fp8 = torch.stack(wo_a_values)
    wo_a_scale = torch.cat(wo_a_scales, dim=0)
    wo_b_fp8, wo_b_scale = _gen_mxfp8_weight_kn(
        (O_GROUPS * O_LORA, D),
        dequant_std=0.25 / (O_GROUPS * O_LORA) ** 0.5,
        chan_cv=0.25,
        k_tile=O_MX_K_TILE,
        n_tile=O_MX_N_TILE,
    )

    return [
        TensorSpec("x_hc", [T, HC_MULT, D], torch.float32, init_value=init_x_hc),
        TensorSpec("hc_attn_fn", [MIX_HC, HC_DIM], torch.float32, init_value=init_hc_attn_fn),
        TensorSpec("hc_attn_scale", [3], torch.float32, init_value=init_hc_attn_scale),
        TensorSpec("hc_attn_base", [MIX_HC], torch.float32, init_value=init_hc_attn_base),
        TensorSpec("attn_norm_w", [D], torch.bfloat16, init_value=init_attn_norm_w),
        TensorSpec("wq_a", [D, Q_LORA], torch.float8_e4m3fn, init_value=lambda: wq_a_fp8),
        TensorSpec("wq_a_scale", [WQA_SCALE_ROWS, QKV_MX_N_TILE], torch.float8_e8m0fnu, init_value=lambda: wq_a_scale),
        TensorSpec("wq_b", [Q_LORA, H * HEAD_DIM], torch.float8_e4m3fn, init_value=lambda: wq_b_fp8),
        TensorSpec("wq_b_scale", [WQB_SCALE_ROWS, QKV_MX_N_TILE], torch.float8_e8m0fnu, init_value=lambda: wq_b_scale),
        TensorSpec("wkv", [D, HEAD_DIM], torch.float8_e4m3fn, init_value=lambda: wkv_fp8),
        TensorSpec("wkv_scale", [WKV_SCALE_ROWS, QKV_MX_N_TILE], torch.float8_e8m0fnu, init_value=lambda: wkv_scale),
        TensorSpec("gamma_cq", [Q_LORA], torch.bfloat16, init_value=init_gamma_cq),
        TensorSpec("gamma_ckv", [HEAD_DIM], torch.bfloat16, init_value=init_gamma_ckv),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("kv_cache", [BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16,
                   init_value=init_kv_cache, is_output=True),
        TensorSpec("block_table", [BLOCK_NUM], torch.int32, init_value=init_block_table),
        TensorSpec("ori_slot_mapping", [T], torch.int64, init_value=init_ori_slot_mapping),
        TensorSpec("position_ids", [T], torch.int32, init_value=init_position_ids),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("wo_a", [O_GROUPS, O_GROUP_IN, O_LORA], torch.float8_e4m3fn, init_value=lambda: wo_a_fp8),
        TensorSpec("wo_a_scale", [WO_A_SCALE_ROWS, O_MX_N_TILE], torch.float8_e8m0fnu, init_value=lambda: wo_a_scale),
        TensorSpec("wo_b", [O_GROUPS * O_LORA, D], torch.float8_e4m3fn, init_value=lambda: wo_b_fp8),
        TensorSpec("wo_b_scale", [WO_B_SCALE_ROWS, O_MX_N_TILE], torch.float8_e8m0fnu, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [T, HC_MULT, D], torch.float32, is_output=True),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_reldiff, run_jit

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 packed prefill SWA correctness test.")
    parser.add_argument("-p", "--platform", type=str, default="a5",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--start-pos", type=int, default=START_POS,
                        help="context_len (multiple of S=WIN); fixture-only, lowered into token metadata.")
    parser.add_argument("--num-tokens", type=int, default=T,
                        help="Active token count (q_len), capped by T; passed to the kernel as num_tokens.")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--enable-dep-gen", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    compare_tokens = args.num_tokens

    result = run_jit(
        fn=prefill_attention_swa_test,
        specs=build_tensor_specs(
            args.start_pos,
            args.num_tokens,
        ),
        golden_fn=golden_prefill_attention_swa,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
            enable_dep_gen=args.enable_dep_gen,
        ),
        compile_only=args.compile_only,
        rtol=1e-2,
        atol=1e-2,
        compare_fn={
            "x_out": ratio_reldiff(diff_thd=3e-3, pct_thd=0.005, max_diff_hd=1,
                                   valid_rows=compare_tokens, zero_tail=True),
            "kv_cache": _mapped_pool_ratio_allclose(
                "ori_slot_mapping",
                num_tokens=compare_tokens,
                atol=1e-4,
                rtol=1e-2,
                max_error_ratio=0.005,
            ),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
