# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=8  # CI: 8-card TP run; the deployment world size, borrowed via task-submit --device-num
"""Full CSA decode layer with the output projection sharded one group per card.

Everything ahead of the projection is replicated exactly as the single-card layer
does it -- same attention, same indexer, same compressor. Only `wo_a` / `wo_b` are
sharded, on the group axis, and only the projection's `[T, D]` FP32 partial crosses
the wire. `O_GROUPS == TP == 8`, so there is no input collective: group `r` is heads
[8r, 8r+8), whose `o_packed` rows every card already produced.

Compare against the single-card layer, which is the same build with the projection
left replicated:

    python decode_csa.py    -p a2a3 -d 0                      # replicated
    python decode_csa_tp.py -p a2a3 -d 0,1,2,3,4,5,6,7        # TP by group
"""

import sys

# The layer's L2 warm covers whichever o-projection weights the build reads, and it
# is one SDMA scope, so decode_csa freezes their flat extents at import.
if "--oproj-tp" not in sys.argv:
    sys.argv.append("--oproj-tp")

import pypto.language as pl
import pypto.language.distributed as pld

import decode_csa as C
from decode_csa import attention_csa_tp
import o_proj_tp as OTP
from config import DECODE_BATCH, DECODE_SEQ, FLASH as M


# The generated signatures below are copied verbatim from `decode_csa`, so they
# reference that module's shape constants; pull them all in rather than restating
# forty of them and risking a drift the compiler would not catch.
globals().update({k: v for k, v in vars(C).items() if k.isupper()})

B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
HC_MULT = M.hc_mult
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = M.num_attention_heads * M.head_dim // O_GROUPS
N_RANKS = OTP.N_RANKS
REDUCE_WINDOW_ROWS = OTP.REDUCE_WINDOW_ROWS
SCALE_WINDOW_ROWS = OTP.SCALE_WINDOW_ROWS
DONE_VALUE = OTP.DONE_VALUE

assert O_GROUPS == N_RANKS, f"one group per card: O_GROUPS={O_GROUPS}, TP={N_RANKS}"

# The attention parameters are copied verbatim from `decode_csa.attention_csa_packed`
# so the two entries cannot drift; only the o-projection tensors differ.
@pl.jit
def attention_csa_tp_test(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[B, MAIN_STATE_MAX_BLOCKS], pl.INT32],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32],
    inner_compress_state_block_table: pl.Tensor[[B, INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8],
    idx_kv_scale: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32],
    idx_block_table: pl.Tensor[[B, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    window_swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    position_ids: pl.Tensor[[T], pl.INT32],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a_shard: pl.Tensor[[1, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_shard: pl.Tensor[[D, O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    reduce_window: pld.DistributedTensor[[REDUCE_WINDOW_ROWS, D], pl.INT32],
    scale_window: pld.DistributedTensor[[SCALE_WINDOW_ROWS, 1], pl.FP32],
    reduce_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    sync_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
):
    return attention_csa_tp(
        x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv, freqs_cos, freqs_sin, cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w, compress_state, compress_state_block_table, idx_wq_b, idx_wq_b_scale, weights_proj, hadamard_idx, inner_wkv, inner_wgate, inner_ape, inner_norm_w, inner_compress_state, inner_compress_state_block_table, kv_cache, cmp_kv, cmp_block_table, idx_kv_cache, idx_kv_scale, idx_block_table, ori_slot_mapping, window_swa_indices, window_swa_lens, cmp_slot_mapping, idx_slot_mapping, state_slot_mapping, inner_state_slot_mapping, position_ids, kv_seq_lens, attn_sink,
        wo_a_shard, wo_b_shard, wo_b_scale, x_out,
        reduce_window, scale_window, reduce_signal, sync_signal, my_rank, done_epoch,
    )


@pl.jit.host
def l3_attention_csa_tp(
    x_hc: pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    wq_a: pl.Tensor[[N_RANKS, D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_RANKS, Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[N_RANKS, H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_RANKS, D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[N_RANKS, Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[N_RANKS, MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[N_RANKS, MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[N_RANKS, COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    compress_state: pl.Tensor[[N_RANKS, MAIN_STATE_BLOCK_NUM_DYN, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[N_RANKS, B, MAIN_STATE_MAX_BLOCKS], pl.INT32],
    idx_wq_b: pl.Tensor[[N_RANKS, Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[N_RANKS, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[N_RANKS, D, IDX_N_HEADS], pl.BF16],
    hadamard_idx: pl.Tensor[[N_RANKS, IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[N_RANKS, INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[N_RANKS, INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[N_RANKS, COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[N_RANKS, IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.Tensor[[N_RANKS, INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32],
    inner_compress_state_block_table: pl.Tensor[[N_RANKS, B, INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[N_RANKS, ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_kv: pl.Tensor[[N_RANKS, CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[N_RANKS, B, CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.Tensor[[N_RANKS, IDX_CACHE_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8],
    idx_kv_scale: pl.Tensor[[N_RANKS, IDX_CACHE_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32],
    idx_block_table: pl.Tensor[[N_RANKS, B, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    window_swa_indices: pl.Tensor[[N_RANKS, T, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[N_RANKS, T], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    idx_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    state_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    position_ids: pl.Tensor[[N_RANKS, T], pl.INT32],
    kv_seq_lens: pl.Tensor[[N_RANKS, B], pl.INT32],
    attn_sink: pl.Tensor[[N_RANKS, H], pl.FP32],
    wo_a_shard: pl.Tensor[[N_RANKS, 1, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_shard: pl.Tensor[[N_RANKS, D, O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32]],
):
    """One orchestration per card, sharing the reduce and signal windows."""
    reduce_window_buf = pld.alloc_window_buffer([REDUCE_WINDOW_ROWS, D], dtype=pl.INT32)
    scale_window_buf = pld.alloc_window_buffer([SCALE_WINDOW_ROWS, 1], dtype=pl.FP32)
    reduce_signal_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    sync_signal_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)

    for r in pl.range(pld.world_size()):
        reduce_window = pld.window(reduce_window_buf, [REDUCE_WINDOW_ROWS, D], dtype=pl.INT32)
        scale_window = pld.window(scale_window_buf, [SCALE_WINDOW_ROWS, 1], dtype=pl.FP32)
        reduce_signal = pld.window(reduce_signal_buf, [N_RANKS, 1], dtype=pl.INT32)
        sync_signal = pld.window(sync_signal_buf, [N_RANKS, 1], dtype=pl.INT32)
        attention_csa_tp_test(
            x_hc[r], hc_attn_fn[r], hc_attn_scale[r], hc_attn_base[r], attn_norm_w[r], wq_a[r], wq_b[r], wq_b_scale[r], wkv[r], gamma_cq[r], gamma_ckv[r], freqs_cos[r], freqs_sin[r], cmp_wkv[r], cmp_wgate[r], cmp_ape[r], cmp_norm_w[r], compress_state[r], compress_state_block_table[r], idx_wq_b[r], idx_wq_b_scale[r], weights_proj[r], hadamard_idx[r], inner_wkv[r], inner_wgate[r], inner_ape[r], inner_norm_w[r], inner_compress_state[r], inner_compress_state_block_table[r], kv_cache[r], cmp_kv[r], cmp_block_table[r], idx_kv_cache[r], idx_kv_scale[r], idx_block_table[r], ori_slot_mapping[r], window_swa_indices[r], window_swa_lens[r], cmp_slot_mapping[r], idx_slot_mapping[r], state_slot_mapping[r], inner_state_slot_mapping[r], position_ids[r], kv_seq_lens[r], attn_sink[r],
            wo_a_shard[r], wo_b_shard[r], wo_b_scale[r], x_out[r],
            reduce_window, scale_window, reduce_signal, sync_signal, r, DONE_VALUE, device=r,
        )


def build_tensor_specs(start_pos=None):
    """The single-card CSA fixture, replicated per rank, with the o-proj weights sharded.

    `run_jit` binds specs positionally, so the two shards must sit exactly where
    `wo_a` / `wo_b` sat in the replicated list.
    """
    import torch
    from golden import TensorSpec

    base = C.build_tensor_specs(start_pos)
    wo_a = next(s for s in base if s.name == "wo_a").init_value()
    wo_b = next(s for s in base if s.name == "wo_b").init_value()

    def ranked(spec):
        init = spec.init_value
        stacked = None if init is None else (lambda f=init: torch.stack([f()] * N_RANKS, dim=0))
        if spec.is_output:
            # An in-out (kv_cache) keeps its seed as well as its output role.
            return TensorSpec(spec.name, [N_RANKS] + list(spec.shape), spec.dtype,
                              init_value=stacked, is_output=True)
        return TensorSpec(
            spec.name, [N_RANKS] + list(spec.shape), spec.dtype,
            init_value=stacked, resident=spec.resident)

    out = []
    for spec in base:
        if spec.name == "wo_a":
            # Card r carries group r's proj_a rows only.
            out.append(TensorSpec(
                "wo_a_shard", [N_RANKS, 1, O_LORA, O_GROUP_IN], torch.bfloat16,
                init_value=lambda: torch.stack(
                    [wo_a[r:r + 1] for r in range(N_RANKS)], dim=0),
                resident="stacked"))
        elif spec.name == "wo_b":
            # ... and proj_b's matching O_LORA column band.
            out.append(TensorSpec(
                "wo_b_shard", [N_RANKS, D, O_LORA], torch.int8,
                init_value=lambda: torch.stack(
                    [wo_b[:, r * O_LORA:(r + 1) * O_LORA] for r in range(N_RANKS)], dim=0),
                resident="stacked"))
        else:
            out.append(ranked(spec))
    return out


def golden_attention_csa_tp(tensors):
    """Per-rank golden: rebuild the full weights from the shards, then reuse the layer's."""
    import torch

    wo_a = torch.cat([tensors["wo_a_shard"][r] for r in range(N_RANKS)], dim=0)
    wo_b = torch.cat([tensors["wo_b_shard"][r] for r in range(N_RANKS)], dim=1)
    for r in range(N_RANKS):
        per_rank = {k: (v[r] if k not in ("wo_a_shard", "wo_b_shard") else v)
                    for k, v in tensors.items()}
        per_rank["wo_a"] = wo_a
        per_rank["wo_b"] = wo_b
        C.golden_attention_csa(per_rank)
        for k, v in per_rank.items():
            if k in tensors and k not in ("wo_a", "wo_b", "wo_a_shard", "wo_b_shard"):
                tensors[k][r] = v


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, ratio_reldiff, run_jit
    from pypto.ir.distributed_compiled_program import DistributedConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=str,
                        default=",".join(str(i) for i in range(N_RANKS)),
                        help=f"comma-separated device ids; need at least {N_RANKS}")
    parser.add_argument("--start-pos", type=int, default=None)
    parser.add_argument("--oproj-tp", action="store_true", default=True,
                        help=argparse.SUPPRESS)
    parser.add_argument("--pre-sync", action="store_true", default=False,
                        help="Emit a leading barrier so host dispatch skew is absorbed "
                             "before the measured layer instead of inside the reduce.")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0,
                        choices=(0, 1, 2, 4))
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) >= N_RANKS, f"need at least {N_RANKS} devices, got {device_ids}"

    result = run_jit(
        fn=l3_attention_csa_tp,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_attention_csa_tp,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:N_RANKS], num_sub_workers=0),
        ),
        runtime_cfg=dict(
            platform=args.platform,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        atol=1e-2,
        rtol=1e-2,
        compare_fn={
            "x_out": ratio_reldiff(diff_thd=3e-3, pct_thd=0.008, max_diff_hd=1),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
