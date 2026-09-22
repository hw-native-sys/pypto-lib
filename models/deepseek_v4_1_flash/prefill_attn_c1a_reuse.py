# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed-prefill C1A reuse attention leaf."""

import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# A5-only; intentionally excluded from the A2/A3 device sweep. `ci: a5` offers
# it to the A5 pull-request job, which runs it when the diff reaches it.
# ci: no-sim
# ci: a5

_SCRIPT_ENTRY_POINT = "__" + "main__"
if __name__ == _SCRIPT_ENTRY_POINT:
    if not any(arg == "--tp" or arg.startswith("--tp=") for arg in sys.argv):
        sys.argv.extend(["--tp", "2"])

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.prefill_c1a_common import prefill_c1a_partial
from models.deepseek_v4_1_flash.qkv_proj_rope import q_proj_qr
from models.deepseek_v4_1_flash.prefill_c1a_test_utils import (
    CASE_DEFAULT,
    CASE_MAX_TOKENS,
    CASE_NAMES,
    CASE_TOKENS,
    CACHE_MAX_RELATIVE_L2,
    COMMON_INPUT_NAMES,
    apply_distributed_golden,
    attention_output_compare,
    golden_prefill_c1a_attention,
    make_tensor_specs,
)
from models.deepseek_v4_1_flash.attention_common import (
    AttentionGoldenResult,
    golden_compressed_attention,
    quantized_cache_compare,
)
from models.deepseek_v4_1_flash.attention_tp import prefill_tp_output_all_reduce
from models.deepseek_v4_1_flash.config import AttentionMode


D = C.D
HEAD_DIM = C.HEAD_DIM
LOCAL_H = C.LOCAL_H
LOCAL_O_WIDTH = C.LOCAL_O_WIDTH
Q_LORA = C.Q_LORA
TP_SIZE = C.TP_SIZE
PREFILL_MAX_TOKENS = C.PREFILL_MAX_TOKENS

REUSE_INPUT_NAMES = COMMON_INPUT_NAMES + ("compressed_indices",)


if TP_SIZE not in (1, 2, 4):
    raise ValueError("Prefill C1A currently supports TP1, TP2, and TP4; TP8 requires head-tile padding")

def golden_prefill_attn_c1a_reuse(
    x: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    attn_sink: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    window_slots: torch.Tensor,
    window_indices: torch.Tensor,
    window_cache: torch.Tensor,
    window_cache_scale: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_cache_scale: torch.Tensor,
    compressed_indices: torch.Tensor,
) -> AttentionGoldenResult:
    """Return one rank's FP32 partial output and reference cache state."""
    return golden_compressed_attention(
        mode=AttentionMode.REUSE,
        ratio=1,
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
        attention_fn=golden_prefill_c1a_attention,
        output_dtype=torch.float32,
    )


def make_prefill_attn_c1a_reuse(reducer=prefill_tp_output_all_reduce, output_tokens=C.T_DYN):
    @pl.jit.inline(auto_scope=False)
    def prefill_attn_c1a_reuse(
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
        output_window: pld.DistributedTensor[[C.PREFILL_MAX_TOKENS, C.D], pl.FP32],
        output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
        output: pl.Tensor[[output_tokens, C.D], pl.BF16],
        group_base: pl.Scalar[pl.INT32],
        tp_rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        """Read published ratio-1 Top-K rows and compute packed-prefill C1A."""
        tokens = pl.tensor.dim(x, 0)
        partial = pl.create_tensor([tokens, D], dtype=pl.FP32)
        if num_tokens > 0:
            query_latent = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
            q_proj_qr(x, wq_a, wq_a_scale, q_norm_weight, query_latent, num_tokens)
            prefill_c1a_partial(
                x,
                query_latent,
                wq_b,
                wq_b_scale,
                wkv,
                wkv_scale,
                kv_norm_weight,
                attn_sink,
                wo_a,
                wo_b,
                wo_b_scale,
                rope_cos,
                rope_sin,
                window_slots,
                window_indices,
                window_cache,
                window_cache_scale,
                compressed_cache,
                compressed_cache_scale,
                compressed_indices,
                partial,
                num_tokens,
            )
        reducer(
            partial,
            output_window,
            output_arrived,
            output,
            group_base,
            tp_rank,
            num_tokens,
            attention_epoch,
        )
        return output

    return prefill_attn_c1a_reuse


prefill_attn_c1a_reuse = make_prefill_attn_c1a_reuse()


@pl.jit
def prefill_attn_c1a_reuse_test(
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
    window_cache_scale: pl.InOut[
        pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0]
    ],
    compressed_cache: pl.Tensor[[C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.COMPRESSED_CACHE_GROUP],
        pl.FP8E4M3FN,
    ],
    compressed_indices: pl.Tensor[[C.T_DYN, C.INDEX_TOPK], pl.INT32],
    output_window: pld.DistributedTensor[[C.PREFILL_MAX_TOKENS, C.D], pl.FP32],
    output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    output: pl.Out[pl.Tensor[[C.T_DYN, C.D], pl.BF16]],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    return prefill_attn_c1a_reuse(
        x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
        kv_norm_weight, attn_sink, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin,
        window_slots, window_indices, window_cache, window_cache_scale,
        compressed_cache, compressed_cache_scale, compressed_indices,
        output_window, output_arrived, output, 0, tp_rank, num_tokens, 1,
    )


@pl.jit.host
def l3_prefill_attn_c1a_reuse_test(
    x: pl.Tensor[[C.TP_SIZE, C.T_DYN, C.D], pl.BF16],
    wq_a: pl.Tensor[[C.TP_SIZE, C.D, C.Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[C.TP_SIZE, C.D // 32, C.Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[C.TP_SIZE, C.Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[C.TP_SIZE, C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[
        [C.TP_SIZE, C.Q_LORA // 32, C.LOCAL_H * C.HEAD_DIM],
        pl.FP8E8M0,
    ],
    wkv: pl.Tensor[[C.TP_SIZE, C.D, C.HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[C.TP_SIZE, C.D // 32, C.HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[C.TP_SIZE, C.HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[C.TP_SIZE, C.LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[C.TP_SIZE, C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[C.TP_SIZE, C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[
        [C.TP_SIZE, C.LOCAL_O_WIDTH // 32, C.D],
        pl.FP8E8M0,
    ],
    rope_cos: pl.Tensor[[C.TP_SIZE, C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[C.TP_SIZE, C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[C.TP_SIZE, C.T_DYN], pl.INT64],
    window_indices: pl.Tensor[[C.TP_SIZE, C.T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[
        pl.Tensor[[C.TP_SIZE, C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM], pl.FP8E4M3FN]
    ],
    window_cache_scale: pl.InOut[
        pl.Tensor[
            [C.TP_SIZE, C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP],
            pl.FP8E8M0,
        ]
    ],
    compressed_cache: pl.Tensor[
        [C.TP_SIZE, C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // 2],
        pl.UINT8,
    ],
    compressed_cache_scale: pl.Tensor[
        [C.TP_SIZE, C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.COMPRESSED_CACHE_GROUP],
        pl.FP8E4M3FN,
    ],
    compressed_indices: pl.Tensor[[C.TP_SIZE, C.T_DYN, C.INDEX_TOPK], pl.INT32],
    output: pl.Out[pl.Tensor[[C.TP_SIZE, C.T_DYN, C.D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    output_window_buf = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.FP32)
    output_arrived_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        output_window = pld.window(output_window_buf, [PREFILL_MAX_TOKENS, D], dtype=pl.FP32)
        output_arrived = pld.window(output_arrived_buf, [TP_SIZE, 1], dtype=pl.INT32)
        # The rank takes these scales as MX_B_NN; a bare slice is ND, so annotate it.
        wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
        wq_b_scale_r: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale[rank]
        wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
        wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
        prefill_attn_c1a_reuse_test(
            x[rank], wq_a[rank], wq_a_scale_r, q_norm_weight[rank],
            wq_b[rank], wq_b_scale_r, wkv[rank], wkv_scale_r,
            kv_norm_weight[rank], attn_sink[rank], wo_a[rank], wo_b[rank],
            wo_b_scale_r, rope_cos[rank], rope_sin[rank], window_slots[rank],
            window_indices[rank], window_cache[rank], window_cache_scale[rank],
            compressed_cache[rank], compressed_cache_scale[rank], compressed_indices[rank],
            output_window, output_arrived, output[rank], rank, num_tokens, device=rank,
        )


def build_tensor_specs(token_count=CASE_TOKENS, case_name=CASE_DEFAULT):
    return make_tensor_specs(REUSE_INPUT_NAMES, ("output",), token_count, case_name)


def golden_prefill_attn_c1a_reuse_case(tensors):
    apply_distributed_golden("reuse", golden_prefill_attn_c1a_reuse, tensors)


__all__ = ["golden_prefill_attn_c1a_reuse", "prefill_attn_c1a_reuse"]

# Keep the pre-split names available to the HC orchestration wrapper.
golden_prefill_c1a_reuse = golden_prefill_attn_c1a_reuse
prefill_c1a_reuse = prefill_attn_c1a_reuse


def validate(argv=None):
    import argparse

    from pypto.ir import DistributedConfig

    from golden import run
    parser = argparse.ArgumentParser(description="DeepSeek V4.1 prefill C1A reuse validation")
    parser.add_argument("-p", "--platform", default="a5", choices=("a5",))
    parser.add_argument("-d", "--device", default=",".join(str(rank) for rank in range(C.TP_SIZE)))
    parser.add_argument("--tp", type=int, default=2, choices=(1, 2, 4))
    parser.add_argument("--dp", type=int, default=1, choices=(1,))
    parser.add_argument("--tokens", type=int, default=CASE_TOKENS, help="token count for the causal case")
    parser.add_argument("--case", default=CASE_DEFAULT, choices=CASE_NAMES)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data")
    parser.add_argument("--dump-passes", action="store_true")
    parser.add_argument("--runtime-dir")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    args = parser.parse_args(argv)

    if args.tp != C.TP_SIZE:
        parser.error(f"--tp was parsed as TP{C.TP_SIZE}, got --tp {args.tp}")
    if args.case == "causal" and not 1 <= args.tokens <= CASE_MAX_TOKENS:
        parser.error(f"--tokens must be in [1, {CASE_MAX_TOKENS}]")
    device_ids = [int(device) for device in args.device.split(",")]
    if len(device_ids) != C.TP_SIZE:
        parser.error(f"need exactly {C.TP_SIZE} devices, got {device_ids}")

    window_cache_compare = quantized_cache_compare(
        "window_cache",
        "window_cache_scale",
        "window_slots",
        CACHE_MAX_RELATIVE_L2,
    )
    result = run(
        fn=l3_prefill_attn_c1a_reuse_test,
        specs=build_tensor_specs(args.tokens, args.case),
        golden_fn=golden_prefill_attn_c1a_reuse_case,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        config={
            "platform": args.platform,
            "distributed_config": DistributedConfig(device_ids=device_ids, num_sub_workers=0),
            "dump_passes": args.dump_passes,
            "enable_chip_swimlane": args.enable_chip_swimlane,
        },
        compare_fn={
            "output": attention_output_compare("reuse"),
            "window_cache": window_cache_compare,
            "window_cache_scale": window_cache_compare,
        },
    )
    return result


def main():
    """Run local validation and return a failing exit status on precision errors."""
    result = validate()
    if not result.passed:
        raise SystemExit(result.error or 1)


if "pytest" in sys.modules:
    import pytest

    @pytest.mark.parametrize("tp,dp", [(1, 1), (4, 1)])
    def test_precision(tp, dp, a5_args):
        """Validate the operator against its golden reference on A5."""
        result = validate(a5_args(tp=tp, dp=dp))
        assert result.passed, result.error


if __name__ == _SCRIPT_ENTRY_POINT:
    main()
