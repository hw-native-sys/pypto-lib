# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CED decoder Full Mode global KV publication from the final encoder stream."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ci: no-sim
# ci: a5

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.hc_pre import mhc_pre
from models.deepseek_v4_1_flash.prefill_attn_c1a_full import publish_c1a_global_kv
from models.deepseek_v4_1_flash.rmsnorm import rms_norm


D = C.D


@pl.jit
def publish_decoder_global_from_encoder_rank(
    x_hc: pl.Tensor[[C.T_DYN, C.HC_MULT, C.D], pl.FP32],
    pre_mix: pl.Tensor[[C.T_DYN, C.HC_MULT], pl.FP32],
    attn_norm_weight: pl.Tensor[[C.D], pl.BF16],
    compressed_rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.BF16],
    compressor_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[C.T_DYN], pl.INT64],
    index_wk: pl.Tensor[[C.HEAD_DIM, C.INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[C.INDEX_DIM], pl.BF16],
    compressed_cache: pl.InOut[
        pl.Tensor[[C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // 2], pl.UINT8]
    ],
    compressed_cache_scale: pl.InOut[
        pl.Tensor[[C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN]
    ],
    index_cache: pl.InOut[
        pl.Tensor[[C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // 2], pl.UINT8]
    ],
    index_cache_scale: pl.InOut[
        pl.Tensor[[C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // C.INDEX_CACHE_GROUP], pl.FP8E8M0]
    ],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Project every encoded row with layer 20's norm and Full Mode weights.

    Only the later attention pass is bounded to 128 rows per request. This entry
    preserves the complete ratio-one global cache needed by sparse attention.
    """
    rows = pl.tensor.dim(x_hc, 0)
    collapsed = pl.create_tensor([rows, D], dtype=pl.BF16)
    mhc_pre(x_hc, pre_mix, collapsed)
    normalized = pl.create_tensor([rows, D], dtype=pl.BF16)
    rms_norm(collapsed, attn_norm_weight, normalized)
    publish_c1a_global_kv(
        normalized, compressed_rope_cos, compressed_rope_sin, compressor_wkv,
        compressor_norm_weight, compressed_slots, index_wk, index_norm_weight,
        compressed_cache, compressed_cache_scale, index_cache, index_cache_scale,
        num_tokens,
    )
    return compressed_cache, compressed_cache_scale, index_cache, index_cache_scale


@pl.jit.host
def l3_publish_decoder_global_from_encoder(
    x_hc: pl.Tensor[[C.TP_SIZE, C.T_DYN, C.HC_MULT, C.D], pl.FP32],
    pre_mix: pl.Tensor[[C.TP_SIZE, C.T_DYN, C.HC_MULT], pl.FP32],
    attn_norm_weight: pl.Tensor[[C.TP_SIZE, C.D], pl.BF16],
    compressed_rope_cos: pl.Tensor[[C.TP_SIZE, C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[C.TP_SIZE, C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[C.TP_SIZE, C.D, C.HEAD_DIM], pl.BF16],
    compressor_norm_weight: pl.Tensor[[C.TP_SIZE, C.HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[C.TP_SIZE, C.T_DYN], pl.INT64],
    index_wk: pl.Tensor[[C.TP_SIZE, C.HEAD_DIM, C.INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[C.TP_SIZE, C.INDEX_DIM], pl.BF16],
    compressed_cache: pl.InOut[
        pl.Tensor[[C.TP_SIZE, C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // 2], pl.UINT8]
    ],
    compressed_cache_scale: pl.InOut[
        pl.Tensor[
            [C.TP_SIZE, C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.COMPRESSED_CACHE_GROUP],
            pl.FP8E4M3FN,
        ]
    ],
    index_cache: pl.InOut[
        pl.Tensor[[C.TP_SIZE, C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // 2], pl.UINT8]
    ],
    index_cache_scale: pl.InOut[
        pl.Tensor[
            [C.TP_SIZE, C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // C.INDEX_CACHE_GROUP],
            pl.FP8E8M0,
        ]
    ],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Launch the encoder-stream publisher on each TP rank."""
    for rank in pl.range(pld.world_size()):
        publish_decoder_global_from_encoder_rank(
            x_hc[rank], pre_mix[rank], attn_norm_weight[rank],
            compressed_rope_cos[rank], compressed_rope_sin[rank],
            compressor_wkv[rank], compressor_norm_weight[rank], compressed_slots[rank],
            index_wk[rank], index_norm_weight[rank], compressed_cache[rank],
            compressed_cache_scale[rank], index_cache[rank], index_cache_scale[rank],
            num_tokens, device=rank,
        )


def golden_publish_decoder_global(tensors):
    """Independent BF16, RoPE, and MXFP4 publication reference per rank."""
    from models.deepseek_v4_1_flash.attention_common import _publish_quantized_cache
    from models.deepseek_v4_1_flash.golden import (
        compressor_ratio1,
        hc_pre,
        index_key,
        rms_norm as golden_rms_norm,
        rope_interleave,
    )
    from models.deepseek_v4_1_flash.quantization import quantize_mxfp4_cache

    active = int(tensors["num_tokens"])
    for rank in range(C.TP_SIZE):
        collapsed = hc_pre(tensors["x_hc"][rank, :active], tensors["pre_mix"][rank, :active])
        x = golden_rms_norm(collapsed.to(torch.bfloat16), tensors["attn_norm_weight"][rank])
        latent = compressor_ratio1(
            x, tensors["compressor_wkv"][rank], tensors["compressor_norm_weight"][rank]
        )
        cos = tensors["compressed_rope_cos"][rank, :active]
        sin = tensors["compressed_rope_sin"][rank, :active]
        rope_dim = cos.shape[-1] * 2
        rotated = torch.cat(
            (latent[..., :-rope_dim], rope_interleave(latent[..., -rope_dim:], cos, sin)), dim=-1
        )
        slots = tensors["compressed_slots"][rank, :active]
        cache, scale = _publish_quantized_cache(
            tensors["compressed_cache"][rank], tensors["compressed_cache_scale"][rank],
            rotated, slots,
            lambda value: quantize_mxfp4_cache(value, group_size=16, scale_format="e4m3"),
        )
        tensors["compressed_cache"][rank] = cache
        tensors["compressed_cache_scale"][rank] = scale
        keys = index_key(
            latent, tensors["index_wk"][rank], tensors["index_norm_weight"][rank], cos, sin
        )
        cache, scale = _publish_quantized_cache(
            tensors["index_cache"][rank], tensors["index_cache_scale"][rank],
            keys, slots,
            lambda value: quantize_mxfp4_cache(value, group_size=32, scale_format="e8m0"),
        )
        tensors["index_cache"][rank] = cache
        tensors["index_cache_scale"][rank] = scale


def main(argv=None):
    """Validate full-token global cache publication on A5 devices."""
    from golden import run
    from models.deepseek_v4_1_flash.attention_common import quantized_cache_compare
    from models.deepseek_v4_1_flash.prefill_c1a_full import build_hc_tensor_specs
    from models.deepseek_v4_1_flash.prefill_c1a_test_utils import MXFP4_CACHE_MAX_RELATIVE_L2
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="CED decoder global KV publisher")
    parser.add_argument("-p", "--platform", default="a5", choices=("a5",))
    parser.add_argument("-d", "--device", default=",".join(map(str, range(C.TP_SIZE))))
    parser.add_argument("--tp", type=int, default=C.TP_SIZE)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--checkpoint", help="official V4.1-Flash checkpoint root for layer-20 weights")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args(argv)
    if args.tp != C.TP_SIZE or len(args.device.split(",")) != C.TP_SIZE:
        parser.error(f"expected TP{C.TP_SIZE} and exactly {C.TP_SIZE} devices")
    names = (
        "x_hc", "pre_mix", "attn_norm_weight", "compressed_rope_cos",
        "compressed_rope_sin", "compressor_wkv", "compressor_norm_weight",
        "compressed_slots", "index_wk", "index_norm_weight", "compressed_cache",
        "compressed_cache_scale", "index_cache", "index_cache_scale", "num_tokens",
    )
    source_specs = {spec.name: spec for spec in build_hc_tensor_specs(args.tokens, "causal")}
    specs = [source_specs[name] for name in names]
    if args.checkpoint:
        from models.deepseek_v4_1_flash.prefill_checkpoint_weights import (
            PrefillCheckpoint,
            bind_checkpoint_weights,
        )

        bind_checkpoint_weights(specs, PrefillCheckpoint(args.checkpoint).decoder_global_weights(C.TP_SIZE))
        print("[CED] loaded checkpoint global publisher weights for layer 20")
    compare = {}
    for name, group_size, scale_format in (
        ("compressed_cache", C.COMPRESSED_CACHE_GROUP, "e4m3"),
        ("index_cache", C.INDEX_CACHE_GROUP, "e8m0"),
    ):
        cache_compare = quantized_cache_compare(
            name, name + "_scale", "compressed_slots", MXFP4_CACHE_MAX_RELATIVE_L2,
            group_size=group_size, scale_format=scale_format,
        )
        compare[name] = cache_compare
        compare[name + "_scale"] = cache_compare
    result = run(
        fn=l3_publish_decoder_global_from_encoder,
        specs=specs,
        golden_fn=golden_publish_decoder_global,
        compile_only=args.compile_only,
        config={
            "platform": args.platform,
            "distributed_config": DistributedConfig(
                device_ids=[int(device) for device in args.device.split(",")], num_sub_workers=0
            ),
            "ring_heap": (1024 * 1024 * 1024,) * 4,
        },
        compare_fn=compare,
    )
    if not result.passed:
        raise SystemExit(result.error or 1)


if "pytest" in sys.modules:
    import pytest

    @pytest.mark.parametrize("tp", [4])
    def test_precision(tp, a5_args):
        """Validate global KV and index cache publication on four A5 devices."""
        main(a5_args(tp=tp) + ["--tokens", "32"])


if __name__ == "__main__":
    main()
