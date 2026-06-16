# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE end-to-end prefill harness.

MoE itself is token-major after the refreshed contract, so the prefill path can
reuse the decode MoE implementation while the current prefill token count equals
the decode token count.
"""

import pypto.language as pl

from config import PREFILL_BATCH, PREFILL_SEQ
from moe import (
    D,
    HC_DIM,
    HC_MULT,
    MIX_HC,
    MOE_INTER,
    N_EXPERTS,
    N_LOCAL_EXPERTS,
    TOPK,
    VOCAB,
    T as MOE_T,
    build_tensor_specs as build_moe_tensor_specs,
    golden_moe as golden_prefill_moe,
    moe,
)


B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S

assert T == MOE_T, (
    "prefill_moe currently reuses token-major decode MoE helpers; update the "
    "helper constants or add prefill-specific helpers when PREFILL_BATCH * "
    "PREFILL_SEQ differs from DECODE_BATCH * DECODE_SEQ"
)


@pl.jit
def prefill_moe(
    x_hc:            pl.Tensor[[T, HC_MULT, D],                  pl.BF16],
    hc_ffn_fn:       pl.Tensor[[MIX_HC, HC_DIM],                 pl.FP32],
    hc_ffn_scale:    pl.Tensor[[3],                              pl.FP32],
    hc_ffn_base:     pl.Tensor[[MIX_HC],                         pl.FP32],
    norm_w:          pl.Tensor[[D],                              pl.FP32],
    gate_w:          pl.Tensor[[N_EXPERTS, D],                   pl.FP32],
    gate_bias:       pl.Tensor[[N_EXPERTS],                      pl.FP32],
    layer_id:        pl.Scalar[pl.INT32],
    tid2eid:         pl.Tensor[[VOCAB, TOPK],                    pl.INT32],
    input_ids:       pl.Tensor[[T],                              pl.INT64],
    routed_w1:       pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D],  pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER],     pl.FP32],
    routed_w3:       pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D],  pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER],     pl.FP32],
    routed_w2:       pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER],  pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D],             pl.FP32],
    shared_w1:       pl.Tensor[[MOE_INTER, D],                   pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER],                      pl.FP32],
    shared_w3:       pl.Tensor[[MOE_INTER, D],                   pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER],                      pl.FP32],
    shared_w2:       pl.Tensor[[D, MOE_INTER],                   pl.INT8],
    shared_w2_scale: pl.Tensor[[D],                              pl.FP32],
    x_next:          pl.Out[pl.Tensor[[T, HC_MULT, D],           pl.BF16]],
):
    x_next = moe(
        x_hc,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        norm_w, gate_w, gate_bias,
        tid2eid, input_ids,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        x_next,
        layer_id,
    )
    return x_next


def build_tensor_specs(layer_id=0):
    return build_moe_tensor_specs(layer_id=layer_id)


if __name__ == "__main__":
    import argparse
    from golden import ratio_reldiff, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3sim",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--layer-id", type=int, default=0,
                        help="layer_id < num_hash_layers picks the hash route; "
                             ">= num_hash_layers picks the sort route")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--enable-pmu", type=int, nargs="?", const=2, default=0,
                        choices=[0, 1, 2, 4])
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    result = run_jit(
        fn=prefill_moe,
        specs=build_tensor_specs(layer_id=args.layer_id),
        golden_fn=golden_prefill_moe,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
            enable_pmu=args.enable_pmu,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "x_next": ratio_reldiff(diff_thd=0.01, pct_thd=0.05),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
