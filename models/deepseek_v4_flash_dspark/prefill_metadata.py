# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared dynamic request axes and device-side lowering for packed prefill."""

import pypto.language as pl


# Dynamic dimensions shared by packed prefill metadata consumers.
REQUESTS_DYN = pl.dynamic("PREFILL_METADATA_REQUESTS_DYN")
QUERY_START_LOC_DYN = pl.dynamic("PREFILL_METADATA_QUERY_START_LOC_DYN")
LOCAL_TOKENS_DYN = pl.dynamic("PREFILL_METADATA_LOCAL_TOKENS_DYN")


@pl.jit.inline
def lower_local_request_ids(
    query_start_loc: pl.Tensor[[QUERY_START_LOC_DYN], pl.INT32],
    local_request_ids: pl.Tensor[[LOCAL_TOKENS_DYN], pl.INT32],
    local_base: pl.Scalar[pl.INT32],
):
    """Lower packed query starts into request ids for one TP rank."""
    request_count = pl.tensor.dim(query_start_loc, 0) - 1
    base = pl.cast(local_base, pl.INT32)
    local_token_count = pl.tensor.dim(local_request_ids, 0)

    with pl.spmd(1, name_hint="prefill_lower_local_request_ids"):
        block_idx = pl.tile.get_block_idx()
        if block_idx == 0:
            for local_token in pl.range(local_token_count):
                packed_token = base + pl.cast(local_token, pl.INT32)
                pl.write(local_request_ids, [local_token], pl.cast(-1, pl.INT32))
                for request in pl.range(request_count):
                    request_start = pl.read(query_start_loc, [request])
                    request_end = pl.read(query_start_loc, [request + 1])
                    if packed_token >= request_start:
                        if packed_token < request_end:
                            pl.write(local_request_ids, [local_token], pl.cast(request, pl.INT32))
    return local_request_ids


@pl.jit
def prefill_metadata_test(
    query_start_loc: pl.Tensor[[QUERY_START_LOC_DYN], pl.INT32],
    local_base: pl.Scalar[pl.INT32],
    request_ids: pl.Out[pl.Tensor[[LOCAL_TOKENS_DYN], pl.INT32]],
):
    """Test rank-local packed prefill metadata lowering."""
    query_start_loc.bind_dynamic(0, QUERY_START_LOC_DYN)
    request_ids.bind_dynamic(0, LOCAL_TOKENS_DYN)
    return lower_local_request_ids(query_start_loc, request_ids, local_base)


def build_tensor_specs():
    import torch
    from golden import ScalarSpec, TensorSpec

    query_start_loc = torch.tensor([0, 3, 7], dtype=torch.int32)
    local_token_count = 4
    request_ids = torch.full((local_token_count,), -1, dtype=torch.int32)
    local_base = local_token_count
    return [
        TensorSpec("query_start_loc", list(query_start_loc.shape), torch.int32, init_value=query_start_loc),
        ScalarSpec("local_base", torch.int32, local_base),
        TensorSpec("request_ids", list(request_ids.shape), torch.int32),
    ]


def golden_prefill_metadata(tensors):
    import torch

    query_start_loc = tensors["query_start_loc"]
    local_token_count = tensors["request_ids"].shape[0]
    local_base = int(tensors["local_base"])
    total_tokens = int(query_start_loc[-1])
    request_ids = tensors["request_ids"]
    request_ids.fill_(-1)
    for local_token in range(local_token_count):
        packed_token = local_base + local_token
        if packed_token >= total_tokens:
            continue
        for request in range(query_start_loc.numel() - 1):
            if query_start_loc[request] <= packed_token < query_start_loc[request + 1]:
                request_ids[local_token] = request
                break
    expected = torch.tensor([1, 1, 1, -1], dtype=torch.int32)
    if not torch.equal(request_ids, expected):
        raise AssertionError(f"unexpected request ids: {request_ids.tolist()}")


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    result = run(
        fn=prefill_metadata_test,
        specs=build_tensor_specs(),
        golden_fn=golden_prefill_metadata,
        compile_only=args.compile_only,
        runtime_cfg={"platform": args.platform, "device_id": args.device},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
