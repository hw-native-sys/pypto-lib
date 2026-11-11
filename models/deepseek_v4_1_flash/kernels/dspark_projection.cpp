// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// BF16 ND draft-tap projection with FP32 accumulation and BF16 output.
#include "bf16_projection.hpp"

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_CUBE__)
    using namespace bf16_projection;
    int rows = args[3];
    auto hidden = reinterpret_cast<__gm__ Tensor*>(args[0]);
    auto weight = reinterpret_cast<__gm__ Tensor*>(args[1]);
    auto output = reinterpret_cast<__gm__ Tensor*>(args[2]);
    StoreOutput<BFloat> emit{data<BFloat>(args, 2), output->strides[0]};
    project<15360, 5120, 160, 128, 64, 3, pto::Layout::ND>(
        data<BFloat>(args, 0), data<BFloat>(args, 1), rows, rows,
        hidden->strides[0], weight->strides[0], 80,
        get_block_idx(args), get_block_num(args), emit);
    pipe_barrier(PIPE_ALL);
#endif
}
