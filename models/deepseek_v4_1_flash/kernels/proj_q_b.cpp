// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// NZ attention Q-B projection.
#include "q_lora_projection.hpp"


#if defined(__DAV_CUBE__)
template <int N, int Workers>
__aicore__ inline void dispatch_qb(__gm__ int64_t* args) {
    int rows = args[7];
    if (rows <= 80) {
        project_qb<N, Workers, false, true>(args);
        return;
    }
    int tasks = (rows + 79) / 80 * 8;
    for (int task = get_block_idx(args); task < tasks; task += get_block_num(args)) {
        int first = task / 8 * 80;
        int count = rows - first < 80 ? rows - first : 80;
        project_qb<N, 0, false, true>(args, first, count, task % 8, 8);
    }
}
#endif
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_CUBE__)
    if (args[8] > 16384) dispatch_qb<32768, 32>(args);
    else if (args[8] > 8192) dispatch_qb<16384, 32>(args);
    else if (args[8] > 4096) dispatch_qb<8192, 32>(args);
    else dispatch_qb<4096, 16>(args);
#elif defined(__DAV_VEC__)
    int rows = args[7];
    if (rows <= 80) {
        project_qb_rope(args);
        return;
    }
    int tasks = (rows + 79) / 80 * 8;
    for (int task = get_block_idx(args); task < tasks; task += get_block_num(args)) {
        int first = task / 8 * 80;
        int count = rows - first < 80 ? rows - first : 80;
        project_qb_rope(args, first, count, task % 8, 8);
    }
#endif
}
