/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <cstdint>

#include "tensor.h"

#ifdef __CPU_SIM
#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) { (void)args; }

#else

// Stage 1 isolation entry: runs ONLY the rope_qkv producer (no attention, no
// cross-core sync) so the C++ rope compute can be validated byte-exact against
// decode_fwd.py's rope_qkv before being fused into the attention kernel.
#define QWEN_FAI_ROPE_ONLY_ABI
#include "../kernel/fai_body.hpp"

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    run_qwen_rope_qkv(args);
}

#endif
