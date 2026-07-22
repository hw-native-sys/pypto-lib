/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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

#include "../kernel/fai_body.hpp"

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    GM_ADDR metadata = tensor_data<uint8_t>(args, 6);
    acquire_qwen_fai_metadata(metadata);
    run_qwen_fai<false, false, KernelCommon::FaiKernel::MaskType::NO_MASK, -1, 7, 8>(args);
}

#endif
