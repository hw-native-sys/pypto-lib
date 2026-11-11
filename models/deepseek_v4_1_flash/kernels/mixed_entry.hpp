// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
#pragma once
#include <pto/pto-inst.hpp>

namespace native_lifecycle {
__aicore__ inline void enter_dual_aiv() {
    pipe_barrier(PIPE_ALL);
#if defined(__DAV_CUBE__)
    set_intra_block(PIPE_FIX, 8); set_intra_block(PIPE_FIX, 24);
    wait_intra_block(PIPE_FIX, 9); wait_intra_block(PIPE_FIX, 25);
    set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID7);
    wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID7);
#elif defined(__DAV_VEC__)
    wait_intra_block(PIPE_V, 8);
    set_intra_block(PIPE_V, 9);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
#endif
}
} // namespace native_lifecycle
