/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0.
 */

#include <cstdint>

#ifdef __CPU_SIM
#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__
#endif
#endif  // __CPU_SIM

#include "tensor.h"

#ifdef __CPU_SIM

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) { (void)args; }

#else

#include "../kernel/fused_body.hpp"

// Debug ABI appends one uniform INT32 scalar after production's 13 tensors and
// 3 scalars:
//   0: split body only, before soft barrier #1
//   1: through soft barrier #1
//   2: through mix body, before soft barrier #2
//   3: through soft barrier #2
//   4: full fused body
// The runtime switch selects compile-time-specialized bodies. No AIV lane may
// receive a different stop value.
extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
  using deepseek_fused_pre_norm::StopAfter;
  const int32_t stop_after =
      static_cast<int32_t>(args[deepseek_fused_pre_norm::kDebugStopAfter]);
  switch (stop_after) {
  case static_cast<int32_t>(StopAfter::kSplitBeforeBarrier1):
    deepseek_fused_pre_norm::run_fused_pre_norm<
        StopAfter::kSplitBeforeBarrier1>(args);
    break;
  case static_cast<int32_t>(StopAfter::kAfterBarrier1):
    deepseek_fused_pre_norm::run_fused_pre_norm<StopAfter::kAfterBarrier1>(
        args);
    break;
  case static_cast<int32_t>(StopAfter::kMixBeforeBarrier2):
    deepseek_fused_pre_norm::run_fused_pre_norm<
        StopAfter::kMixBeforeBarrier2>(args);
    break;
  case static_cast<int32_t>(StopAfter::kAfterBarrier2):
    deepseek_fused_pre_norm::run_fused_pre_norm<StopAfter::kAfterBarrier2>(
        args);
    break;
  case static_cast<int32_t>(StopAfter::kFull):
  default:
    deepseek_fused_pre_norm::run_fused_pre_norm<StopAfter::kFull>(args);
    break;
  }
}

#endif
