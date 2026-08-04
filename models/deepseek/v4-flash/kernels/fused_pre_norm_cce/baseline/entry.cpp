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

// Test-only correctness baseline: every dependency edge that exists for this
// shape is an atomic 8-way barrier. The normal model entry continues to use
// entry.cpp and the default dense-target candidate; neither policy is ready
// for release until PTO-ISA's finite poll timeout is fail-closed.
extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
  deepseek_fused_pre_norm::run_fused_pre_norm<
      deepseek_fused_pre_norm::StopAfter::kFull,
      deepseek_fused_pre_norm::BarrierPolicy::kAtomicEightWayBaseline>(args);
}

#endif
