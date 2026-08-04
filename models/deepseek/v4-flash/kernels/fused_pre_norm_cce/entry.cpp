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

#include "kernel/fused_body.hpp"

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
  deepseek_fused_pre_norm::run_fused_pre_norm<
      deepseek_fused_pre_norm::StopAfter::kFull>(args);
}

#endif
