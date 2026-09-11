/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0.
 * Please refer to the LICENSE file in the root of the software repository.
 */

#ifndef PYPTO_QWEN_RUNTIME_TENSOR_COMPAT_HPP
#define PYPTO_QWEN_RUNTIME_TENSOR_COMPAT_HPP

#include "tensor.h"

// `Tensor` is the 128-byte descriptor a kernel reads out of the task payload:
// each runtime's own working type (`simpler::tmr::Tensor` /
// `simpler::hbg::Tensor`), exported unqualified by the per-runtime `tensor.h`
// umbrella included above so a source compiled under either runtime need not
// pick one.
//
// **Supported Simpler ABI: #2044 and newer.**  The descriptor has been spelled
// three ways, and this file used to probe for all of them:
//
//   Tensor      -- #2044 named the umbrella's export `Tensor` again, once
//                  #2032 and #2038 had cut the kernel and orchestration include
//                  paths to `task_interface/buffer.h`, whose global `Tensor`
//                  was what the older name worked around.  Current.
//   TaskTensor  -- #1974 split the fused type: `ChipTensor` kept the name but
//                  became the 72-byte *argument* as it arrives at the boundary,
//                  while the 128-byte descriptor became each runtime's own
//                  type, exported as `TaskTensor`.
//   ChipTensor  -- #1681 renamed the descriptor for the address-free Buffer
//                  ABI, which also added task_interface/buffer.h.
//
// The probe is gone because it could not be made honest.  It keyed on
// `tensormap_and_ringbuffer/tensor.h`, which #1974 added and #2044 left in
// place, so it kept selecting `TaskTensor` after the rename and broke every
// translation unit of this extern.  No `__has_include` fixes that: #2044
// changed only the exported name, adding no header to the kernel include path,
// so nothing on that path distinguishes it from the #1974 window.
//
// Naming one spelling and stating the supported range instead means an
// unsupported runtime fails loudly here rather than being silently
// mis-selected, and it gives up nothing reachable: every Simpler revision PyPTO
// has pinned since the rename is post-#2044, including the one it later rolled
// back to.  Widening the range again needs a real version signal from Simpler
// (a version macro, or a header that moves with the ABI), not another proxy.
//
// The static_assert is the other half of failing loudly.  Naming `ChipTensor`
// against a post-#1974 runtime still *compiles* -- the kernel just reads
// `owner_task_id`'s bytes as `start_offset` and garbage as `shapes`/`strides`,
// which surfaces on device as an fftsplus aivector error rather than as a
// diagnostic.
using PyPTORuntimeTensor = Tensor;

static_assert(
    sizeof(PyPTORuntimeTensor) == 128,
    "PyPTORuntimeTensor must be the 128-byte payload descriptor; a 72-byte "
    "match means this picked up the boundary ChipTensor"
);

#endif  // PYPTO_QWEN_RUNTIME_TENSOR_COMPAT_HPP
