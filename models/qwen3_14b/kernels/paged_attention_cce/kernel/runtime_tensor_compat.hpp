/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0.
 * Please refer to the LICENSE file in the root of the software repository.
 */

#ifndef PYPTO_QWEN_RUNTIME_TENSOR_COMPAT_HPP
#define PYPTO_QWEN_RUNTIME_TENSOR_COMPAT_HPP

#include "tensor.h"

// The 128-byte descriptor a kernel reads out of the task payload has been
// spelled several ways across Simpler ABIs.  Keep this external kernel source
// compilable with the ones PyPTO has pinned, newest first, so a pypto-lib
// change can land before PyPTO updates its runtime pin.
//
//   Tensor      -- simpler#2044 renamed the per-runtime alias of the 128-byte
//                  descriptor (`simpler::tmr::Tensor` / `simpler::hbg::Tensor`)
//                  from `TaskTensor` to `Tensor` in the `tensor.h` shim this
//                  file already includes.
//   TaskTensor  -- simpler#1974 split the fused type: `ChipTensor` kept the
//                  name but became the 72-byte *argument* as it arrives at the
//                  boundary, while the 128-byte descriptor became each
//                  runtime's own type, aliased `TaskTensor`.  Only the short
//                  window up to #2044 used this spelling; it is not probed.
//   ChipTensor  -- simpler#1681 renamed the descriptor for the address-free
//                  Buffer ABI, which also added task_interface/buffer.h.
//   Tensor      -- the original spelling.
//
// The first probe is the header the split introduced to define the per-runtime
// descriptor.  It is a version probe, not a runtime selector: both split
// headers live under src/common and arrive together, whichever runtime is
// being built for.
//
// Getting this wrong is silent.  Naming `ChipTensor` against a post-#1974
// runtime still compiles -- the kernel just reads `owner_task_id`'s bytes as
// `start_offset` and garbage as `shapes`/`strides`, which surfaces on device
// as an fftsplus aivector error rather than as a diagnostic.  The static_assert
// below is what turns that back into a compile-time failure.
#if __has_include("tensormap_and_ringbuffer/tensor.h")
using PyPTORuntimeTensor = Tensor;
#elif __has_include("task_interface/buffer.h")
using PyPTORuntimeTensor = ChipTensor;
#else
using PyPTORuntimeTensor = Tensor;
#endif

static_assert(
    sizeof(PyPTORuntimeTensor) == 128,
    "PyPTORuntimeTensor must be the 128-byte payload descriptor; a 72-byte "
    "match means this picked up the post-#1974 boundary ChipTensor"
);

#endif  // PYPTO_QWEN_RUNTIME_TENSOR_COMPAT_HPP
