/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0.
 */

#ifndef PYPTO_DEEPSEEK_FUSED_PRE_NORM_BODY_HPP
#define PYPTO_DEEPSEEK_FUSED_PRE_NORM_BODY_HPP

#include <cstdint>

#include <pto/pto-inst.hpp>

#include "intrinsic.h"
#include "kernel_operator.h"
#include "tensor.h"

#include "ffn_norm_generated.hpp"
#include "mix_x_generated.hpp"
#include "split_pre_post_generated.hpp"

#ifdef __PTO_AUTO__
#error "fused_pre_norm soft SYNCALL requires the manual extern build path"
#endif

namespace deepseek_fused_pre_norm {

// This entry is a pure-AIV extern. On Ascend A2/A3 it must be launched as one
// synchronously-started 8-AIV wave. Each soft barrier is called by the dense
// prefix of logical lanes that covers both sides of that dependency edge.
constexpr int32_t kAivLanes = 8;
constexpr int64_t kTokenTile = 8;
constexpr int64_t kMixSlicesPerTokenTile = 4;
constexpr int64_t kGateTokenTile = 16;

// PyPTO packs every Tensor argument before every scalar argument. Keep this
// table in lockstep with the Python @pl.jit.extern signature.
enum TensorArg : int32_t {
  kXMixed = 0,       // BF16 [T, 4096], first Out and return[0]
  kXFlat = 1,        // FP32 [T, 16384]
  kInvRms = 2,       // FP32 [t_linear, 1]
  kMixesRaw = 3,     // FP32 [t_linear, 32]
  kHcBase = 4,       // FP32 [24]
  kNormWeight = 5,   // BF16 [4096] or [1, 4096]
  kPreValue = 6,     // FP32 [t_linear, 8]
  kPost = 7,         // FP32 [T, 4]
  kXg = 8,           // FP32 [T_PAD, 4096]
  kFfnInvRms = 9,    // FP32 [T_PAD, 1]
  kXnScale = 10,     // FP32 [T_PAD, 1]
  kXNormScale = 11,  // FP32 [T, 1]
  kSyncWorkspace = 12,  // INT32 [kSoftSyncWorkspaceWords], per-grid InOut
  kTensorArgCount = 13,
};

enum ScalarArg : int32_t {
  kScale0 = kTensorArgCount,  // FP32 bit pattern
  kScale1,                    // FP32 bit pattern
  kNumTokens,                 // INT32 in the low 32 bits
  kProductionArgCount,
  kDebugStopAfter = kProductionArgCount,
  kDebugArgCount,
};

// The debug-only entry selects one of these compile-time bodies with one
// uniform scalar. Production always instantiates kFull and contains no
// stop-mode branch.
enum class StopAfter : int32_t {
  kSplitBeforeBarrier1 = 0,
  kAfterBarrier1 = 1,
  kMixBeforeBarrier2 = 2,
  kAfterBarrier2 = 3,
  kFull = 4,
};

// The normal fused entry uses the participant-minimal dense-prefix candidate.
// The test-only correctness baseline forces every existing barrier to use all
// 8 launched AIV lanes. Both remain experimental until PTO-ISA's finite poll
// timeout is fail-closed; the policy is a template argument so neither kernel
// contains a runtime mode branch.
enum class BarrierPolicy : int32_t {
  kDenseTarget = 0,
  kAtomicEightWayBaseline = 1,
};

static_assert(
    kAivLanes == 8,
    "the atomic 8/8 correctness baseline requires an 8-AIV launch");

#ifdef __DAV_C220_VEC__
constexpr uint64_t kA2A3DcciLineBytes = 64U;
constexpr int32_t kSoftSyncCounterWords =
    pto::SYNCALL_SOFT_WORKSPACE_INT32;
constexpr int32_t kBarrier1OffsetWords = 0;
constexpr int32_t kBarrier2OffsetWords = kSoftSyncCounterWords;
constexpr int32_t kSoftSyncWorkspaceWords =
    2 * kSoftSyncCounterWords;
constexpr int64_t kPreValueWidth = 8;
constexpr int64_t kHiddenDim = 4096;
constexpr int64_t kMixSliceWidth = 1024;

static_assert(
    kSoftSyncCounterWords * sizeof(int32_t) == kA2A3DcciLineBytes);
static_assert(
    (kBarrier2OffsetWords * sizeof(int32_t)) % kA2A3DcciLineBytes == 0U);
static_assert(kSoftSyncWorkspaceWords == 32);
static_assert(
    kA2A3DcciLineBytes != 0U &&
    (kA2A3DcciLineBytes & (kA2A3DcciLineBytes - 1U)) == 0U);
static_assert(kHiddenDim == kMixSlicesPerTokenTile * kMixSliceWidth);

// A2/A3 dcci operates on a complete 64-byte line. The producer form writes
// dirty data out; the consumer form invalidates any stale local copy. Callers
// issue one DSB after all ranges owned by the lane have been processed.
static __aicore__ __attribute__((always_inline)) void dcci_range(
    __gm__ void *address, uint64_t bytes, bool publish) {
  if (bytes == 0U) {
    return;
  }

  const uint64_t raw = reinterpret_cast<uint64_t>(address);
  const uint64_t first = raw & ~(kA2A3DcciLineBytes - 1U);
  const uint64_t end =
      (raw + bytes + kA2A3DcciLineBytes - 1U) &
      ~(kA2A3DcciLineBytes - 1U);
  for (uint64_t line = first; line < end; line += kA2A3DcciLineBytes) {
    __asm__ __volatile__("");
    if (publish) {
      dcci(reinterpret_cast<__gm__ void *>(line), SINGLE_CACHE_LINE,
           CACHELINE_OUT);
    } else {
      dcci(reinterpret_cast<__gm__ void *>(line), SINGLE_CACHE_LINE);
    }
    __asm__ __volatile__("");
  }
}

static __aicore__ __attribute__((always_inline)) void publish_pre_value(
    __gm__ float *pre_value, int32_t lane, int32_t split_work) {
  pipe_barrier(PIPE_ALL);
  constexpr uint64_t kTileBytes =
      kTokenTile * kPreValueWidth * sizeof(float);
  constexpr int64_t kTileElements = kTokenTile * kPreValueWidth;
  for (int32_t logical_block = lane; logical_block < split_work;
       logical_block += kAivLanes) {
    dcci_range(
        static_cast<__gm__ void *>(
            pre_value + static_cast<int64_t>(logical_block) * kTileElements),
        kTileBytes, true);
  }
  dsb(DSB_DDR);
}

static __aicore__ __attribute__((always_inline)) void acquire_pre_value(
    __gm__ float *pre_value, int32_t lane, int32_t mix_work) {
  constexpr uint64_t kTileBytes =
      kTokenTile * kPreValueWidth * sizeof(float);
  constexpr int64_t kTileElements = kTokenTile * kPreValueWidth;
  for (int32_t logical_block = lane; logical_block < mix_work;
       logical_block += kAivLanes) {
    const int64_t token_tile =
        logical_block / kMixSlicesPerTokenTile;
    dcci_range(
        static_cast<__gm__ void *>(pre_value + token_tile * kTileElements),
        kTileBytes, false);
  }
  dsb(DSB_DDR);
}

static __aicore__ __attribute__((always_inline)) void publish_x_mixed(
    __gm__ bfloat16_t *x_mixed, int32_t lane, int32_t mix_work) {
  pipe_barrier(PIPE_ALL);
  constexpr uint64_t kSliceBytes =
      kMixSliceWidth * sizeof(bfloat16_t);
  for (int32_t logical_block = lane; logical_block < mix_work;
       logical_block += kAivLanes) {
    const int64_t token_tile =
        logical_block / kMixSlicesPerTokenTile;
    const int64_t d_slice =
        logical_block % kMixSlicesPerTokenTile;
    const int64_t first_token = token_tile * kTokenTile;
    for (int64_t row = 0; row < kTokenTile; ++row) {
      const int64_t offset =
          (first_token + row) * kHiddenDim + d_slice * kMixSliceWidth;
      dcci_range(
          static_cast<__gm__ void *>(x_mixed + offset), kSliceBytes, true);
    }
  }
  dsb(DSB_DDR);
}

static __aicore__ __attribute__((always_inline)) void acquire_x_mixed(
    __gm__ bfloat16_t *x_mixed, int32_t lane, int32_t ffn_work) {
  constexpr uint64_t kRowBytes = kHiddenDim * sizeof(bfloat16_t);
  for (int32_t logical_block = lane; logical_block < ffn_work;
       logical_block += kAivLanes) {
    dcci_range(
        static_cast<__gm__ void *>(
            x_mixed + static_cast<int64_t>(logical_block) * kHiddenDim),
        kRowBytes, false);
  }
  dsb(DSB_DDR);
}

static __aicore__ __attribute__((always_inline)) void soft_sync_aiv(
    __gm__ int32_t *workspace_base,
    int32_t offset_words,
    int32_t participants) {
  pto::GlobalTensor<
      int32_t,
      pto::Shape<>,
      pto::Stride<>
  > workspace(workspace_base + offset_words);

  pto::SYNCALL<
      pto::SyncAllMode::Soft,
      pto::SyncCoreType::AIVOnly
  >(workspace, participants);
}
#endif  // __DAV_C220_VEC__

template <typename T>
static __aicore__ __attribute__((always_inline)) __gm__ T *
tensor_data(__gm__ int64_t *args, int32_t index) {
  __gm__ Tensor *tensor = reinterpret_cast<__gm__ Tensor *>(args[index]);
  return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) +
         tensor->start_offset;
}

static __aicore__ __attribute__((always_inline)) int64_t
tensor_dim(__gm__ int64_t *args, int32_t index, int32_t axis) {
  __gm__ Tensor *tensor = reinterpret_cast<__gm__ Tensor *>(args[index]);
  return static_cast<int64_t>(tensor->shapes[axis]);
}

static __aicore__ __attribute__((always_inline)) float
unpack_float_scalar(__gm__ int64_t *args, int32_t index) {
  union {
    uint64_t bits;
    float value;
  } scalar;
  scalar.bits = static_cast<uint64_t>(args[index]);
  return scalar.value;
}

static __aicore__ __attribute__((always_inline)) int32_t
compute_ffn_work(int64_t num_tokens, int64_t tokens) {
  int64_t active_tokens = num_tokens;
  if (active_tokens < 0) {
    active_tokens = 0;
  }
  if (active_tokens > tokens) {
    active_tokens = tokens;
  }

  int64_t active_gate_tokens =
      ((active_tokens + kGateTokenTile - 1) / kGateTokenTile) *
      kGateTokenTile;
  if (active_gate_tokens > tokens) {
    active_gate_tokens = tokens;
  }
  return static_cast<int32_t>(active_gate_tokens);
}

template <BarrierPolicy Policy>
static __aicore__ __attribute__((always_inline)) int32_t
select_barrier_participants(int32_t dense_participants) {
  static_assert(
      Policy == BarrierPolicy::kDenseTarget ||
          Policy == BarrierPolicy::kAtomicEightWayBaseline,
      "unsupported fused_pre_norm barrier policy");
  if constexpr (Policy == BarrierPolicy::kAtomicEightWayBaseline) {
    return dense_participants > 0 ? kAivLanes : 0;
  }
  return dense_participants;
}

template <
    StopAfter Stop,
    BarrierPolicy Policy = BarrierPolicy::kDenseTarget
>
static __aicore__ __attribute__((always_inline)) void
run_fused_pre_norm(__gm__ int64_t *args) {
#ifdef __DAV_C220_VEC__
  const int32_t lane = static_cast<int32_t>(get_block_idx(args));
  const int64_t tokens = tensor_dim(args, kXMixed, 0);
  const int64_t t_linear = tensor_dim(args, kPreValue, 0);
  const int32_t split_work =
      static_cast<int32_t>(tokens / kTokenTile);
  const int32_t mix_work =
      static_cast<int32_t>(split_work * kMixSlicesPerTokenTile);
  const int64_t num_tokens =
      static_cast<int64_t>(static_cast<int32_t>(args[kNumTokens]));
  const int32_t ffn_work = compute_ffn_work(num_tokens, tokens);
  const int32_t active_split =
      split_work < kAivLanes ? split_work : kAivLanes;
  const int32_t active_mix =
      mix_work < kAivLanes ? mix_work : kAivLanes;
  const int32_t active_ffn =
      ffn_work < kAivLanes ? ffn_work : kAivLanes;
  const int32_t dense_barrier1_participants =
      active_split > active_mix ? active_split : active_mix;
  const int32_t dense_barrier2_participants =
      active_mix > active_ffn ? active_mix : active_ffn;
  const int32_t barrier1_participants =
      select_barrier_participants<Policy>(dense_barrier1_participants);
  const int32_t barrier2_participants =
      select_barrier_participants<Policy>(dense_barrier2_participants);

  __gm__ bfloat16_t *x_mixed = tensor_data<bfloat16_t>(args, kXMixed);
  __gm__ float *x_flat = tensor_data<float>(args, kXFlat);
  __gm__ float *inv_rms = tensor_data<float>(args, kInvRms);
  __gm__ float *mixes_raw = tensor_data<float>(args, kMixesRaw);
  __gm__ float *hc_base = tensor_data<float>(args, kHcBase);
  __gm__ bfloat16_t *norm_weight =
      tensor_data<bfloat16_t>(args, kNormWeight);
  __gm__ float *pre_value = tensor_data<float>(args, kPreValue);
  __gm__ float *post = tensor_data<float>(args, kPost);
  __gm__ float *xg = tensor_data<float>(args, kXg);
  __gm__ float *ffn_inv_rms = tensor_data<float>(args, kFfnInvRms);
  __gm__ float *xn_scale = tensor_data<float>(args, kXnScale);
  __gm__ float *x_norm_scale = tensor_data<float>(args, kXNormScale);
  __gm__ int32_t *sync_workspace =
      tensor_data<int32_t>(args, kSyncWorkspace);

  const float scale0 = unpack_float_scalar(args, kScale0);
  const float scale1 = unpack_float_scalar(args, kScale1);

  // Preserve the standalone split_pre_post logical block mapping. Physical
  // lanes with lane >= split_work do no work in this phase.
  for (int32_t logical_block = lane; logical_block < split_work;
       logical_block += kAivLanes) {
    deepseek_fused_pre_norm_split_generated::split_pre_post(
        inv_rms, hc_base, mixes_raw, pre_value, post, scale0, scale1, t_linear,
        tokens, logical_block, split_work);
  }

  if constexpr (Stop == StopAfter::kSplitBeforeBarrier1) {
    return;
  }

  // Publish this lane's contiguous 8x8 FP32 tiles before mix_x repartitions
  // work by (token tile, 1024-wide D slice).
  publish_pre_value(pre_value, lane, split_work);
  if (barrier1_participants > 0 && lane < barrier1_participants) {
    soft_sync_aiv(
        sync_workspace, kBarrier1OffsetWords, barrier1_participants);
  }

  if constexpr (Stop == StopAfter::kAfterBarrier1) {
    return;
  }

  acquire_pre_value(pre_value, lane, mix_work);

  // Preserve mix_x's logical block count rather than passing the physical
  // 8-lane count. Prefill T=128 has 64 tasks, hence the grid-stride loop.
  for (int32_t logical_block = lane; logical_block < mix_work;
       logical_block += kAivLanes) {
    deepseek_fused_pre_norm_mix_generated::mix_x(
        pre_value, x_mixed, x_flat, t_linear, tokens, tokens, logical_block,
        mix_work);
  }

  if constexpr (Stop == StopAfter::kMixBeforeBarrier2) {
    return;
  }

  // A mix task writes one 1024-wide BF16 slice in each of 8 rows. Publish
  // those strided slices before ffn_norm reloads complete 4096-element rows.
  publish_x_mixed(x_mixed, lane, mix_work);
  if (barrier2_participants > 0 && lane < barrier2_participants) {
    soft_sync_aiv(
        sync_workspace, kBarrier2OffsetWords, barrier2_participants);
  }

  if constexpr (Stop == StopAfter::kAfterBarrier2) {
    return;
  }

  acquire_x_mixed(x_mixed, lane, ffn_work);

  // ffn_norm preserves the gate's clamp/round-to-16/clamp-to-T semantics.
  // With num_tokens=0 this is a zero-trip loop after all active mix lanes have
  // crossed barrier 2; T=128 can require multiple logical tokens per AIV lane.
  for (int32_t logical_block = lane; logical_block < ffn_work;
       logical_block += kAivLanes) {
    deepseek_fused_pre_norm_ffn_generated::ffn_norm(
        x_mixed, norm_weight, xg, ffn_inv_rms, x_norm_scale, xn_scale,
        logical_block, ffn_work);
  }
#else
  (void)args;
#endif  // __DAV_C220_VEC__
}

}  // namespace deepseek_fused_pre_norm

#endif  // PYPTO_DEEPSEEK_FUSED_PRE_NORM_BODY_HPP
