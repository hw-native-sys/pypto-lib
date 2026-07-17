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

#ifndef PYPTO_QWEN_FAI_BODY_HPP
#define PYPTO_QWEN_FAI_BODY_HPP

#include <cstdint>
#include <type_traits>

#ifndef TILING_KEY_VAR
#define TILING_KEY_VAR 0
#endif
#ifndef ASC_DEVKIT_MAJOR
#define ASC_DEVKIT_MAJOR 9
#define ASC_DEVKIT_MINOR 0
#define ASC_DEVKIT_PATCH 0
#define ASC_DEVKIT_VERSION_NUM 90000000
#endif

#include "tensor.h"
#include "intrinsic.h"

#include "../generated/kernel_tiling/kernel_tiling.h"
#include "metadata_layout.h"

#include "../vendor/fused_infer_attention_score/flash_attention_regular.h"

constexpr uint64_t kQwenFaiHeadDim = 128;

static __aicore__ __attribute__((always_inline)) void acquire_qwen_fai_metadata(GM_ADDR metadata) {
    uint64_t first_line = reinterpret_cast<uint64_t>(metadata) &
                          ~(static_cast<uint64_t>(qwen_fai_metadata::kDcciLineBytes) - 1);
    uint64_t end = reinterpret_cast<uint64_t>(metadata) + qwen_fai_metadata::kBarrierAlignmentOffset;
    for (uint64_t line = first_line; line < end; line += qwen_fai_metadata::kDcciLineBytes) {
        dcci(reinterpret_cast<__gm__ void *>(line), SINGLE_CACHE_LINE);
    }
    dsb(DSB_DDR);
}

template <typename T>
static __aicore__ __attribute__((always_inline)) GM_ADDR tensor_data(__gm__ int64_t *args, int32_t index) {
    __gm__ Tensor *tensor = reinterpret_cast<__gm__ Tensor *>(args[index]);
    __gm__ T *data = reinterpret_cast<__gm__ T *>(tensor->buffer.addr) + tensor->start_offset;
    return reinterpret_cast<GM_ADDR>(data);
}

template <bool IsFlashDecode>
static __aicore__ __attribute__((always_inline)) void
run_qwen_fai(__gm__ int64_t *args, __gm__ int32_t *barrier_state = nullptr) {
    using namespace NpuArch;
    using namespace KernelCommon;

    using ElementQ = bfloat16_t;
    using ElementK = bfloat16_t;
    using ElementV = bfloat16_t;
    using ElementS = float;
    using ElementP = bfloat16_t;
    using ElementO = bfloat16_t;
    using ElementLse = float;
    using ElementMask = int8_t;
    using ElementOTmp = float;
    using ElementUpdate = float;
    using ElementSink = bfloat16_t;

    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    using LayoutV = layout::RowMajor;
    using LayoutS = layout::RowMajor;
    using LayoutP = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutLse = layout::RowMajor;
    using LayoutMask = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;
    using LayoutUpdate = layout::RowMajor;
    using LayoutSink = layout::RowMajor;

    using L1TileShapeQK = GemmShape<Q_TILE_CEIL, 128, 128>;
    using L0TileShapeQK = GemmShape<128, 128, 128>;
    using DispatchPolicyQK = Gemm::MmadAtlasA2FAIQK<true, false>;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using SinkType = Gemm::GemmType<ElementSink, LayoutSink>;
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, QType, KType, SType>;

    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using PseShiftType = Gemm::GemmType<ElementQ, LayoutQ>;
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmax<
        Epilogue::LseMode::NONE, Epilogue::SinkMode::DISABLE,
        static_cast<Epilogue::MaskMode>(FaiKernel::MaskType::NO_MASK), float>;
    using EpilogueOnlineSoftmax =
        Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, MaskType, SinkType, PseShiftType>;

    using L1TileShapePV = GemmShape<128, 128, 256>;
    using L0TileShapePV = GemmShape<128, 128, 128>;
    using DispatchPolicyPV = Gemm::MmadAtlasA2FAIPV<true, false>;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, PType, VType, OTmpType>;

    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    using LseType = Gemm::GemmType<ElementLse, LayoutLse>;
    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleO<Epilogue::LseMode::NONE, float>;
    using EpilogueRescaleO =
        Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType, OUpdateType, LseType>;
    using DispatchPolicyInitOut = Epilogue::EpilogueAtlasA2InitOutWhenZero<Epilogue::LseMode::NONE>;
    using EpilogueInitOut = Epilogue::Block::BlockEpilogue<DispatchPolicyInitOut, OType, LseType>;
    using CombineScale = Epilogue::Block::CombineScale<OType, LseType>;

    using FdKernel = SplitFuse::FAInferKernel<
        BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO, EpilogueInitOut, true,
        FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND, CombineScale, true, true, true>;
    using NonFdKernel = SplitFuse::FAInferKernel<
        BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO, EpilogueInitOut, true,
        FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND>;
    using Kernel = std::conditional_t<IsFlashDecode, FdKernel, NonFdKernel>;

    GM_ADDR metadata = tensor_data<uint8_t>(args, 6);
    uint64_t cache_row_offset = static_cast<uint64_t>(args[7]);
    uint64_t cache_byte_offset = cache_row_offset * kQwenFaiHeadDim * sizeof(uint16_t);
    GM_ADDR key = tensor_data<uint16_t>(args, 1) + cache_byte_offset;
    GM_ADDR value = tensor_data<uint16_t>(args, 2) + cache_byte_offset;

    FAIKernelParams params{
        tensor_data<uint16_t>(args, 0),
        key,
        value,
        nullptr,
        nullptr,
        tensor_data<int32_t>(args, 3),
        metadata + qwen_fai_metadata::kCumulativeQOffset,
        metadata + qwen_fai_metadata::kKvLengthsOffset,
        tensor_data<uint16_t>(args, 4),
        nullptr,
        tensor_data<uint8_t>(args, 5),
        metadata + qwen_fai_metadata::kTilingOffset,
        nullptr
    };

    uint32_t sub_block_idx = 0;
#ifdef __DAV_C220_VEC__
    sub_block_idx = static_cast<uint32_t>(get_sub_block_id(args));
#endif
    Arch::PtoTopology topology{
        static_cast<uint32_t>(get_block_idx(args)), static_cast<uint32_t>(get_block_num(args)), sub_block_idx, 2
    };
    Kernel kernel;
    kernel(params, topology, barrier_state);
}

// ══════════════════════════════════════════════════════════════════════════════
// rope_qkv producer (Stage 1: validated in isolation via the rope_only entry).
// Gated so the attention path (which does not define QWEN_FAI_ROPE_ONLY_ABI) is
// untouched. Mirrors decode_fwd.py:471-590 (the standalone rope_qkv SPMD task).
// ══════════════════════════════════════════════════════════════════════════════
#ifdef QWEN_FAI_ROPE_ONLY_ABI
#include "kernel_operator.h"

// rope runs on AIV lanes only; on the AIC variant of the mixed extern every symbol
// below would be unused, so the whole compute is compiled under __DAV_C220_VEC__.
// run_qwen_rope_qkv stays declared for both core types (cube gets the no-op stub).
#ifdef __DAV_C220_VEC__

// Arg indices for the rope_only extern ABI (paged_attention_rope_only_cce).
constexpr int32_t kRopeQProjArg = 0;
constexpr int32_t kRopeKProjArg = 1;
constexpr int32_t kRopeVProjArg = 2;
constexpr int32_t kRopeInvRmsArg = 3;
constexpr int32_t kRopeQNormArg = 4;
constexpr int32_t kRopeKNormArg = 5;
constexpr int32_t kRopeSeqLensArg = 6;
constexpr int32_t kRopeSlotMappingArg = 7;
constexpr int32_t kRopeCosArg = 8;
constexpr int32_t kRopeSinArg = 9;
constexpr int32_t kRopeQueryArg = 10;       // InOut BF16: rotated Q (q_tnd_flat)
constexpr int32_t kRopeKeyCacheArg = 11;    // InOut BF16: paged K cache
constexpr int32_t kRopeValueCacheArg = 12;  // InOut BF16: paged V cache
constexpr int32_t kRopeCacheRowOffsetArg = 13;

// Static batch-16 architecture constants (Qwen3-14B decode contract).
constexpr uint32_t kQwenBatch = 16;
constexpr uint32_t kQwenNumHeads = 40;
constexpr uint32_t kQwenNumKvHeads = 8;
constexpr uint32_t kQwenHeadsPerKv = 5;  // Q_PER_KV
constexpr float kQwenRopeEps = 1.0e-6F;
constexpr float kQwenRopeHeadDimInv = 1.0F / static_cast<float>(kQwenFaiHeadDim);  // 1/128

static __aicore__ __attribute__((always_inline)) void qwen_mte2_to_v() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(e);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(e);
}
static __aicore__ __attribute__((always_inline)) void qwen_v_to_s() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(e);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(e);
}
// Return fence after an S-side GetValue: pto-isa's SumLastBlockElements does the
// FULL V->S ... S->V handshake. Without this, the scalar read by GetValue is not
// committed before the next V op (Muls) consumes it, so across repeated Q-head
// calls the scalar goes stale -> x=0 -> query=0.
static __aicore__ __attribute__((always_inline)) void qwen_s_to_v() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
    AscendC::SetFlag<AscendC::HardEvent::S_V>(e);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(e);
}
static __aicore__ __attribute__((always_inline)) void qwen_v_to_mte3() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(e);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(e);
}
static __aicore__ __attribute__((always_inline)) void qwen_mte3_to_v() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(e);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(e);
}
// Drain a UB->GM store (MTE3) before the next GM->UB load (MTE2). On C220 MTE2/MTE3
// share the external-memory DMA path; without this fence a rapid load issued right
// after a store can tear the in-flight store (manifests as whole-head GM garbage,
// intermittent, proportional to write count). qwen_mte3_to_v alone only fences V.
static __aicore__ __attribute__((always_inline)) void qwen_mte3_to_mte2() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(e);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(e);
}

// Normalize one 128-element head row and apply NeoX half-split RoPE, matching
// decode_fwd.py rope_qkv exactly:
//   x  = proj_row * inv_rms           (deferred input-RMSNorm factor; gamma upstream)
//   ih = rsqrt(sum(x^2)/128 + eps)    (per-head QK-norm)
//   x  = (x * weight) * ih            (weight broadcast; kernel's mul order)
//   rot_lo = lo*cos_lo - hi*sin_lo ; rot_hi = hi*cos_hi + lo*sin_hi   (cos/sin cols dup)
// All intermediates FP32; the rotated row is left FP32 in `out`.
static __aicore__ __attribute__((always_inline)) void qwen_normalize_rope_row(
    const AscendC::LocalTensor<float> &x,
    const AscendC::LocalTensor<float> &weight,
    const AscendC::LocalTensor<float> &cos,
    const AscendC::LocalTensor<float> &sin,
    const AscendC::LocalTensor<float> &square,
    AscendC::LocalTensor<float> &reduce,
    const AscendC::LocalTensor<float> &reduce_work,
    AscendC::LocalTensor<float> &out,
    const AscendC::LocalTensor<float> &tmp,
    float inv_rms) {
    // C220 V-pipe does NOT serialize dependent ops on the same buffer: a later op can
    // read a buffer before an earlier op's write commits (RAW hazard). EVERY place a V
    // op reads another V op's output needs a PipeBarrier<PIPE_V>() between them (mirror
    // the vendor CombineScale, which barriers between every dependent V op).
    AscendC::Muls(x, x, inv_rms, kQwenFaiHeadDim);
    AscendC::PipeBarrier<PIPE_V>();  // square must read the inv_rms-scaled x
    AscendC::Mul(square, x, x, kQwenFaiHeadDim);
    AscendC::PipeBarrier<PIPE_V>();
    // Tree-reduce the 128 squares to 8 partial sums via in-place Adds (64/32/16/8).
    AscendC::Add(square, square, square[kQwenFaiHeadDim / 2], kQwenFaiHeadDim / 2);   // 128 -> 64
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(square, square, square[kQwenFaiHeadDim / 4], kQwenFaiHeadDim / 4);   // 64 -> 32
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(square, square, square[kQwenFaiHeadDim / 8], kQwenFaiHeadDim / 8);   // 32 -> 16
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(square, square, square[kQwenFaiHeadDim / 16], kQwenFaiHeadDim / 16); // 16 -> 8
    AscendC::PipeBarrier<PIPE_V>();
    // Sum the 8 partial sums in S with LITERAL CONSTANT-index GetValue. A DYNAMIC-index
    // GetValue on a UB LocalTensor misreads on C220 (~128x); constant-index UB GetValue
    // is fine. Reducing 8->1 in V needs count<8 Adds, which HANG on C220, so sum in S.
    qwen_v_to_s();
    float ss = square.GetValue(0) + square.GetValue(1) + square.GetValue(2) + square.GetValue(3)
             + square.GetValue(4) + square.GetValue(5) + square.GetValue(6) + square.GetValue(7);
    reduce.SetValue(0, ss);
    qwen_s_to_v();
    AscendC::Muls(reduce, reduce, kQwenRopeHeadDimInv, 8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(reduce, reduce, kQwenRopeEps, 8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Rsqrt(reduce, reduce, 8);
    AscendC::PipeBarrier<PIPE_V>();
    qwen_v_to_s();
    float inv_head = reduce.GetValue(0);
    qwen_s_to_v();  // commit the scalar before V consumes it (full V->S->S->V handshake)
    AscendC::Mul(x, x, weight, kQwenFaiHeadDim);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(x, x, inv_head, kQwenFaiHeadDim);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(out, x, cos, kQwenFaiHeadDim / 2);                       // lo * cos_lo
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(tmp, x[kQwenFaiHeadDim / 2], sin, kQwenFaiHeadDim / 2);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(out, out, tmp, kQwenFaiHeadDim / 2);                     // rot_lo
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(out[kQwenFaiHeadDim / 2], x[kQwenFaiHeadDim / 2], cos[kQwenFaiHeadDim / 2],
                 kQwenFaiHeadDim / 2);                                     // hi * cos_hi
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(tmp[kQwenFaiHeadDim / 2], x, sin[kQwenFaiHeadDim / 2], kQwenFaiHeadDim / 2);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(out[kQwenFaiHeadDim / 2], out[kQwenFaiHeadDim / 2], tmp[kQwenFaiHeadDim / 2],
                 kQwenFaiHeadDim / 2);                                     // rot_hi
    AscendC::PipeBarrier<PIPE_V>();
}

#endif  // __DAV_C220_VEC__

// Per (kv_head, batch): QK-norm + RoPE on Q (5 heads) and K (1 head) -> BF16 query
// buffer and paged K cache; V scaled by inv_rms -> paged V cache (no norm, no rope).
// On the cube (AIC) variant this is a no-op — RoPE is vector-only.
static __aicore__ void run_qwen_rope_qkv(__gm__ int64_t *args) {
#ifdef __DAV_C220_VEC__
    constexpr uint32_t kItems = kQwenBatch * kQwenNumKvHeads;  // 128

    AscendC::GlobalTensor<float> q_proj;
    AscendC::GlobalTensor<float> k_proj;
    AscendC::GlobalTensor<float> v_proj;
    AscendC::GlobalTensor<float> inv_rms;
    AscendC::GlobalTensor<float> q_norm;
    AscendC::GlobalTensor<float> k_norm;
    AscendC::GlobalTensor<int32_t> seq_lens;
    AscendC::GlobalTensor<int32_t> slot_mapping;
    AscendC::GlobalTensor<float> rope_cos;
    AscendC::GlobalTensor<float> rope_sin;
    AscendC::GlobalTensor<bfloat16_t> query;
    AscendC::GlobalTensor<bfloat16_t> key_cache;
    AscendC::GlobalTensor<bfloat16_t> value_cache;
    q_proj.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tensor_data<float>(args, kRopeQProjArg)));
    k_proj.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tensor_data<float>(args, kRopeKProjArg)));
    v_proj.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tensor_data<float>(args, kRopeVProjArg)));
    inv_rms.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tensor_data<float>(args, kRopeInvRmsArg)));
    q_norm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tensor_data<float>(args, kRopeQNormArg)));
    k_norm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tensor_data<float>(args, kRopeKNormArg)));
    seq_lens.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tensor_data<int32_t>(args, kRopeSeqLensArg)));
    slot_mapping.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tensor_data<int32_t>(args, kRopeSlotMappingArg)));
    rope_cos.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tensor_data<float>(args, kRopeCosArg)));
    rope_sin.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tensor_data<float>(args, kRopeSinArg)));
    query.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(tensor_data<uint16_t>(args, kRopeQueryArg)));
    key_cache.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(tensor_data<uint16_t>(args, kRopeKeyCacheArg)));
    value_cache.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(tensor_data<uint16_t>(args, kRopeValueCacheArg)));

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf;
    pipe.InitBuffer(buf, 8192);
    AscendC::LocalTensor<float> mem = buf.Get<float>();
    AscendC::LocalTensor<float> x = mem[0];
    AscendC::LocalTensor<float> square = mem[128];
    AscendC::LocalTensor<float> weight = mem[256];
    AscendC::LocalTensor<float> cos = mem[384];
    AscendC::LocalTensor<float> sin = mem[512];
    AscendC::LocalTensor<float> out = mem[640];
    AscendC::LocalTensor<float> tmp = mem[768];
    AscendC::LocalTensor<float> reduce = mem[896];
    AscendC::LocalTensor<float> reduce_work = mem[1024];
    AscendC::LocalTensor<bfloat16_t> out_bf16 = buf.Get<bfloat16_t>()[3072];

    uint64_t cache_row_offset = static_cast<uint64_t>(args[kRopeCacheRowOffsetArg]);
    uint32_t physical_lane =
        static_cast<uint32_t>(get_block_idx(args)) * 2 + static_cast<uint32_t>(get_sub_block_id(args));
    uint32_t physical_lanes = static_cast<uint32_t>(get_block_num(args)) * 2;

    for (uint32_t item = physical_lane; item < kItems; item += physical_lanes) {
        uint32_t kv_head = item / kQwenBatch;
        uint32_t batch = item - kv_head * kQwenBatch;
        uint32_t pos = static_cast<uint32_t>(seq_lens.GetValue(batch)) - 1;
        uint32_t slot = static_cast<uint32_t>(slot_mapping.GetValue(batch));
        float inv_rms_value = inv_rms.GetValue(batch);

        AscendC::DataCopy(cos, rope_cos[static_cast<uint64_t>(pos) * kQwenFaiHeadDim], kQwenFaiHeadDim);
        AscendC::DataCopy(sin, rope_sin[static_cast<uint64_t>(pos) * kQwenFaiHeadDim], kQwenFaiHeadDim);

        AscendC::DataCopy(weight, q_norm, kQwenFaiHeadDim);
        qwen_mte2_to_v();
        for (uint32_t q = 0; q < kQwenHeadsPerKv; ++q) {
            uint32_t head = kv_head * kQwenHeadsPerKv + q;
            AscendC::DataCopy(x, q_proj[batch * kQwenNumHeads * kQwenFaiHeadDim + head * kQwenFaiHeadDim],
                              kQwenFaiHeadDim);
            qwen_mte2_to_v();
            qwen_normalize_rope_row(x, weight, cos, sin, square, reduce, reduce_work, out, tmp, inv_rms_value);
            AscendC::Cast(out_bf16, out, AscendC::RoundMode::CAST_ROUND, kQwenFaiHeadDim);
            qwen_v_to_mte3();
            AscendC::DataCopy(query[batch * kQwenNumHeads * kQwenFaiHeadDim + head * kQwenFaiHeadDim], out_bf16,
                              kQwenFaiHeadDim);
            qwen_mte3_to_v();
            qwen_mte3_to_mte2();
        }

        AscendC::DataCopy(weight, k_norm, kQwenFaiHeadDim);
        AscendC::DataCopy(x, k_proj[batch * kQwenNumKvHeads * kQwenFaiHeadDim + kv_head * kQwenFaiHeadDim],
                          kQwenFaiHeadDim);
        qwen_mte2_to_v();
        qwen_normalize_rope_row(x, weight, cos, sin, square, reduce, reduce_work, out, tmp, inv_rms_value);
        AscendC::Cast(out_bf16, out, AscendC::RoundMode::CAST_ROUND, kQwenFaiHeadDim);
        qwen_v_to_mte3();
        uint64_t cache_row = cache_row_offset + static_cast<uint64_t>(slot) * kQwenNumKvHeads + kv_head;
        AscendC::DataCopy(key_cache[cache_row * kQwenFaiHeadDim], out_bf16, kQwenFaiHeadDim);
        qwen_mte3_to_v();
        qwen_mte3_to_mte2();

        AscendC::DataCopy(x, v_proj[batch * kQwenNumKvHeads * kQwenFaiHeadDim + kv_head * kQwenFaiHeadDim],
                          kQwenFaiHeadDim);
        qwen_mte2_to_v();
        AscendC::Muls(x, x, inv_rms_value, kQwenFaiHeadDim);
        AscendC::PipeBarrier<PIPE_V>();  // Cast must read the inv_rms-scaled x
        AscendC::Cast(out_bf16, x, AscendC::RoundMode::CAST_ROUND, kQwenFaiHeadDim);
        qwen_v_to_mte3();
        AscendC::DataCopy(value_cache[cache_row * kQwenFaiHeadDim], out_bf16, kQwenFaiHeadDim);
        qwen_mte3_to_v();
        qwen_mte3_to_mte2();
    }
    pipe.Destroy();
#else
    (void)args;
#endif
}

#endif  // QWEN_FAI_ROPE_ONLY_ABI

#endif
