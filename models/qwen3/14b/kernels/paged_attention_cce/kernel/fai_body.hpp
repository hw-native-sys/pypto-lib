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

#endif
