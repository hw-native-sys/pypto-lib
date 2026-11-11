/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
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

#include "intrinsic.h"
#include "runtime_tensor_compat.hpp"

#include "../generated/kernel_tiling/kernel_tiling.h"
#include "metadata_layout.h"

#include "../vendor/fused_infer_attention_score/flash_attention_regular.h"

#include "rope_qkv_generated.hpp"

constexpr uint64_t kQwenFaiHeadDim = 128;
// The generated RoPE body is specialized to the standalone 32-lane dispatch.
constexpr uint32_t kQwenRopeCores = 32;

// Global cube<->vector barrier between phase-0 RoPE and the attention phase.
// The FFTS flag-region base is set by the simpler runtime at launch.
// AscendC::SyncAll<false> is the fused (mixed AIC+AIV) all-core barrier; the
// default SyncAll() is AIV-only and never releases the Cube cores.
static __aicore__ __attribute__((always_inline)) void qwen_fai_syncall_mix() {
  AscendC::PipeBarrier<PIPE_ALL>();
  // isAIVOnly=false: fused Cube+Vector whole-core barrier.
  AscendC::SyncAll<false>();
}

static __aicore__ __attribute__((always_inline)) void
acquire_qwen_fai_metadata(GM_ADDR metadata) {
  uint64_t first_line =
      reinterpret_cast<uint64_t>(metadata) &
      ~(static_cast<uint64_t>(qwen_fai_metadata::kDcciLineBytes) - 1);
  uint64_t end = reinterpret_cast<uint64_t>(metadata) +
                 qwen_fai_metadata::kBarrierAlignmentOffset;
  for (uint64_t line = first_line; line < end;
       line += qwen_fai_metadata::kDcciLineBytes) {
    dcci(reinterpret_cast<__gm__ void *>(line), SINGLE_CACHE_LINE);
  }
  dsb(DSB_DDR);
}

template <typename T>
static __aicore__ __attribute__((always_inline)) GM_ADDR
tensor_data(__gm__ int64_t *args, int32_t index) {
  __gm__ PyPTORuntimeTensor *tensor =
      reinterpret_cast<__gm__ PyPTORuntimeTensor *>(args[index]);
  __gm__ T *data =
      reinterpret_cast<__gm__ T *>(tensor->buffer.addr) + tensor->start_offset;
  return reinterpret_cast<GM_ADDR>(data);
}

template <bool IsFlashDecode, bool WithRope = false>
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
  using BlockMmadQK =
      Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK,
                             QType, KType, SType>;

  using PType = Gemm::GemmType<ElementP, LayoutP>;
  using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
  using PseShiftType = Gemm::GemmType<ElementQ, LayoutQ>;
  using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmax<
      Epilogue::LseMode::NONE, Epilogue::SinkMode::DISABLE,
      static_cast<Epilogue::MaskMode>(FaiKernel::MaskType::NO_MASK), float>;
  using EpilogueOnlineSoftmax =
      Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType,
                                     MaskType, SinkType, PseShiftType>;

  using L1TileShapePV = GemmShape<128, 128, 256>;
  using L0TileShapePV = GemmShape<128, 128, 128>;
  using DispatchPolicyPV = Gemm::MmadAtlasA2FAIPV<true, false>;
  using VType = Gemm::GemmType<ElementV, LayoutV>;
  using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
  using BlockMmadPV =
      Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShapePV, L0TileShapePV,
                             PType, VType, OTmpType>;

  using OType = Gemm::GemmType<ElementO, LayoutO>;
  using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
  using LseType = Gemm::GemmType<ElementLse, LayoutLse>;
  using DispatchPolicyRescaleO =
      Epilogue::EpilogueAtlasA2RescaleO<Epilogue::LseMode::NONE, float>;
  using EpilogueRescaleO =
      Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType,
                                     OUpdateType, LseType>;
  using DispatchPolicyInitOut =
      Epilogue::EpilogueAtlasA2InitOutWhenZero<Epilogue::LseMode::NONE>;
  using EpilogueInitOut =
      Epilogue::Block::BlockEpilogue<DispatchPolicyInitOut, OType, LseType>;
  using CombineScale = Epilogue::Block::CombineScale<OType, LseType>;

  using FdKernel = SplitFuse::FAInferKernel<
      BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO,
      EpilogueInitOut, true, FaiKernel::MaskType::NO_MASK,
      FaiKernel::inputLayout::TND, CombineScale, true, true, true>;
  using NonFdKernel =
      SplitFuse::FAInferKernel<BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax,
                               EpilogueRescaleO, EpilogueInitOut, true,
                               FaiKernel::MaskType::NO_MASK,
                               FaiKernel::inputLayout::TND>;
  using Kernel = std::conditional_t<IsFlashDecode, FdKernel, NonFdKernel>;

  GM_ADDR metadata = tensor_data<uint8_t>(args, 6);
  __gm__ const FAInferTilingData *tiling =
      reinterpret_cast<__gm__ const FAInferTilingData *>(
          metadata + qwen_fai_metadata::kTilingOffset);
  // pypto packs tensors first, then the scalars last: the rope-fused ABI has 17
  // tensors so cache_row_offset is at args[17] and batch_offset at args[18]; the
  // attention-only ABI has 7 tensors so cache_row_offset is at args[7].
  uint64_t cache_row_offset =
      static_cast<uint64_t>(WithRope ? args[17] : args[7]);
  uint64_t cache_byte_offset =
      cache_row_offset * kQwenFaiHeadDim * sizeof(uint16_t);
  // First public batch row this call serves. decode_fwd runs a batch above
  // kMaxBatch as consecutive row chunks: the per-row inputs it slices itself
  // (q/k/v_proj, q_tnd, out) are already chunk-local, but seq_lens, slot_mapping
  // and the block table are whole-batch, so they are indexed from this row. The
  // tiling was built for the same window, hence its 0-based batch indices. Zero
  // for the attention-only ABI, which has no chunking caller. (Reading `tiling`
  // is safe on both: each entry.cpp invalidates the metadata lines first.)
  uint64_t batch_offset = WithRope ? static_cast<uint64_t>(args[18]) : 0;
  constexpr int32_t query_arg = WithRope ? 1 : 0;
  constexpr int32_t key_arg = WithRope ? 2 : 1;
  constexpr int32_t value_arg = WithRope ? 3 : 2;
  constexpr int32_t block_table_arg = WithRope ? 4 : 3;
  constexpr int32_t out_arg = WithRope ? 0 : 4;
  GM_ADDR key = tensor_data<uint16_t>(args, key_arg) + cache_byte_offset;
  GM_ADDR value = tensor_data<uint16_t>(args, value_arg) + cache_byte_offset;
  // BYTES. GM_ADDR is a byte pointer, so every displacement applied to one is a
  // byte count -- unlike the typed `__gm__ int32_t *` arithmetic in the RoPE
  // call below, which advances ELEMENTS. The two units are one cast apart, so
  // name the byte quantities rather than inlining the sizeof().
  uint64_t block_table_byte_offset =
      batch_offset * tiling->maxNumBlocksPerBatch * sizeof(int32_t);
  GM_ADDR block_table =
      tensor_data<int32_t>(args, block_table_arg) + block_table_byte_offset;

  FAIKernelParams params{tensor_data<uint16_t>(args, query_arg),
                         key,
                         value,
                         nullptr,
                         nullptr,
                         block_table,
                         metadata + qwen_fai_metadata::kCumulativeQOffset,
                         metadata + qwen_fai_metadata::kKvLengthsOffset,
                         tensor_data<uint16_t>(args, out_arg),
                         nullptr,
                         tensor_data<uint8_t>(args, 5),
                         metadata + qwen_fai_metadata::kTilingOffset,
                         nullptr};

  uint32_t sub_block_idx = 0;
#ifdef __DAV_C220_VEC__
  sub_block_idx = static_cast<uint32_t>(get_sub_block_id(args));
#endif
  uint32_t block_idx = static_cast<uint32_t>(get_block_idx(args));
  uint32_t block_num = static_cast<uint32_t>(get_block_num(args));

  // Fold QK-norm + RoPE in as phase 0: the AIV lanes rotate Q/K and publish
  // paged K plus projected V, then a global cube<->vec FFTS barrier makes those
  // GM writes visible to every core before the attention phase reads them.
  if constexpr (WithRope) {
#ifdef __DAV_C220_VEC__
    // Guard the hand-written mapping below against ABI drift in the generated
    // body. Regenerating rope_qkv_regen.py with a different dynamic-dim or
    // captured-scalar set changes this signature's ARITY, which this catches at
    // compile time. It cannot catch two same-typed int64 parameters being
    // swapped, so keep the order comment below in sync with the emitted
    // launcher's forwarding call.
    static_assert(
        __is_same(
            decltype(&qwen_rope_gen::rope_qkv),
            void (*)(__gm__ bfloat16_t *, __gm__ bfloat16_t *, __gm__ bfloat16_t *,
                     __gm__ int32_t *, __gm__ float *, __gm__ int32_t *, __gm__ float *,
                     __gm__ float *, __gm__ float *, __gm__ float *, __gm__ float *,
                     __gm__ float *, __gm__ float *, int64_t, int64_t, int64_t,
                     int32_t, int32_t)),
        "rope_qkv signature drifted from the arg mapping below -- re-derive it "
        "from the launcher's forwarding call in rope_qkv_generated.hpp's preamble");
    // Drive the golden-correct pypto-generated rope_qkv. The fused ABI packs 17
    // tensors then the sole scalar; map them to the generated parameter order,
    // which is reproduced verbatim in rope_qkv_generated.hpp's preamble:
    //   0-12 k_cache, q_tnd, v_cache, seq_lens, inv_rms, slot_mapping,
    //        rope_cos, rope_sin, k_proj, k_norm_w, v_proj, q_proj, q_norm_w
    //   13   unused captured loop scalar (dead in the body -- see below)
    //   14   layer_cache_base
    //   15   batch          <- rows this call serves, from the tiling
    //   16-17 block_idx, block_num
    //
    // The batch is read back from the TILING the tiler just emitted, so the RoPE
    // phase and the attention phase can never disagree about how many sequences
    // are live -- including when decode_fwd chunks a public batch above
    // kMaxBatch and this call serves only a window of it.
    int64_t rope_batch = static_cast<int64_t>(tiling->batch);
    // Parameter 13 is a loop-induction scalar the pypto scope captured while
    // making layer_cache_base an opaque runtime value; it is provably dead in
    // the generated body (its only occurrence is the signature), so any value
    // works. Kept explicit rather than removed: dropping it would desynchronize
    // this call from the emitted signature.
    constexpr int64_t kUnusedCapturedScalar = 0;
    uint32_t rope_lane = block_idx * 2 + sub_block_idx;
    if (rope_lane < kQwenRopeCores) {
      qwen_rope_gen::rope_qkv(
          reinterpret_cast<__gm__ bfloat16_t *>(tensor_data<uint16_t>(args, 2)),
          reinterpret_cast<__gm__ bfloat16_t *>(tensor_data<uint16_t>(args, 1)),
          reinterpret_cast<__gm__ bfloat16_t *>(tensor_data<uint16_t>(args, 3)),
          // seq_lens and slot_mapping are WHOLE-BATCH, so they start at the
          // window's first row; inv_rms_states and the projections below are
          // chunk-local buffers the caller already sliced, so they start at 0.
          // ELEMENTS: these are typed int32 pointers holding one entry per row,
          // so `+ batch_offset` advances rows directly -- no sizeof(), unlike
          // the byte arithmetic on the GM_ADDR values above.
          reinterpret_cast<__gm__ int32_t *>(tensor_data<int32_t>(args, 16)) +
              batch_offset,
          reinterpret_cast<__gm__ float *>(tensor_data<float>(args, 14)),
          reinterpret_cast<__gm__ int32_t *>(tensor_data<int32_t>(args, 15)) +
              batch_offset,
          reinterpret_cast<__gm__ float *>(tensor_data<float>(args, 12)),
          reinterpret_cast<__gm__ float *>(tensor_data<float>(args, 13)),
          reinterpret_cast<__gm__ float *>(tensor_data<float>(args, 8)),
          reinterpret_cast<__gm__ float *>(tensor_data<float>(args, 11)),
          reinterpret_cast<__gm__ float *>(tensor_data<float>(args, 9)),
          reinterpret_cast<__gm__ float *>(tensor_data<float>(args, 7)),
          reinterpret_cast<__gm__ float *>(tensor_data<float>(args, 10)),
          kUnusedCapturedScalar, static_cast<int64_t>(args[17]), rope_batch,
          static_cast<int32_t>(rope_lane),
          static_cast<int32_t>(kQwenRopeCores));
    }
#endif
    qwen_fai_syncall_mix();
  }

  Arch::PtoTopology topology{block_idx, block_num, sub_block_idx, 2};
  Kernel kernel;
  kernel(params, topology, barrier_state);
}

#endif
