/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0.
 */

#ifndef PYPTO_DEEPSEEK_FUSED_PRE_NORM_SPLIT_GENERATED_HPP
#define PYPTO_DEEPSEEK_FUSED_PRE_NORM_SPLIT_GENERATED_HPP

#include <cstdint>

#ifdef __DAV_C220_VEC__
#include <pto/pto-inst.hpp>
#include "intrinsic.h"
#include "tensor.h"

namespace deepseek_fused_pre_norm_split_generated {
using namespace pto;

// PyPTO/PTOAS generated split_pre_post body. Keep the code below verbatim;
// regenerate it from the hc_pre split_pre_post scope when that scope changes.
enum class PTOAutoSyncTailMode : int {
  kBarrierAll = 0,
  kSetWaitMte3ToSEvent0 = 1,
};

static __aicore__ inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
  switch (mode) {
  case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    break;
  case PTOAutoSyncTailMode::kBarrierAll:
  default:
    pipe_barrier(PIPE_ALL);
    break;
  }
}

static __aicore__ void split_pre_post(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ float* v4, __gm__ float* v5, float v6, float v7, int64_t v8, int64_t v9, int32_t v10, int32_t v11) {
  const float v12 = 2.0f;
  const float v13 = 9.99999997E-7f;
  const float v14 = 1.0f;
  const int64_t v15 = 4;
  const int64_t v16 = 8;
  const int64_t v17 = 32;
  const int64_t v18 = 1;
  const int64_t v19 = 0;
  const int64_t v20 = 320;
  const int64_t v21 = 288;
  const int64_t v22 = 256;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v23 = (int64_t) ((uint64_t) ((int64_t) v10) * (uint64_t) v16);
  Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v24 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v18);
  uint64_t v25 = (uint64_t) v22;
  TASSIGN(v24, v25);
  pto::Shape<1, 1, 1, 8, 1> v26 = pto::Shape<1, 1, 1, 8, 1>();
  pto::Stride<8, 8, 8, 1, -1> v27 = pto::Stride<8, 8, 8, 1, -1>(v8);
  GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<8, 8, 8, 1, -1>, pto::Layout::DN> v28 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<8, 8, 8, 1, -1>, pto::Layout::DN>(v1 + (v19 + v23 * v18 + v19 * v8), v26, v27);
  TLOAD(v24, v28);
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v29 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v16);
  uint64_t v30 = (uint64_t) v21;
  TASSIGN(v29, v30);
  pto::Shape<1, 1, 1, 1, 8> v31 = pto::Shape<1, 1, 1, 1, 8>();
  pto::Stride<8, 8, 8, 8, 1> v32 = pto::Stride<8, 8, 8, 8, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<8, 8, 8, 8, 1>, pto::Layout::ND> v33 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<8, 8, 8, 8, 1>, pto::Layout::ND>(v2 + (v19 + v19 * v18), v31, v32);
  TLOAD(v29, v33);
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v34 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v16);
  uint64_t v35 = (uint64_t) v21;
  TASSIGN(v34, v35);
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v36 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  uint64_t v37 = (uint64_t) v20;
  TASSIGN(v36, v37);
  pto::Shape<1, 1, 1, 8, 8> v38 = pto::Shape<1, 1, 1, 8, 8>();
  pto::Stride<256, 256, 256, 32, 1> v39 = pto::Stride<256, 256, 256, 32, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 8, 8>, pto::Stride<256, 256, 256, 32, 1>, pto::Layout::ND> v40 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 8>, pto::Stride<256, 256, 256, 32, 1>, pto::Layout::ND>(v3 + (v19 + v23 * v17 + v19 * v18), v38, v39);
  TLOAD(v36, v40);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v41 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  uint64_t v42 = (uint64_t) v20;
  TASSIGN(v41, v42);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TROWEXPANDMUL(v41, v36, v24);
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v43 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  uint64_t v44 = (uint64_t) v20;
  TASSIGN(v43, v44);
  pipe_barrier(PIPE_V);
  TMULS(v43, v41, v6);
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v45 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  uint64_t v46 = (uint64_t) v19;
  TASSIGN(v45, v46);
  TCOLEXPAND(v45, v34);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v47 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  uint64_t v48 = (uint64_t) v20;
  TASSIGN(v47, v48);
  pipe_barrier(PIPE_V);
  TADD(v47, v43, v45);
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v49 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  uint64_t v50 = (uint64_t) v20;
  TASSIGN(v49, v50);
  pipe_barrier(PIPE_V);
  TNEG(v49, v47);
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v51 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  uint64_t v52 = (uint64_t) v20;
  TASSIGN(v51, v52);
  pipe_barrier(PIPE_V);
  TEXP(v51, v49);
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v53 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  uint64_t v54 = (uint64_t) v20;
  TASSIGN(v53, v54);
  pipe_barrier(PIPE_V);
  TADDS(v53, v51, v14);
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v55 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  uint64_t v56 = (uint64_t) v19;
  TASSIGN(v55, v56);
  pipe_barrier(PIPE_V);
  TRECIP(v55, v53);
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v57 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  uint64_t v58 = (uint64_t) v20;
  TASSIGN(v57, v58);
  pipe_barrier(PIPE_V);
  TADDS(v57, v55, v13);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  pto::Shape<1, 1, 1, 8, 8> v59 = pto::Shape<1, 1, 1, 8, 8>();
  pto::Stride<64, 64, 64, 8, 1> v60 = pto::Stride<64, 64, 64, 8, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 8, 8>, pto::Stride<64, 64, 64, 8, 1>, pto::Layout::ND> v61 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 8>, pto::Stride<64, 64, 64, 8, 1>, pto::Layout::ND>(v4 + (v19 + v23 * v16 + v19 * v18), v59, v60);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v61, v57);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v62 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v16);
  uint64_t v63 = (uint64_t) v21;
  TASSIGN(v62, v63);
  pto::Shape<1, 1, 1, 1, 8> v64 = pto::Shape<1, 1, 1, 1, 8>();
  pto::Stride<8, 8, 8, 8, 1> v65 = pto::Stride<8, 8, 8, 8, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<8, 8, 8, 8, 1>, pto::Layout::ND> v66 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<8, 8, 8, 8, 1>, pto::Layout::ND>(v2 + (v19 + v15 * v18), v64, v65);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  TLOAD(v62, v66);
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v67 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v16);
  uint64_t v68 = (uint64_t) v21;
  TASSIGN(v67, v68);
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v69 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  uint64_t v70 = (uint64_t) v20;
  TASSIGN(v69, v70);
  pto::Shape<1, 1, 1, 8, 8> v71 = pto::Shape<1, 1, 1, 8, 8>();
  pto::Stride<256, 256, 256, 32, 1> v72 = pto::Stride<256, 256, 256, 32, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 8, 8>, pto::Stride<256, 256, 256, 32, 1>, pto::Layout::ND> v73 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 8>, pto::Stride<256, 256, 256, 32, 1>, pto::Layout::ND>(v3 + (v19 + v23 * v17 + v15 * v18), v71, v72);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  TLOAD(v69, v73);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v74 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  uint64_t v75 = (uint64_t) v20;
  TASSIGN(v74, v75);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
  TROWEXPANDMUL(v74, v69, v24);
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v76 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  uint64_t v77 = (uint64_t) v20;
  TASSIGN(v76, v77);
  pipe_barrier(PIPE_V);
  TMULS(v76, v74, v7);
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v78 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  uint64_t v79 = (uint64_t) v19;
  TASSIGN(v78, v79);
  TCOLEXPAND(v78, v67);
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v80 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  uint64_t v81 = (uint64_t) v20;
  TASSIGN(v80, v81);
  pipe_barrier(PIPE_V);
  TADD(v80, v76, v78);
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v82 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  uint64_t v83 = (uint64_t) v20;
  TASSIGN(v82, v83);
  pipe_barrier(PIPE_V);
  TNEG(v82, v80);
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v84 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  uint64_t v85 = (uint64_t) v20;
  TASSIGN(v84, v85);
  pipe_barrier(PIPE_V);
  TEXP(v84, v82);
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v86 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  uint64_t v87 = (uint64_t) v20;
  TASSIGN(v86, v87);
  pipe_barrier(PIPE_V);
  TADDS(v86, v84, v14);
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v88 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  uint64_t v89 = (uint64_t) v19;
  TASSIGN(v88, v89);
  pipe_barrier(PIPE_V);
  TRECIP(v88, v86);
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v90 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  uint64_t v91 = (uint64_t) v20;
  TASSIGN(v90, v91);
  pipe_barrier(PIPE_V);
  TMULS(v90, v88, v12);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, 8, 4, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v92;
  uint64_t v93 = (uint64_t) v20;
  TASSIGN(v92, v93);
  pto::Shape<1, 1, 1, 8, 4> v94 = pto::Shape<1, 1, 1, 8, 4>();
  pto::Stride<32, 32, 32, 4, 1> v95 = pto::Stride<32, 32, 32, 4, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 8, 4>, pto::Stride<32, 32, 32, 4, 1>, pto::Layout::ND> v96 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 4>, pto::Stride<32, 32, 32, 4, 1>, pto::Layout::ND>(v5 + (v19 + v23 * v15 + v19 * v18), v94, v95);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  TSTORE(v96, v92);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}

}  // namespace deepseek_fused_pre_norm_split_generated
#endif  // __DAV_C220_VEC__

#endif  // PYPTO_DEEPSEEK_FUSED_PRE_NORM_SPLIT_GENERATED_HPP
