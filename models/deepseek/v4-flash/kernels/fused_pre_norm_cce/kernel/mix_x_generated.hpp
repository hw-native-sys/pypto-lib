/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0.
 */

#ifndef PYPTO_DEEPSEEK_FUSED_PRE_NORM_MIX_GENERATED_HPP
#define PYPTO_DEEPSEEK_FUSED_PRE_NORM_MIX_GENERATED_HPP

#include <cstdint>

#ifdef __DAV_C220_VEC__
#include <pto/pto-inst.hpp>
#include "intrinsic.h"
#include "tensor.h"

namespace deepseek_fused_pre_norm_mix_generated {
using namespace pto;

// PyPTO/PTOAS generated mix_x body. Keep the code below verbatim; regenerate
// it from the hc_pre mix_x scope when that scope changes.
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

static __aicore__ void mix_x(__gm__ float* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, int64_t v4, int64_t v5, int64_t v6, int32_t v7, int32_t v8) {
  SaturationMode v9 = SaturationMode::OFF;
  RoundMode v10 = RoundMode::CAST_RINT;
  const int64_t v11 = 12288;
  const int64_t v12 = 256;
  const int64_t v13 = 2;
  const int64_t v14 = 1024;
  const int64_t v15 = 4;
  const int64_t v16 = 4096;
  const int64_t v17 = 1;
  const int64_t v18 = 8;
  const int64_t v19 = 24576;
  const int64_t v20 = 16384;
  const int64_t v21 = 8192;
  const int64_t v22 = 0;
  const int64_t v23 = 57600;
  const int64_t v24 = 49408;
  const int64_t v25 = 32864;
  const int64_t v26 = 32832;
  const int64_t v27 = 32800;
  const int64_t v28 = 32768;
  const int64_t v29 = 41216;
  const int64_t v30 = 33024;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v31 = (int64_t) v7;
  int64_t v32 = (int64_t) ((uint64_t) (v31 / v15) * (uint64_t) v18);
  int64_t v33 = (int64_t) ((uint64_t) (v31 % v15) * (uint64_t) v14);
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v34 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v18);
  uint64_t v35 = (uint64_t) v30;
  TASSIGN(v34, v35);
  pto::Shape<1, 1, 1, 8, 8> v36 = pto::Shape<1, 1, 1, 8, 8>();
  pto::Stride<64, 64, 64, 8, 1> v37 = pto::Stride<64, 64, 64, 8, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 8, 8>, pto::Stride<64, 64, 64, 8, 1>, pto::Layout::ND> v38 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 8>, pto::Stride<64, 64, 64, 8, 1>, pto::Layout::ND>(v1 + (v22 + v32 * v18 + v22 * v17), v36, v37);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  TLOAD(v34, v38);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v39 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v18);
  uint64_t v40 = (uint64_t) v29;
  TASSIGN(v39, v40);
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v41 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v18);
  uint64_t v42 = (uint64_t) v28;
  TASSIGN(v41, v42);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TTRANS(v41, v34, v39);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v43 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v17);
  uint64_t v44 = (uint64_t) v28;
  TASSIGN(v43, v44);
  Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v45 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v17);
  uint64_t v46 = (uint64_t) v27;
  TASSIGN(v45, v46);
  Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v47 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v17);
  uint64_t v48 = (uint64_t) v26;
  TASSIGN(v47, v48);
  Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v49 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v17);
  uint64_t v50 = (uint64_t) v25;
  TASSIGN(v49, v50);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  for (size_t v51 = (size_t) v22; v51 < ((size_t) v15); v51 += (size_t) v13) {
    int64_t v52 = (int64_t) ((uint64_t) ((int64_t) v51) * (uint64_t) v12);
    int64_t v53 = (int64_t) ((uint64_t) v33 + (uint64_t) v52);
    int64_t v54 = (int64_t) ((uint64_t) v33 + (uint64_t) ((int64_t) (uint64_t) v52 + (uint64_t) v12));
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v55 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v12);
    uint64_t v56 = (uint64_t) v30;
    TASSIGN(v55, v56);
    pto::Shape<1, 1, 1, 8, 256> v57 = pto::Shape<1, 1, 1, 8, 256>();
    pto::Stride<131072, 131072, 131072, 16384, 1> v58 = pto::Stride<131072, 131072, 131072, 16384, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND> v59 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>(v3 + (v22 + v32 * v20 + v53 * v17), v57, v58);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v55, v59);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v60 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v12);
    uint64_t v61 = (uint64_t) v29;
    TASSIGN(v60, v61);
    pto::Shape<1, 1, 1, 8, 256> v62 = pto::Shape<1, 1, 1, 8, 256>();
    pto::Stride<131072, 131072, 131072, 16384, 1> v63 = pto::Stride<131072, 131072, 131072, 16384, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND> v64 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>(v3 + (v22 + v32 * v20 + (int64_t) ((uint64_t) v53 + (uint64_t) v16) * v17), v62, v63);
    TLOAD(v60, v64);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v65 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v12);
    uint64_t v66 = (uint64_t) v24;
    TASSIGN(v65, v66);
    pto::Shape<1, 1, 1, 8, 256> v67 = pto::Shape<1, 1, 1, 8, 256>();
    pto::Stride<131072, 131072, 131072, 16384, 1> v68 = pto::Stride<131072, 131072, 131072, 16384, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND> v69 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>(v3 + (v22 + v32 * v20 + (int64_t) ((uint64_t) v53 + (uint64_t) v21) * v17), v67, v68);
    TLOAD(v65, v69);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v70 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v12);
    uint64_t v71 = (uint64_t) v23;
    TASSIGN(v70, v71);
    pto::Shape<1, 1, 1, 8, 256> v72 = pto::Shape<1, 1, 1, 8, 256>();
    pto::Stride<131072, 131072, 131072, 16384, 1> v73 = pto::Stride<131072, 131072, 131072, 16384, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND> v74 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>(v3 + (v22 + v32 * v20 + (int64_t) ((uint64_t) v53 + (uint64_t) v11) * v17), v72, v73);
    TLOAD(v70, v74);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v75 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v12);
    uint64_t v76 = (uint64_t) v22;
    TASSIGN(v75, v76);
    pto::Shape<1, 1, 1, 8, 256> v77 = pto::Shape<1, 1, 1, 8, 256>();
    pto::Stride<131072, 131072, 131072, 16384, 1> v78 = pto::Stride<131072, 131072, 131072, 16384, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND> v79 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>(v3 + (v22 + v32 * v20 + v54 * v17), v77, v78);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v75, v79);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v80 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v12);
    uint64_t v81 = (uint64_t) v21;
    TASSIGN(v80, v81);
    pto::Shape<1, 1, 1, 8, 256> v82 = pto::Shape<1, 1, 1, 8, 256>();
    pto::Stride<131072, 131072, 131072, 16384, 1> v83 = pto::Stride<131072, 131072, 131072, 16384, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND> v84 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>(v3 + (v22 + v32 * v20 + (int64_t) ((uint64_t) v54 + (uint64_t) v16) * v17), v82, v83);
    TLOAD(v80, v84);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v85 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v12);
    uint64_t v86 = (uint64_t) v20;
    TASSIGN(v85, v86);
    pto::Shape<1, 1, 1, 8, 256> v87 = pto::Shape<1, 1, 1, 8, 256>();
    pto::Stride<131072, 131072, 131072, 16384, 1> v88 = pto::Stride<131072, 131072, 131072, 16384, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND> v89 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>(v3 + (v22 + v32 * v20 + (int64_t) ((uint64_t) v54 + (uint64_t) v21) * v17), v87, v88);
    TLOAD(v85, v89);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v90 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v12);
    uint64_t v91 = (uint64_t) v19;
    TASSIGN(v90, v91);
    pto::Shape<1, 1, 1, 8, 256> v92 = pto::Shape<1, 1, 1, 8, 256>();
    pto::Stride<131072, 131072, 131072, 16384, 1> v93 = pto::Stride<131072, 131072, 131072, 16384, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND> v94 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>(v3 + (v22 + v32 * v20 + (int64_t) ((uint64_t) v54 + (uint64_t) v11) * v17), v92, v93);
    TLOAD(v90, v94);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v95 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v12);
    uint64_t v96 = (uint64_t) v30;
    TASSIGN(v95, v96);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TROWEXPANDMUL(v95, v55, v43);
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v97 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v12);
    uint64_t v98 = (uint64_t) v29;
    TASSIGN(v97, v98);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    TROWEXPANDMUL(v97, v60, v45);
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v99 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v12);
    uint64_t v100 = (uint64_t) v24;
    TASSIGN(v99, v100);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    TROWEXPANDMUL(v99, v65, v47);
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v101 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v12);
    uint64_t v102 = (uint64_t) v23;
    TASSIGN(v101, v102);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    TROWEXPANDMUL(v101, v70, v49);
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v103 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v12);
    uint64_t v104 = (uint64_t) v30;
    TASSIGN(v103, v104);
    pipe_barrier(PIPE_V);
    TADD(v103, v95, v97);
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v105 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v12);
    uint64_t v106 = (uint64_t) v29;
    TASSIGN(v105, v106);
    pipe_barrier(PIPE_V);
    TADD(v105, v99, v101);
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v107 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v12);
    uint64_t v108 = (uint64_t) v30;
    TASSIGN(v107, v108);
    pipe_barrier(PIPE_V);
    TADD(v107, v103, v105);
    Tile<TileType::Vec, bfloat16_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v109 = Tile<TileType::Vec, bfloat16_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v12);
    uint64_t v110 = (uint64_t) v30;
    TASSIGN(v109, v110);
    pipe_barrier(PIPE_V);
    TCVT(v109, v107, v10, v9);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pto::Shape<1, 1, 1, 8, 256> v111 = pto::Shape<1, 1, 1, 8, 256>();
    pto::Stride<32768, 32768, 32768, 4096, 1> v112 = pto::Stride<32768, 32768, 32768, 4096, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND> v113 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>(v2 + (v22 + v32 * v16 + v53 * v17), v111, v112);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v113, v109);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v114 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v12);
    uint64_t v115 = (uint64_t) v22;
    TASSIGN(v114, v115);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    TROWEXPANDMUL(v114, v75, v43);
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v116 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v12);
    uint64_t v117 = (uint64_t) v21;
    TASSIGN(v116, v117);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
    TROWEXPANDMUL(v116, v80, v45);
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v118 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v12);
    uint64_t v119 = (uint64_t) v20;
    TASSIGN(v118, v119);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    TROWEXPANDMUL(v118, v85, v47);
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v120 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v12);
    uint64_t v121 = (uint64_t) v19;
    TASSIGN(v120, v121);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TROWEXPANDMUL(v120, v90, v49);
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v122 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v12);
    uint64_t v123 = (uint64_t) v22;
    TASSIGN(v122, v123);
    pipe_barrier(PIPE_V);
    TADD(v122, v114, v116);
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v124 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v12);
    uint64_t v125 = (uint64_t) v21;
    TASSIGN(v124, v125);
    pipe_barrier(PIPE_V);
    TADD(v124, v118, v120);
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v126 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v12);
    uint64_t v127 = (uint64_t) v22;
    TASSIGN(v126, v127);
    pipe_barrier(PIPE_V);
    TADD(v126, v122, v124);
    Tile<TileType::Vec, bfloat16_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v128 = Tile<TileType::Vec, bfloat16_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v12);
    uint64_t v129 = (uint64_t) v22;
    TASSIGN(v128, v129);
    pipe_barrier(PIPE_V);
    TCVT(v128, v126, v10, v9);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    pto::Shape<1, 1, 1, 8, 256> v130 = pto::Shape<1, 1, 1, 8, 256>();
    pto::Stride<32768, 32768, 32768, 4096, 1> v131 = pto::Stride<32768, 32768, 32768, 4096, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND> v132 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>(v2 + (v22 + v32 * v16 + v54 * v17), v130, v131);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v132, v128);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}

}  // namespace deepseek_fused_pre_norm_mix_generated
#endif  // __DAV_C220_VEC__

#endif  // PYPTO_DEEPSEEK_FUSED_PRE_NORM_MIX_GENERATED_HPP
