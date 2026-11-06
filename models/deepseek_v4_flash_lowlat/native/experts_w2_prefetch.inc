// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------


// Reviewed Cube body inserted into the actual generated AIC function.

  #if defined(__DAV_CUBE__)
  const int32_t v17 = 0;
  // pto: %c0_i64
  const int64_t v18 = 0;
  // pto: %c65536_i64
  const int64_t v19 = 65536;
  // pto: %c73728_i64
  const int64_t v20 = 73728;
  // pto: %c204800_i64
  const int64_t v21 = 204800;
  // pto: %c212992_i64
  const int64_t v22 = 212992;
  // pto: %c2048_i64
  const int64_t v23 = 2048;
  // pto: %c32768_i64
  const int64_t v24 = 32768;
  // pto: %c4096_i64
  const int64_t v25 = 4096;
  // pto: %c6144_i64
  const int64_t v26 = 6144;
  // pto: %c592_index
  const int64_t v27 = 592;
  // pto: %c4096_index
  const int64_t v28 = 4096;
  // pto: %c1_index
  const int64_t v29 = 1;
  // pto: %c37_index
  const int64_t v30 = 37;
  // pto: %c8_index
  const int64_t v31 = 8;
  // pto: %c65792_index
  const int64_t v32 = 65792;
  // pto: %c256_index
  const int64_t v33 = 256;
  // pto: %c16_index
  const int64_t v34 = 16;
  // pto: %c1052672_index
  const int64_t v35 = 1052672;
  // pto: %c24_index
  const int64_t v36 = 24;
  // pto: %c0_index
  const int64_t v37 = 0;
  // pto: %c15_index
  const int64_t v38 = 15;
  // pto: %c2_index
  const int64_t v39 = 2;
  // pto: %c512_index
  const int64_t v40 = 512;
  // pto: %c128_index
  const int64_t v41 = 128;
  // pto: %cn128_index
  const int64_t v42 = -128;
  // pto: %cn1_index
  const int64_t v43 = -1;
  // pto: %c64_index
  const int64_t v44 = 64;
  // pto: %cn64_index
  const int64_t v45 = -64;
  // pto: %recv_x_flat_inline622__ssa_v0_view
  const int64_t v46 = 1;
  // pto: %recv_x_flat_inline622__ssa_v0_view
  const int64_t v47 = 1;
  // pto: %recv_x_flat_inline622__ssa_v0_view
  const int64_t v48 = 1;
  // pto: %recv_x_flat_inline622__ssa_v0_view
  int64_t v49 = v27 * v28;
  // pto: %recv_x_flat_inline622__ssa_v0_view
  int64_t v50 = v48 * v49;
  // pto: %recv_x_flat_inline622__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v51 = pto::Shape<1, 1, 1, -1, -1>(v46, v47, v48, v27, v28);
  // pto: %recv_x_flat_inline622__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v52 = pto::Stride<-1, -1, -1, -1, -1>(v47 * v50, v50, v49, v28, v29);
  // pto: %recv_x_flat_inline622__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v53 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v5, v51, v52);
  // pto: %w1_2d_inline624__ssa_v0_view
  const int64_t v54 = 1;
  // pto: %w1_2d_inline624__ssa_v0_view
  const int64_t v55 = 1;
  // pto: %w1_2d_inline624__ssa_v0_view
  const int64_t v56 = 1;
  // pto: %w1_2d_inline624__ssa_v0_view
  int64_t v57 = v32 * v28;
  // pto: %w1_2d_inline624__ssa_v0_view
  int64_t v58 = v56 * v57;
  // pto: %w1_2d_inline624__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v59 = pto::Shape<1, 1, 1, -1, -1>(v54, v55, v56, v32, v28);
  // pto: %w1_2d_inline624__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v60 = pto::Stride<-1, -1, -1, -1, -1>(v55 * v58, v58, v57, v28, v29);
  // pto: %w1_2d_inline624__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v61 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v59, v60);
  // pto: %w3_2d_inline625__ssa_v0_view
  const int64_t v62 = 1;
  // pto: %w3_2d_inline625__ssa_v0_view
  const int64_t v63 = 1;
  // pto: %w3_2d_inline625__ssa_v0_view
  const int64_t v64 = 1;
  // pto: %w3_2d_inline625__ssa_v0_view
  int64_t v65 = v32 * v28;
  // pto: %w3_2d_inline625__ssa_v0_view
  int64_t v66 = v64 * v65;
  // pto: %w3_2d_inline625__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v67 = pto::Shape<1, 1, 1, -1, -1>(v62, v63, v64, v32, v28);
  // pto: %w3_2d_inline625__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v68 = pto::Stride<-1, -1, -1, -1, -1>(v63 * v66, v66, v65, v28, v29);
  // pto: %w3_2d_inline625__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v69 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v7, v67, v68);
  // pto: %w2_2d_inline626__ssa_v0_view
  const int64_t v70 = 1;
  // pto: %w2_2d_inline626__ssa_v0_view
  const int64_t v71 = 1;
  // pto: %w2_2d_inline626__ssa_v0_view
  const int64_t v72 = 1;
  // pto: %w2_2d_inline626__ssa_v0_view
  int64_t v73 = v35 * v33;
  // pto: %w2_2d_inline626__ssa_v0_view
  int64_t v74 = v72 * v73;
  // pto: %w2_2d_inline626__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v75 = pto::Shape<1, 1, 1, -1, -1>(v70, v71, v72, v35, v33);
  // pto: %w2_2d_inline626__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v76 = pto::Stride<-1, -1, -1, -1, -1>(v71 * v74, v74, v73, v33, v29);
  // pto: %w2_2d_inline626__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v77 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v12, v75, v76);
  auto v78 = TPipe<0, Direction::DIR_BOTH, 32768, 2, 2, false>(v14, v17, v17);
  // pto: %core_inline627__ssa_v0
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID3);
  for (int64_t v79 = (int64_t) v15; v79 < v30; v79 += v36) {
    // pto: %t__tile
    ;
    int32_t v80 = (v2)[v79];
    // pto: %36
    ;
    int64_t v81 = (int64_t) v80;
    // pto: %flat_offset_mul, %37
    ;
    int32_t v82 = (v3)[(int64_t) ((uint64_t) v81 * (uint64_t) v31)];
    // pto: %38
    ;
    int64_t v83 = (int64_t) v82;
    // pto: %n_rows_inline631__tile
    ;
    int32_t v84 = (v4)[v79];
    // pto: %39
    ;
    int64_t v85 = (int64_t) ((uint64_t) v83 * (uint64_t) v33);
    // pto: %40
    ;
    int64_t v86 = (int64_t) ((uint64_t) v83 * (uint64_t) v28);
    // pto: %42
    ;
    int64_t v87 = (int64_t) v84;
    // pto: %43, %44
    ;
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    for (int64_t v88 = v37; v88 < ((int64_t) ((uint64_t) v87 + (uint64_t) v38) / v34); v88 += v29) {
      // pto: %45
      ;
      int64_t v89 = (int64_t) ((uint64_t) v88 * (uint64_t) v34);
      // pto: %41, %46
      ;
      int64_t v90 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v81 * (uint64_t) v34) + (uint64_t) v89);
      // pto: %48
      ;
      int64_t v91 = (int64_t) ((uint64_t) v87 - (uint64_t) v89);
      // pto: %49
      ;
      int64_t v92 = v91 < v34 ? v91 : v34;
      // pto: %gate_acc_inline639__tile
      ;
      Tile<TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v93 = Tile<TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v34, v33);
      // pto: %gate_acc_inline639__tile
      ;
      uint64_t v94 = (uint64_t) v18;
      TASSIGN(v93, v94);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      for (int64_t v95 = v37; v95 < v31; v95 += v39) {
        // pto: %50
        ;
        int64_t v96 = (int64_t) ((uint64_t) v95 * (uint64_t) v40);
        // pto: %52
        ;
        int64_t v97 = (int64_t) ((uint64_t) v96 + (uint64_t) v40);
        // pto: %x_k_inline642__tile
        ;
        Tile<TileType::Mat, int8_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v98 = Tile<TileType::Mat, int8_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v92, v40);
        // pto: %x_k_inline642__tile
        ;
        uint64_t v99 = (uint64_t) v19;
        TASSIGN(v98, v99);
        // pto: %53
        ;
        int64_t v100 = v90 < v37 ? v37 : v90;
        // pto: %54
        ;
        int64_t v101 = v96 < v37 ? v37 : v96;
        // pto: %recv_x_flat_inline622__ssa_v0_pview
        ;
        const int64_t v102 = 0;
        // pto: %recv_x_flat_inline622__ssa_v0_pview
        ;
        __gm__ int8_t* v103 = PTOAS__GLOBAL_TENSOR_DATA(v53);
        // pto: %recv_x_flat_inline622__ssa_v0_pview
        ;
        const int64_t v104 = 1;
        // pto: %recv_x_flat_inline622__ssa_v0_pview
        ;
        const int64_t v105 = 1;
        // pto: %recv_x_flat_inline622__ssa_v0_pview
        ;
        const int64_t v106 = 1;
        // pto: %recv_x_flat_inline622__ssa_v0_pview
        ;
        int64_t v107 = v92 * v28;
        // pto: %recv_x_flat_inline622__ssa_v0_pview
        ;
        int64_t v108 = v106 * v107;
        // pto: %recv_x_flat_inline622__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, -1, 512> v109 = pto::Shape<1, 1, 1, -1, 512>(v104, v105, v106, v92, v40);
        // pto: %recv_x_flat_inline622__ssa_v0_pview
        ;
        pto::Stride<-1, -1, -1, -1, -1> v110 = pto::Stride<-1, -1, -1, -1, -1>(v105 * v108, v108, v107, v28, v29);
        // pto: %recv_x_flat_inline622__ssa_v0_pview
        ;
        GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v111 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v103 + (v102 + v100 * v28 + v101 * v29), v109, v110);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        TLOAD(v98, v111);
        // pto: %w1_k_inline643__tile
        ;
        Tile<TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v112 = Tile<TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v33, v40);
        // pto: %w1_k_inline643__tile
        ;
        uint64_t v113 = (uint64_t) v20;
        TASSIGN(v112, v113);
        // pto: %55
        ;
        int64_t v114 = v85 < v37 ? v37 : v85;
        // pto: %w1_2d_inline624__ssa_v0_pview
        ;
        __gm__ int8_t* v115 = PTOAS__GLOBAL_TENSOR_DATA(v61);
        // pto: %w1_2d_inline624__ssa_v0_pview
        ;
        const int64_t v116 = 0;
        // pto: %w1_2d_inline624__ssa_v0_pview
        ;
        const int64_t v117 = 4096;
        // pto: %w1_2d_inline624__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 256, 512> v118 = pto::Shape<1, 1, 1, 256, 512>();
        // pto: %w1_2d_inline624__ssa_v0_pview
        ;
        pto::Stride<1048576, 1048576, 1048576, 4096, 1> v119 = pto::Stride<1048576, 1048576, 1048576, 4096, 1>();
        // pto: %w1_2d_inline624__ssa_v0_pview
        ;
        GlobalTensor<int8_t, pto::Shape<1, 1, 1, 256, 512>, pto::Stride<1048576, 1048576, 1048576, 4096, 1>, pto::Layout::ND> v120 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 256, 512>, pto::Stride<1048576, 1048576, 1048576, 4096, 1>, pto::Layout::ND>(v115 + (v116 + v114 * v117 + v101), v118, v119);
        TLOAD(v112, v120);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        // pto: %0
        ;
        Tile<TileType::Mat, int8_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v121 = Tile<TileType::Mat, int8_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v92, v40);
        // pto: %0
        ;
        uint64_t v122 = (uint64_t) v21;
        TASSIGN(v121, v122);
        // pto: %58
        ;
        int64_t v123 = v97 < v37 ? v37 : v97;
        // pto: %59
        ;
        const int64_t v124 = 0;
        // pto: %59
        ;
        __gm__ int8_t* v125 = PTOAS__GLOBAL_TENSOR_DATA(v53);
        // pto: %59
        ;
        const int64_t v126 = 1;
        // pto: %59
        ;
        const int64_t v127 = 1;
        // pto: %59
        ;
        const int64_t v128 = 1;
        // pto: %59
        ;
        int64_t v129 = v92 * v28;
        // pto: %59
        ;
        int64_t v130 = v128 * v129;
        // pto: %59
        ;
        pto::Shape<1, 1, 1, -1, 512> v131 = pto::Shape<1, 1, 1, -1, 512>(v126, v127, v128, v92, v40);
        // pto: %59
        ;
        pto::Stride<-1, -1, -1, -1, -1> v132 = pto::Stride<-1, -1, -1, -1, -1>(v127 * v130, v130, v129, v28, v29);
        // pto: %59
        ;
        GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v133 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v125 + (v124 + v100 * v28 + v123 * v29), v131, v132);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        TLOAD(v121, v133);
        // pto: %1
        ;
        Tile<TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v134 = Tile<TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v33, v40);
        // pto: %1
        ;
        uint64_t v135 = (uint64_t) v22;
        TASSIGN(v134, v135);
        // pto: %62
        ;
        __gm__ int8_t* v136 = PTOAS__GLOBAL_TENSOR_DATA(v61);
        // pto: %62
        ;
        const int64_t v137 = 0;
        // pto: %62
        ;
        const int64_t v138 = 4096;
        // pto: %62
        ;
        pto::Shape<1, 1, 1, 256, 512> v139 = pto::Shape<1, 1, 1, 256, 512>();
        // pto: %62
        ;
        pto::Stride<1048576, 1048576, 1048576, 4096, 1> v140 = pto::Stride<1048576, 1048576, 1048576, 4096, 1>();
        // pto: %62
        ;
        GlobalTensor<int8_t, pto::Shape<1, 1, 1, 256, 512>, pto::Stride<1048576, 1048576, 1048576, 4096, 1>, pto::Layout::ND> v141 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 256, 512>, pto::Stride<1048576, 1048576, 1048576, 4096, 1>, pto::Layout::ND>(v136 + (v137 + v114 * v138 + v123), v139, v140);
        TLOAD(v134, v141);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        // pto: %w1_k_inline643__tile_t
        ;
        Tile<TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v142 = Tile<TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v40, v33);
        // pto: %w1_k_inline643__tile_t
        ;
        uint64_t v143 = (uint64_t) v20;
        TASSIGN(v142, v143);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        for (int64_t v144 = v37; v144 < v40; v144 += v33) {
          // pto: %gate_acc_inline639__tile_l0_a
          ;
          Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v145 = Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v92, v41);
          // pto: %gate_acc_inline639__tile_l0_a
          ;
          uint64_t v146 = (uint64_t) v18;
          TASSIGN(v145, v146);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
          pipe_barrier(PIPE_MTE1);
          uint16_t v147 = static_cast<uint16_t>(v37);
          uint16_t v148 = static_cast<uint16_t>(v144);
          TEXTRACT(v145, v98, v147, v148);
          // pto: %gate_acc_inline639__tile_l0_b
          ;
          Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v149 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v41, v33);
          // pto: %gate_acc_inline639__tile_l0_b
          ;
          uint64_t v150 = (uint64_t) v18;
          TASSIGN(v149, v150);
          uint16_t v151 = static_cast<uint16_t>(v144);
          uint16_t v152 = static_cast<uint16_t>(v37);
          TEXTRACT(v149, v142, v151, v152);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          // pto: %2
          ;
          Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v153 = Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v92, v41);
          // pto: %2
          ;
          uint64_t v154 = (uint64_t) v23;
          TASSIGN(v153, v154);
          // pto: %63
          ;
          int64_t v155 = (int64_t) ((uint64_t) v144 + (uint64_t) v41);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
          uint16_t v156 = static_cast<uint16_t>(v37);
          uint16_t v157 = static_cast<uint16_t>(v155);
          TEXTRACT(v153, v98, v156, v157);
          // pto: %3
          ;
          Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v158 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v41, v33);
          // pto: %3
          ;
          uint64_t v159 = (uint64_t) v24;
          TASSIGN(v158, v159);
          uint16_t v160 = static_cast<uint16_t>(v155);
          uint16_t v161 = static_cast<uint16_t>(v37);
          TEXTRACT(v158, v142, v160, v161);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
          // pto: %65
          ;
          bool v162 = v95 == v37;
          // pto: %66, %67
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          if (v162 & v144 == v37) {
            pipe_barrier(PIPE_M);
            TMATMUL(v93, v145, v149);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v93, v93, v145, v149);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
          // pto: %69, %70
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
          if (v162 & v144 == v42) {
            pipe_barrier(PIPE_M);
            TMATMUL(v93, v153, v158);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v93, v93, v153, v158);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        };
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        // pto: %5
        ;
        Tile<TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v163 = Tile<TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v40, v33);
        // pto: %5
        ;
        uint64_t v164 = (uint64_t) v22;
        TASSIGN(v163, v164);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
        for (int64_t v165 = v37; v165 < v40; v165 += v33) {
          // pto: %6
          ;
          Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v166 = Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v92, v41);
          // pto: %6
          ;
          uint64_t v167 = (uint64_t) v25;
          TASSIGN(v166, v167);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
          uint16_t v168 = static_cast<uint16_t>(v37);
          uint16_t v169 = static_cast<uint16_t>(v165);
          TEXTRACT(v166, v121, v168, v169);
          // pto: %7
          ;
          Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v170 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v41, v33);
          // pto: %7
          ;
          uint64_t v171 = (uint64_t) v18;
          TASSIGN(v170, v171);
          pipe_barrier(PIPE_MTE1);
          uint16_t v172 = static_cast<uint16_t>(v165);
          uint16_t v173 = static_cast<uint16_t>(v37);
          TEXTRACT(v170, v163, v172, v173);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
          // pto: %8
          ;
          Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v174 = Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v92, v41);
          // pto: %8
          ;
          uint64_t v175 = (uint64_t) v26;
          TASSIGN(v174, v175);
          // pto: %72
          ;
          int64_t v176 = (int64_t) ((uint64_t) v165 + (uint64_t) v41);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
          uint16_t v177 = static_cast<uint16_t>(v37);
          uint16_t v178 = static_cast<uint16_t>(v176);
          TEXTRACT(v174, v121, v177, v178);
          // pto: %9
          ;
          Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v179 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v41, v33);
          // pto: %9
          ;
          uint64_t v180 = (uint64_t) v24;
          TASSIGN(v179, v180);
          uint16_t v181 = static_cast<uint16_t>(v176);
          uint16_t v182 = static_cast<uint16_t>(v37);
          TEXTRACT(v179, v163, v181, v182);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
          // pto: %74
          ;
          bool v183 = v95 == v43;
          // pto: %75, %76
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
          if (v183 & v165 == v37) {
            pipe_barrier(PIPE_M);
            TMATMUL(v93, v166, v170);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v93, v93, v166, v170);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
          // pto: %78, %79
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
          if (v183 & v165 == v42) {
            pipe_barrier(PIPE_M);
            TMATMUL(v93, v174, v179);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v93, v93, v174, v179);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
        };
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      };
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
      // pto: %up_acc_inline644__tile
      ;
      Tile<TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v184 = Tile<TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v34, v33);
      // pto: %up_acc_inline644__tile
      ;
      uint64_t v185 = (uint64_t) v19;
      TASSIGN(v184, v185);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
      for (int64_t v186 = v37; v186 < v31; v186 += v39) {
        // pto: %80
        ;
        int64_t v187 = (int64_t) ((uint64_t) v186 * (uint64_t) v40);
        // pto: %82
        ;
        int64_t v188 = (int64_t) ((uint64_t) v187 + (uint64_t) v40);
        // pto: %x_u_inline647__tile
        ;
        Tile<TileType::Mat, int8_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v189 = Tile<TileType::Mat, int8_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v92, v40);
        // pto: %x_u_inline647__tile
        ;
        uint64_t v190 = (uint64_t) v19;
        TASSIGN(v189, v190);
        // pto: %83
        ;
        int64_t v191 = v90 < v37 ? v37 : v90;
        // pto: %84
        ;
        int64_t v192 = v187 < v37 ? v37 : v187;
        // pto: %85
        ;
        const int64_t v193 = 0;
        // pto: %85
        ;
        __gm__ int8_t* v194 = PTOAS__GLOBAL_TENSOR_DATA(v53);
        // pto: %85
        ;
        const int64_t v195 = 1;
        // pto: %85
        ;
        const int64_t v196 = 1;
        // pto: %85
        ;
        const int64_t v197 = 1;
        // pto: %85
        ;
        int64_t v198 = v92 * v28;
        // pto: %85
        ;
        int64_t v199 = v197 * v198;
        // pto: %85
        ;
        pto::Shape<1, 1, 1, -1, 512> v200 = pto::Shape<1, 1, 1, -1, 512>(v195, v196, v197, v92, v40);
        // pto: %85
        ;
        pto::Stride<-1, -1, -1, -1, -1> v201 = pto::Stride<-1, -1, -1, -1, -1>(v196 * v199, v199, v198, v28, v29);
        // pto: %85
        ;
        GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v202 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v194 + (v193 + v191 * v28 + v192 * v29), v200, v201);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
        TLOAD(v189, v202);
        // pto: %w3_k_inline648__tile
        ;
        Tile<TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v203 = Tile<TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v33, v40);
        // pto: %w3_k_inline648__tile
        ;
        uint64_t v204 = (uint64_t) v20;
        TASSIGN(v203, v204);
        // pto: %86
        ;
        int64_t v205 = v85 < v37 ? v37 : v85;
        // pto: %w3_2d_inline625__ssa_v0_pview
        ;
        __gm__ int8_t* v206 = PTOAS__GLOBAL_TENSOR_DATA(v69);
        // pto: %w3_2d_inline625__ssa_v0_pview
        ;
        const int64_t v207 = 0;
        // pto: %w3_2d_inline625__ssa_v0_pview
        ;
        const int64_t v208 = 4096;
        // pto: %w3_2d_inline625__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 256, 512> v209 = pto::Shape<1, 1, 1, 256, 512>();
        // pto: %w3_2d_inline625__ssa_v0_pview
        ;
        pto::Stride<1048576, 1048576, 1048576, 4096, 1> v210 = pto::Stride<1048576, 1048576, 1048576, 4096, 1>();
        // pto: %w3_2d_inline625__ssa_v0_pview
        ;
        GlobalTensor<int8_t, pto::Shape<1, 1, 1, 256, 512>, pto::Stride<1048576, 1048576, 1048576, 4096, 1>, pto::Layout::ND> v211 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 256, 512>, pto::Stride<1048576, 1048576, 1048576, 4096, 1>, pto::Layout::ND>(v206 + (v207 + v205 * v208 + v192), v209, v210);
        TLOAD(v203, v211);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        // pto: %12
        ;
        Tile<TileType::Mat, int8_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v212 = Tile<TileType::Mat, int8_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v92, v40);
        // pto: %12
        ;
        uint64_t v213 = (uint64_t) v21;
        TASSIGN(v212, v213);
        // pto: %89
        ;
        int64_t v214 = v188 < v37 ? v37 : v188;
        // pto: %90
        ;
        const int64_t v215 = 0;
        // pto: %90
        ;
        __gm__ int8_t* v216 = PTOAS__GLOBAL_TENSOR_DATA(v53);
        // pto: %90
        ;
        const int64_t v217 = 1;
        // pto: %90
        ;
        const int64_t v218 = 1;
        // pto: %90
        ;
        const int64_t v219 = 1;
        // pto: %90
        ;
        int64_t v220 = v92 * v28;
        // pto: %90
        ;
        int64_t v221 = v219 * v220;
        // pto: %90
        ;
        pto::Shape<1, 1, 1, -1, 512> v222 = pto::Shape<1, 1, 1, -1, 512>(v217, v218, v219, v92, v40);
        // pto: %90
        ;
        pto::Stride<-1, -1, -1, -1, -1> v223 = pto::Stride<-1, -1, -1, -1, -1>(v218 * v221, v221, v220, v28, v29);
        // pto: %90
        ;
        GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v224 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v216 + (v215 + v191 * v28 + v214 * v29), v222, v223);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
        TLOAD(v212, v224);
        // pto: %13
        ;
        Tile<TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v225 = Tile<TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v33, v40);
        // pto: %13
        ;
        uint64_t v226 = (uint64_t) v22;
        TASSIGN(v225, v226);
        // pto: %93
        ;
        __gm__ int8_t* v227 = PTOAS__GLOBAL_TENSOR_DATA(v69);
        // pto: %93
        ;
        const int64_t v228 = 0;
        // pto: %93
        ;
        const int64_t v229 = 4096;
        // pto: %93
        ;
        pto::Shape<1, 1, 1, 256, 512> v230 = pto::Shape<1, 1, 1, 256, 512>();
        // pto: %93
        ;
        pto::Stride<1048576, 1048576, 1048576, 4096, 1> v231 = pto::Stride<1048576, 1048576, 1048576, 4096, 1>();
        // pto: %93
        ;
        GlobalTensor<int8_t, pto::Shape<1, 1, 1, 256, 512>, pto::Stride<1048576, 1048576, 1048576, 4096, 1>, pto::Layout::ND> v232 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 256, 512>, pto::Stride<1048576, 1048576, 1048576, 4096, 1>, pto::Layout::ND>(v227 + (v228 + v205 * v229 + v214), v230, v231);
        TLOAD(v225, v232);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
        // pto: %w3_k_inline648__tile_t
        ;
        Tile<TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v233 = Tile<TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v40, v33);
        // pto: %w3_k_inline648__tile_t
        ;
        uint64_t v234 = (uint64_t) v20;
        TASSIGN(v233, v234);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        for (int64_t v235 = v37; v235 < v40; v235 += v33) {
          // pto: %up_acc_inline644__tile_l0_a
          ;
          Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v236 = Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v92, v41);
          // pto: %up_acc_inline644__tile_l0_a
          ;
          uint64_t v237 = (uint64_t) v18;
          TASSIGN(v236, v237);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
          uint16_t v238 = static_cast<uint16_t>(v37);
          uint16_t v239 = static_cast<uint16_t>(v235);
          TEXTRACT(v236, v189, v238, v239);
          // pto: %up_acc_inline644__tile_l0_b
          ;
          Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v240 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v41, v33);
          // pto: %up_acc_inline644__tile_l0_b
          ;
          uint64_t v241 = (uint64_t) v18;
          TASSIGN(v240, v241);
          pipe_barrier(PIPE_MTE1);
          uint16_t v242 = static_cast<uint16_t>(v235);
          uint16_t v243 = static_cast<uint16_t>(v37);
          TEXTRACT(v240, v233, v242, v243);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
          // pto: %14
          ;
          Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v244 = Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v92, v41);
          // pto: %14
          ;
          uint64_t v245 = (uint64_t) v23;
          TASSIGN(v244, v245);
          // pto: %94
          ;
          int64_t v246 = (int64_t) ((uint64_t) v235 + (uint64_t) v41);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
          uint16_t v247 = static_cast<uint16_t>(v37);
          uint16_t v248 = static_cast<uint16_t>(v246);
          TEXTRACT(v244, v189, v247, v248);
          // pto: %15
          ;
          Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v249 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v41, v33);
          // pto: %15
          ;
          uint64_t v250 = (uint64_t) v24;
          TASSIGN(v249, v250);
          uint16_t v251 = static_cast<uint16_t>(v246);
          uint16_t v252 = static_cast<uint16_t>(v37);
          TEXTRACT(v249, v233, v251, v252);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
          // pto: %96
          ;
          bool v253 = v186 == v37;
          // pto: %97, %98
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
          if (v253 & v235 == v37) {
            pipe_barrier(PIPE_M);
            TMATMUL(v184, v236, v240);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v184, v184, v236, v240);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
          // pto: %100, %101
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
          if (v253 & v235 == v42) {
            pipe_barrier(PIPE_M);
            TMATMUL(v184, v244, v249);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v184, v184, v244, v249);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        };
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        // pto: %17
        ;
        Tile<TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v254 = Tile<TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v40, v33);
        // pto: %17
        ;
        uint64_t v255 = (uint64_t) v22;
        TASSIGN(v254, v255);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        for (int64_t v256 = v37; v256 < v40; v256 += v33) {
          // pto: %18
          ;
          Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v257 = Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v92, v41);
          // pto: %18
          ;
          uint64_t v258 = (uint64_t) v25;
          TASSIGN(v257, v258);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
          uint16_t v259 = static_cast<uint16_t>(v37);
          uint16_t v260 = static_cast<uint16_t>(v256);
          TEXTRACT(v257, v212, v259, v260);
          // pto: %19
          ;
          Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v261 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v41, v33);
          // pto: %19
          ;
          uint64_t v262 = (uint64_t) v18;
          TASSIGN(v261, v262);
          pipe_barrier(PIPE_MTE1);
          uint16_t v263 = static_cast<uint16_t>(v256);
          uint16_t v264 = static_cast<uint16_t>(v37);
          TEXTRACT(v261, v254, v263, v264);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
          // pto: %20
          ;
          Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v265 = Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v92, v41);
          // pto: %20
          ;
          uint64_t v266 = (uint64_t) v26;
          TASSIGN(v265, v266);
          // pto: %103
          ;
          int64_t v267 = (int64_t) ((uint64_t) v256 + (uint64_t) v41);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
          uint16_t v268 = static_cast<uint16_t>(v37);
          uint16_t v269 = static_cast<uint16_t>(v267);
          TEXTRACT(v265, v212, v268, v269);
          // pto: %21
          ;
          Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v270 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v41, v33);
          // pto: %21
          ;
          uint64_t v271 = (uint64_t) v24;
          TASSIGN(v270, v271);
          uint16_t v272 = static_cast<uint16_t>(v267);
          uint16_t v273 = static_cast<uint16_t>(v37);
          TEXTRACT(v270, v254, v272, v273);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
          // pto: %105
          ;
          bool v274 = v186 == v43;
          // pto: %106, %107
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
          if (v274 & v256 == v37) {
            pipe_barrier(PIPE_M);
            TMATMUL(v184, v257, v261);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v184, v184, v257, v261);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
          // pto: %109, %110
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
          if (v274 & v256 == v42) {
            pipe_barrier(PIPE_M);
            TMATMUL(v184, v265, v270);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v184, v184, v265, v270);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        };
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      };
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID6);
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      TPUSH<TPipe<0, Direction::DIR_BOTH, 32768, 2, 2, false>, Tile<TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v78, v93);
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
      TPUSH<TPipe<0, Direction::DIR_BOTH, 32768, 2, 2, false>, Tile<TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v78, v184);
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
      // WP_FIRST_BEGIN: original Up L1 users and final MMA have completed.
      using wp_tile_t = Tile<TileType::Mat, int8_t, 512, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>;
      wp_tile_t wp_first(512, 256);
      TASSIGN(wp_first, static_cast<uint64_t>(v20));
      int64_t wp_row = v86 < v37 ? v37 : v86;
      auto wp_addr = PTOAS__GLOBAL_TENSOR_DATA(v77) + wp_row * 256;
      using wp_shape_t = pto::Shape<1, 1, 1, 512, 256>;
      using wp_stride_t = pto::Stride<131072, 131072, 131072, 256, 1>;
      GlobalTensor<int8_t, wp_shape_t, wp_stride_t, pto::Layout::ND> wp_global(wp_addr, wp_shape_t(), wp_stride_t());

      TLOAD(wp_first, wp_global);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID5);

      // WP_FIRST_END

      // pto: %h_gathered_inline675__tile
      ;
      const int32_t v275 = 16;
      // pto: %h_gathered_inline675__tile
      ;
      const int32_t v276 = 256;
      // pto: %h_gathered_inline675__tile
      ;
      Tile<TileType::Mat, int8_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v277 = Tile<TileType::Mat, int8_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v275, v276);
      pipe_barrier(PIPE_MTE2);
      TPOP<TPipe<0, Direction::DIR_BOTH, 32768, 2, 2, false>, Tile<TileType::Mat, int8_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v78, v277);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID6);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
      for (int64_t v278 = v37; v278 < v31; v278 += v39) {
        // pto: %111
        ;
        int64_t v279 = (int64_t) ((uint64_t) v278 * (uint64_t) v40);
        // pto: %w2_k_inline678__tile
        ;
        Tile<TileType::Mat, int8_t, 512, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v280 = Tile<TileType::Mat, int8_t, 512, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v40, v33);
        // pto: %w2_k_inline678__tile
        ;
        uint64_t v281 = (uint64_t) v20;
        TASSIGN(v280, v281);
        // pto: %114
        ;
        int64_t v282 = (int64_t) ((uint64_t) v86 + (uint64_t) v279);
        // pto: %115
        ;
        int64_t v283 = v282 < v37 ? v37 : v282;
        // pto: %w2_2d_inline626__ssa_v0_pview
        ;
        __gm__ int8_t* v284 = PTOAS__GLOBAL_TENSOR_DATA(v77);
        // pto: %w2_2d_inline626__ssa_v0_pview
        ;
        const int64_t v285 = 0;
        // pto: %w2_2d_inline626__ssa_v0_pview
        ;
        const int64_t v286 = 256;
        // pto: %w2_2d_inline626__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 512, 256> v287 = pto::Shape<1, 1, 1, 512, 256>();
        // pto: %w2_2d_inline626__ssa_v0_pview
        ;
        pto::Stride<131072, 131072, 131072, 256, 1> v288 = pto::Stride<131072, 131072, 131072, 256, 1>();
        // pto: %w2_2d_inline626__ssa_v0_pview
        ;
        GlobalTensor<int8_t, pto::Shape<1, 1, 1, 512, 256>, pto::Stride<131072, 131072, 131072, 256, 1>, pto::Layout::ND> v289 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 512, 256>, pto::Stride<131072, 131072, 131072, 256, 1>, pto::Layout::ND>(v284 + (v285 + v283 * v286), v287, v288);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
        // WP_SKIP_BEGIN: first panel already loaded, original ID7 credit still consumed above.
        if (v278 != v37) {
        TLOAD(v280, v289);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID5);
        }
        // WP_SKIP_END
        // pto: %24
        ;
        Tile<TileType::Mat, int8_t, 512, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v290 = Tile<TileType::Mat, int8_t, 512, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v40, v33);
        // pto: %24
        ;
        uint64_t v291 = (uint64_t) v22;
        TASSIGN(v290, v291);
        // pto: %116, %113
        ;
        int64_t v292 = (int64_t) ((uint64_t) v86 + (uint64_t) ((int64_t) (uint64_t) v279 + (uint64_t) v40));
        // pto: %117
        ;
        int64_t v293 = v292 < v37 ? v37 : v292;
        // pto: %118
        ;
        __gm__ int8_t* v294 = PTOAS__GLOBAL_TENSOR_DATA(v77);
        // pto: %118
        ;
        const int64_t v295 = 0;
        // pto: %118
        ;
        const int64_t v296 = 256;
        // pto: %118
        ;
        pto::Shape<1, 1, 1, 512, 256> v297 = pto::Shape<1, 1, 1, 512, 256>();
        // pto: %118
        ;
        pto::Stride<131072, 131072, 131072, 256, 1> v298 = pto::Stride<131072, 131072, 131072, 256, 1>();
        // pto: %118
        ;
        GlobalTensor<int8_t, pto::Shape<1, 1, 1, 512, 256>, pto::Stride<131072, 131072, 131072, 256, 1>, pto::Layout::ND> v299 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 512, 256>, pto::Stride<131072, 131072, 131072, 256, 1>, pto::Layout::ND>(v294 + (v295 + v293 * v296), v297, v298);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        TLOAD(v290, v299);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID6);
        // pto: %w2_k_inline678__tile_t
        ;
        Tile<TileType::Mat, int8_t, 256, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v300 = Tile<TileType::Mat, int8_t, 256, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v33, v40);
        // pto: %w2_k_inline678__tile_t
        ;
        uint64_t v301 = (uint64_t) v20;
        TASSIGN(v300, v301);
        // pto: %y_acc_inline679__tile_l0_init
        ;
        Tile<TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v302 = Tile<TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v34, v40);
        // pto: %y_acc_inline679__tile_l0_init
        ;
        uint64_t v303 = (uint64_t) v18;
        TASSIGN(v302, v303);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID5);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        for (int64_t v304 = v37; v304 < v33; v304 += v41) {
          // pto: %y_acc_inline679__tile_l0_a
          ;
          Tile<TileType::Left, int8_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v305 = Tile<TileType::Left, int8_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v34, v44);
          // pto: %y_acc_inline679__tile_l0_a
          ;
          uint64_t v306 = (uint64_t) v18;
          TASSIGN(v305, v306);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
          uint16_t v307 = static_cast<uint16_t>(v37);
          uint16_t v308 = static_cast<uint16_t>(v304);
          TEXTRACT(v305, v277, v307, v308);
          // pto: %y_acc_inline679__tile_l0_b
          ;
          Tile<TileType::Right, int8_t, 64, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v309 = Tile<TileType::Right, int8_t, 64, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v44, v40);
          // pto: %y_acc_inline679__tile_l0_b
          ;
          uint64_t v310 = (uint64_t) v18;
          TASSIGN(v309, v310);
          pipe_barrier(PIPE_MTE1);
          uint16_t v311 = static_cast<uint16_t>(v304);
          uint16_t v312 = static_cast<uint16_t>(v37);
          TEXTRACT(v309, v300, v311, v312);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          // pto: %25
          ;
          Tile<TileType::Left, int8_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v313 = Tile<TileType::Left, int8_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v34, v44);
          // pto: %25
          ;
          uint64_t v314 = (uint64_t) v23;
          TASSIGN(v313, v314);
          // pto: %119
          ;
          int64_t v315 = (int64_t) ((uint64_t) v304 + (uint64_t) v44);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
          uint16_t v316 = static_cast<uint16_t>(v37);
          uint16_t v317 = static_cast<uint16_t>(v315);
          TEXTRACT(v313, v277, v316, v317);
          // pto: %26
          ;
          Tile<TileType::Right, int8_t, 64, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v318 = Tile<TileType::Right, int8_t, 64, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v44, v40);
          // pto: %26
          ;
          uint64_t v319 = (uint64_t) v24;
          TASSIGN(v318, v319);
          uint16_t v320 = static_cast<uint16_t>(v315);
          uint16_t v321 = static_cast<uint16_t>(v37);
          TEXTRACT(v318, v300, v320, v321);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
          // pto: %121
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          if (v304 == v37) {
            pipe_barrier(PIPE_M);
            TMATMUL(v302, v305, v309);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v302, v302, v305, v309);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
          // pto: %122
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
          if (v304 == v45) {
            pipe_barrier(PIPE_M);
            TMATMUL(v302, v313, v318);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v302, v302, v313, v318);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        };
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
        TPUSH<TPipe<0, Direction::DIR_BOTH, 32768, 2, 2, false>, Tile<TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v78, v302);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
        // pto: %28
        ;
        Tile<TileType::Mat, int8_t, 256, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v322 = Tile<TileType::Mat, int8_t, 256, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v33, v40);
        // pto: %28
        ;
        uint64_t v323 = (uint64_t) v22;
        TASSIGN(v322, v323);
        // pto: %29
        ;
        Tile<TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v324 = Tile<TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v34, v40);
        // pto: %29
        ;
        uint64_t v325 = (uint64_t) v19;
        TASSIGN(v324, v325);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID6);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID3);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        for (int64_t v326 = v37; v326 < v33; v326 += v41) {
          // pto: %30
          ;
          Tile<TileType::Left, int8_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v327 = Tile<TileType::Left, int8_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v34, v44);
          // pto: %30
          ;
          uint64_t v328 = (uint64_t) v25;
          TASSIGN(v327, v328);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
          uint16_t v329 = static_cast<uint16_t>(v37);
          uint16_t v330 = static_cast<uint16_t>(v326);
          TEXTRACT(v327, v277, v329, v330);
          // pto: %31
          ;
          Tile<TileType::Right, int8_t, 64, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v331 = Tile<TileType::Right, int8_t, 64, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v44, v40);
          // pto: %31
          ;
          uint64_t v332 = (uint64_t) v18;
          TASSIGN(v331, v332);
          pipe_barrier(PIPE_MTE1);
          uint16_t v333 = static_cast<uint16_t>(v326);
          uint16_t v334 = static_cast<uint16_t>(v37);
          TEXTRACT(v331, v322, v333, v334);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          // pto: %32
          ;
          Tile<TileType::Left, int8_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v335 = Tile<TileType::Left, int8_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v34, v44);
          // pto: %32
          ;
          uint64_t v336 = (uint64_t) v26;
          TASSIGN(v335, v336);
          // pto: %124
          ;
          int64_t v337 = (int64_t) ((uint64_t) v326 + (uint64_t) v44);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
          uint16_t v338 = static_cast<uint16_t>(v37);
          uint16_t v339 = static_cast<uint16_t>(v337);
          TEXTRACT(v335, v277, v338, v339);
          // pto: %33
          ;
          Tile<TileType::Right, int8_t, 64, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v340 = Tile<TileType::Right, int8_t, 64, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v44, v40);
          // pto: %33
          ;
          uint64_t v341 = (uint64_t) v24;
          TASSIGN(v340, v341);
          uint16_t v342 = static_cast<uint16_t>(v337);
          uint16_t v343 = static_cast<uint16_t>(v37);
          TEXTRACT(v340, v322, v342, v343);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
          // pto: %126
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          if (v326 == v37) {
            pipe_barrier(PIPE_M);
            TMATMUL(v324, v327, v331);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v324, v324, v327, v331);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
          // pto: %127
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
          if (v326 == v45) {
            pipe_barrier(PIPE_M);
            TMATMUL(v324, v335, v340);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v324, v324, v335, v340);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        };
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID3);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID3);
        TPUSH<TPipe<0, Direction::DIR_BOTH, 32768, 2, 2, false>, Tile<TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v78, v324);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID3);
      };
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TFREE<TPipe<0, Direction::DIR_BOTH, 32768, 2, 2, false>, TileSplitAxis::TILE_UP_DOWN>(v78);
    };
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  }
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID3);
  #endif // __DAV_CUBE__
