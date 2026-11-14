// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
#include "intrinsic.h"
#include "tensor.h"
#include <cstdint>
#include <pto/pto-inst.hpp>
using namespace pto;
// Independent gate/up L1 release and readiness events on the K256 DB schedule.
template <int M>
static __aicore__ void
gate_up(__gm__ int8_t *x, __gm__ int32_t *gate, __gm__ int32_t *up,
        __gm__ int8_t *w1, __gm__ int8_t *w3, int64_t row, int64_t expert,
        int64_t cache_offset, int32_t block, int32_t blocks) {
#if defined(__DAV_CUBE__)
  using XMat = Tile<TileType::Mat, int8_t, M, 4096, BLayout::ColMajor, M, 4096,
                    SLayout::RowMajor, 512>;
  using WMat = Tile<TileType::Mat, int8_t, 256, 256, BLayout::ColMajor, 256,
                    256, SLayout::RowMajor, 512>;
  using WTranspose = Tile<TileType::Mat, int8_t, 256, 256, BLayout::RowMajor,
                          256, 256, SLayout::ColMajor, 512>;
  using A = Tile<TileType::Left, int8_t, M, 128, BLayout::RowMajor, M, 128,
                 SLayout::RowMajor, 512>;
  using B = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, 128, 256,
                 SLayout::ColMajor, 512>;
  using C = Tile<TileType::Acc, int32_t, M, 256, BLayout::ColMajor, M, 256,
                 SLayout::RowMajor, 1024>;
  XMat xl;
  WMat wg, wu;
  WTranspose wgt, wut;
  C cg, cu;
  TASSIGN(xl, 0);
  TASSIGN(cg, 0);
  TASSIGN(cu, 65536);
  using XGlobal =
      GlobalTensor<int8_t, Shape<1, 1, 1, M, 4096>,
                   Stride<M * 4096, M * 4096, M * 4096, 4096, 1>, Layout::ND>;
  using WGlobal = GlobalTensor<int8_t, Shape<1, 8, 16, 16, 32>,
                               Stride<8388608, 65536, 512, 32, 1>, Layout::NZ>;
  using OGlobal =
      GlobalTensor<int32_t, Shape<1, 1, 1, M, 256>,
                   Stride<M * 2048, M * 2048, M * 2048, 2048, 1>, Layout::ND>;
  XGlobal xin(x + row * 4096);
  TLOAD(xl, xin);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
  // Event 0/1 protect the two L0B slots; 2/3 protect shared X slots.
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  for (int ninner = 0; ninner < 2; ++ninner) {
    int n = block * 512 + ninner * 256;
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    // Gate slots use event 0/1; up slots use event 2/3.
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TASSIGN(wg, 262144);
    TASSIGN(wu, 327680);
    int64_t first_offset = expert * 8388608 + (n / 16) * 512 + cache_offset;
    WGlobal first_g(w1 + first_offset), first_u(w3 + first_offset);
    TLOAD(wg, first_g);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    TLOAD(wu, first_u);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
    for (int k = 0; k < 4096; k += 256) {
      int panel = (k / 256) & 1;
      // Preload the next pair while the current panel feeds L0.
      if (k + 256 < 4096) {
        int next = panel ^ 1;
        if (next == 0)
          wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        else
          wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        TASSIGN(wg, 262144 + next * 131072);
        TASSIGN(wu, 327680 + next * 131072);
        int64_t next_offset = expert * 8388608 + ((k + 256) / 32) * 65536 +
                              (n / 16) * 512 + cache_offset;
        WGlobal next_g(w1 + next_offset), next_u(w3 + next_offset);
        TLOAD(wg, next_g);
        if (next == 0)
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        else
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        if (next == 0)
          wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        else
          wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        TLOAD(wu, next_u);
        if (next == 0)
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        else
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
      }
      if (panel == 0)
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      else
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TASSIGN(wgt, 262144 + panel * 131072);
      TASSIGN(wut, 327680 + panel * 131072);
      for (int kk = 0; kk < 256; kk += 128) {
        int slot = (kk / 128) & 1;
        A av;
        B bg, bu;
        TASSIGN(av, slot * 8192);
        TASSIGN(bg, 0);
        TASSIGN(bu, 32768);
        if (slot == 0)
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        else
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        TEXTRACT(av, xl, 0, k + kk);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        TEXTRACT(bg, wgt, kk, 0);
        if (kk == 128) {
          if (panel == 0)
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
          else
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        }
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        if (kk == 0) {
          if (panel == 0)
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
          else
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
        }
        TEXTRACT(bu, wut, kk, 0);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        if (kk == 128) {
          if (panel == 0)
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
          else
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        }
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        if (k == 0 && kk == 0)
          TMATMUL(cg, av, bg);
        else
          TMATMUL_ACC(cg, cg, av, bg);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        if (k == 0 && kk == 0)
          TMATMUL(cu, av, bu);
        else
          TMATMUL_ACC(cu, cu, av, bu);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        if (slot == 0)
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        else
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
      }
    }
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    OGlobal gout(gate + n), uout(up + n);
    TSTORE(gout, cg);
    TSTORE(uout, cu);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
#endif
  pipe_barrier(PIPE_ALL);
}

// Pack gate/up along N in L1; alternate two complete L0 operand slots.
template <int M, int N, int KP, int KL, bool FullX>
static __aicore__ void
paired_gate_up(__gm__ int8_t *x, __gm__ int32_t *gate, __gm__ int32_t *up,
               __gm__ int8_t *w1, __gm__ int8_t *w3, int64_t row,
               int64_t expert, int64_t cache_offset, int32_t block) {
#if defined(__DAV_CUBE__)
  constexpr int XK = FullX ? 4096 : KP;
  constexpr int XB = M * XK;
  constexpr int WB = N * KP;
  constexpr int BASE = FullX ? XB : 2 * XB;
  using XMat = Tile<TileType::Mat, int8_t, M, XK, BLayout::ColMajor, M, XK,
                    SLayout::RowMajor, 512>;
  using WMat = Tile<TileType::Mat, int8_t, 2 * N, KP, BLayout::ColMajor, N, KP,
                    SLayout::RowMajor, 512>;
  using WT = Tile<TileType::Mat, int8_t, KP, 2 * N, BLayout::RowMajor, KP,
                  2 * N, SLayout::ColMajor, 512>;
  using A = Tile<TileType::Left, int8_t, M, KL, BLayout::RowMajor, M, KL,
                 SLayout::RowMajor, 512>;
  using B = Tile<TileType::Right, int8_t, KL, 2 * N, BLayout::RowMajor, KL,
                 2 * N, SLayout::ColMajor, 512>;
  using C = Tile<TileType::Acc, int32_t, M, 2 * N, BLayout::ColMajor, M, 2 * N,
                 SLayout::RowMajor, 1024>;
  using HalfC = Tile<TileType::Acc, int32_t, M, N, BLayout::ColMajor, M, N,
                     SLayout::RowMajor, 1024>;
  using XGlobal =
      GlobalTensor<int8_t, Shape<1, 1, 1, M, XK>,
                   Stride<M * 4096, M * 4096, M * 4096, 4096, 1>, Layout::ND>;
  using WGlobal = GlobalTensor<int8_t, Shape<1, KP / 32, N / 16, 16, 32>,
                               Stride<8388608, 65536, 512, 32, 1>, Layout::NZ>;
  using OGlobal =
      GlobalTensor<int32_t, Shape<1, 1, 1, M, N>,
                   Stride<M * 2048, M * 2048, M * 2048, 2048, 1>, Layout::ND>;
  XMat xl;
  WMat wg, wu;
  WT wt;
  C c;
  HalfC cg, cu;
  TASSIGN(c, 0);
  TASSIGN(cg, 0);
  TASSIGN(cu, M * N * 4);
  if constexpr (FullX) {
    TASSIGN(xl, 0);
    XGlobal xin(x + row * 4096);
    TLOAD(xl, xin);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
  }
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  for (int ni = 0; ni < 512 / N; ++ni) {
    const int n = block * 512 + ni * N;
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    if (ni == 0) {
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      if constexpr (!FullX) {
        TASSIGN(xl, 0);
        XGlobal xin(x + row * 4096);
        TLOAD(xl, xin);
      }
      TASSIGN(wg, BASE);
      TASSIGN(wu, BASE + N * 32);
      int64_t offset = expert * 8388608 + (n / 16) * 512 + cache_offset;
      WGlobal gin(w1 + offset), uin(w3 + offset);
      TLOAD(wg, gin);
      TLOAD(wu, uin);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    }
    for (int k = 0; k < 4096; k += KP) {
      int panel = (k / KP) & 1;
      if (k + KP < 4096 || ni + 1 < 512 / N) {
        int next = panel ^ 1;
        if (next == 0)
          wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        else
          wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        if constexpr (!FullX) {
          TASSIGN(xl, next * XB);
          XGlobal xin(x + row * 4096 + (k + KP) % 4096);
          TLOAD(xl, xin);
        }
        TASSIGN(wg, BASE + next * 2 * WB);
        TASSIGN(wu, BASE + next * 2 * WB + N * 32);
        int64_t offset = expert * 8388608 + (((k + KP) % 4096) / 32) * 65536 +
                         ((n + (k + KP == 4096 ? N : 0)) / 16) * 512 +
                         cache_offset;
        WGlobal gin(w1 + offset), uin(w3 + offset);
        TLOAD(wg, gin);
        TLOAD(wu, uin);
        if (next == 0)
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        else
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      }
      if (panel == 0)
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      else
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TASSIGN(xl, FullX ? 0 : panel * XB);
      TASSIGN(wt, BASE + panel * 2 * WB);
      for (int kk = 0; kk < KP; kk += KL) {
        int slot = ((k + kk) / KL) & 1;
        A a;
        B b;
        TASSIGN(a, slot * M * KL);
        TASSIGN(b, slot * KL * 2 * N);
        if (slot == 0)
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        else
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        TEXTRACT(a, xl, 0, (FullX ? k : 0) + kk);
        TEXTRACT(b, wt, kk, 0);
        if (kk + KL == KP) {
          if (panel == 0)
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
          else
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        }
        if (slot == 0) {
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        } else {
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        }
        if (k == 0 && kk == 0)
          TMATMUL(c, a, b);
        else
          TMATMUL_ACC(c, c, a, b);
        if (slot == 0)
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        else
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      }
    }
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    OGlobal gout(gate + n), uout(up + n);
    TSTORE(gout, cg);
    TSTORE(uout, cu);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
#endif
  pipe_barrier(PIPE_ALL);
}
template <typename T>
static __aicore__ inline __gm__ T *data(__gm__ int64_t *args, int index) {
  auto tensor = reinterpret_cast<__gm__ Tensor *>(args[index]);
  return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) +
         tensor->start_offset;
}
extern "C" __aicore__ __attribute__((always_inline)) void
kernel_entry(__gm__ int64_t *args) {
  set_atomic_none();
  set_ctrl(sbitset0(get_ctrl(), pto::MAD_MODE_BIT));
  auto output = reinterpret_cast<__gm__ Tensor *>(args[1]);
  if (output->shapes[0] == 128) {
    paired_gate_up<128, 128, 512, 128, false>(
        data<int8_t>(args, 0), data<int32_t>(args, 1), data<int32_t>(args, 2),
        data<int8_t>(args, 3), data<int8_t>(args, 4), args[5], args[6], 0,
        get_block_idx(args));
  } else if (output->shapes[0] == 16) {
    gate_up<16>(data<int8_t>(args, 0), data<int32_t>(args, 1),
                data<int32_t>(args, 2), data<int8_t>(args, 3),
                data<int8_t>(args, 4), args[5], args[6], 0, get_block_idx(args),
                get_block_num(args));
  } else {
    gate_up<64>(data<int8_t>(args, 0), data<int32_t>(args, 1),
                data<int32_t>(args, 2), data<int8_t>(args, 3),
                data<int8_t>(args, 4), args[5], args[6], 0, get_block_idx(args),
                get_block_num(args));
  }
}
