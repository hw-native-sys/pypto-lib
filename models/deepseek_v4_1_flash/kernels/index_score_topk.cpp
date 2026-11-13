// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Sharded paged scores, local TopK, and candidate merge in one MIX kernel.
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"

using namespace pto;

namespace {
namespace score_core {
using F4 = float4_e2m1x2_t;
using S = float8_e8m0_t;

template <typename T, int R, int C, int BR, int BC, Layout L = Layout::ND>
using GM = GlobalTensor<T, TileShape2D<T, R, C, L>, BaseShape2D<T, BR, BC, L>, L>;
template <typename T, int R, int C>
using Vec = Tile<TileType::Vec, T, R, C>;
using QueryScale = Tile<TileType::Mat, S, 192, 4, BLayout::RowMajor, DYNAMIC, 4, SLayout::RowMajor, 32>;
using KeyScale = Tile<TileType::Mat, S, 4, 128, BLayout::ColMajor, 4, 128, SLayout::ColMajor, 32>;

template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int index) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

__aicore__ inline int read(__gm__ int32_t* ptr) {
    dcci((__gm__ void*)ptr, cache_line_t::SINGLE_CACHE_LINE);
    return *ptr;
}

__aicore__ inline int minimum(int a, int b) { return a < b ? a : b; }

#if defined(__DAV_VEC__)
__aicore__ inline void prepare_weights(__gm__ int64_t* args, int begin, int count) {
    Tile<TileType::Vec, bfloat16_t, 6, 32, BLayout::RowMajor, DYNAMIC, 32> weight(count);
    Vec<bfloat16_t, 1, 192> weight_raw;
    Vec<float, 1, 192> scaled;
    Vec<bfloat16_t, 1, 192> rounded;
    TASSIGN(weight, 256000);
    TASSIGN(weight_raw, 256000);
    TASSIGN(scaled, 256512);
    TASSIGN(rounded, 257280);
    GM<bfloat16_t, DYNAMIC, 32, DYNAMIC, 32> wg(data<bfloat16_t>(args, 2) + begin * 32,
        TileShape2D<bfloat16_t, DYNAMIC, 32>(count, 32), BaseShape2D<bfloat16_t, DYNAMIC, 32>(count, 32));
    // Preserve the previous fragment's vector reads before reusing weight storage.
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(weight, wg);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCVT(scaled, weight_raw, RoundMode::CAST_RINT);
    TMULS(scaled, scaled, 0.015625f);
    TCVT(rounded, scaled, RoundMode::CAST_RINT);
}

#endif
}
}

using namespace score_core;
namespace {
    namespace fused {
        // Main-phase UB: QK [0, 98304), three uint16 caches [98304, 251904),
        // cumulative histograms [251904, 253440), and weights above 256000.
        // Local TopK reuses the drained QK slots; merge reuses the main-phase buffers.
#if defined(__DAV_VEC__)
        template <typename T>
        __aicore__ inline __ubuf__ T* ub(int offset) {
            Vec<T, 1, 32> tile;
            TASSIGN(tile, offset);
            return tile.data();
        }
        using CandidateKeys = Vec<uint16_t, 1, 512>;
        using CandidateIndices = Vec<uint32_t, 1, 512>;
        __aicore__ inline void write_cached_row(RegTensor<bfloat16_t>& a, RegTensor<bfloat16_t>& b,
        __ubuf__ uint16_t* dst, __ubuf__ uint16_t* hist, int valid0, int valid1,
        int span0, int span1, bool reset, __ubuf__ uint16_t* pool_valid, bool gathered) {
            vector_u16 h0, h1, pos, negkey, sign, key, high, invalid, lane_valid;
            vector_u8 bytes;
            vector_bool negative, pool_mask;
            auto full = pset_b16(PAT_ALL);
            mem_bar(VST_VLD);
            if (reset) {
                vbr(h0, uint16_t(0));
                vbr(h1, uint16_t(0));
            }
            else {
                vlds(h0, hist, 0, NORM);
                vlds(h1, hist, 128, NORM);
            }
            vbr(sign, uint16_t(0x8000));
            vbr(invalid, uint16_t(0x007f));
            vadds(a, a, bfloat16_t(0), full, MODE_ZEROING);
            vcmps_lt(negative, (vector_s16&)a, int16_t(0), full);
            vxor(pos, (vector_u16&)a, sign, full, MODE_ZEROING);
            vnot(negkey, (vector_u16&)a, full, MODE_ZEROING);
            vsel(key, negkey, pos, negative);
            uint32_t remaining = valid0;
            auto valid = CreatePredicate<uint16_t>(remaining);
            vsel(key, key, invalid, valid);
            if (gathered) {
                vlds(lane_valid, pool_valid, 0, NORM);
                vcmps_eq(pool_mask, lane_valid, uint16_t(1), full);
                vsel(key, key, invalid, pool_mask);
            }
            vsts(key, dst, 0, NORM_B16, full);
            vshrs(high, key, 8, full, MODE_ZEROING);
            vpack(bytes, high, LOWER);
            remaining = span0;
            auto hist_mask = CreatePredicate<uint8_t>(remaining);
            chistv2(h0, bytes, hist_mask, Bin_N0);
            chistv2(h1, bytes, hist_mask, Bin_N1);
            vadds(b, b, bfloat16_t(0), full, MODE_ZEROING);
            vcmps_lt(negative, (vector_s16&)b, int16_t(0), full);
            vxor(pos, (vector_u16&)b, sign, full, MODE_ZEROING);
            vnot(negkey, (vector_u16&)b, full, MODE_ZEROING);
            vsel(key, negkey, pos, negative);
            remaining = valid1;
            valid = CreatePredicate<uint16_t>(remaining);
            vsel(key, key, invalid, valid);
            if (gathered) {
                vlds(lane_valid, pool_valid, 128, NORM);
                vcmps_eq(pool_mask, lane_valid, uint16_t(1), full);
                vsel(key, key, invalid, pool_mask);
            }
            vsts(key, dst, 128, NORM_B16, full);
            vshrs(high, key, 8, full, MODE_ZEROING);
            vpack(bytes, high, LOWER);
            remaining = span1;
            hist_mask = CreatePredicate<uint8_t>(remaining);
            chistv2(h0, bytes, hist_mask, Bin_N0);
            chistv2(h1, bytes, hist_mask, Bin_N1);
            vsts(h0, hist, 0, NORM_B16, full);
            vsts(h1, hist, 128, NORM_B16, full);
        }
        __tf__ __aicore__ void score_cache(__ubuf__ bfloat16_t* src, __ubuf__ bfloat16_t* weights,
        __ubuf__ uint16_t* cache, __ubuf__ uint16_t* hist, int column, int span0, int span1,
        int v00, int v01, int v10, int v11, int v20, int v21, bool have_dot,
        __ubuf__ uint16_t* pool_valid, bool gathered) {
            __VEC_SCOPE__ {
                RegTensor<bfloat16_t> s00, s01, s10, s11, s20, s21, d00, d01, d10, d11, d20, d21, w0, w1, w2;
                auto full = pset_b16(PAT_ALL);
                vbr(s00, bfloat16_t(0));
                vbr(s01, bfloat16_t(0));
                vbr(s10, bfloat16_t(0));
                vbr(s11, bfloat16_t(0));
                vbr(s20, bfloat16_t(0));
                vbr(s21, bfloat16_t(0));
                if (have_dot) {
#pragma unroll 4
                    for (uint16_t h = 0; h < 32; ++h) {
                        vlds(w0, weights, h, BRC_B16);
                        vlds(w1, weights, h + 32, BRC_B16);
                        vlds(w2, weights, h + 64, BRC_B16);
                        vlds(d00, src, h * 256, NORM);
                        vlds(d01, src, h * 256 + 128, NORM);
                        vlds(d10, src, (h + 32) * 256, NORM);
                        vlds(d11, src, (h + 32) * 256 + 128, NORM);
                        vlds(d20, src, (h + 64) * 256, NORM);
                        vlds(d21, src, (h + 64) * 256 + 128, NORM);
                        vmaxs(d00, d00, bfloat16_t(0), full, MODE_ZEROING);
                        vmaxs(d01, d01, bfloat16_t(0), full, MODE_ZEROING);
                        vmaxs(d10, d10, bfloat16_t(0), full, MODE_ZEROING);
                        vmaxs(d11, d11, bfloat16_t(0), full, MODE_ZEROING);
                        vmaxs(d20, d20, bfloat16_t(0), full, MODE_ZEROING);
                        vmaxs(d21, d21, bfloat16_t(0), full, MODE_ZEROING);
                        vmadd(d00, w0, s00, full, MODE_ZEROING);
                        vmadd(d01, w0, s01, full, MODE_ZEROING);
                        vmadd(d10, w1, s10, full, MODE_ZEROING);
                        vmadd(d11, w1, s11, full, MODE_ZEROING);
                        vmadd(d20, w2, s20, full, MODE_ZEROING);
                        vmadd(d21, w2, s21, full, MODE_ZEROING);
                        s00 = d00;
                        s01 = d01;
                        s10 = d10;
                        s11 = d11;
                        s20 = d20;
                        s21 = d21;
                    }
                }
                write_cached_row(s00, s01, cache+column, hist, v00, v01, span0, span1, column == 0, pool_valid, gathered);
                write_cached_row(s10, s11, cache+25600+column, hist+256, v10, v11, span0, span1, column == 0, pool_valid, gathered);
                write_cached_row(s20, s21, cache+51200+column, hist+512, v20, v21, span0, span1, column == 0, pool_valid, gathered);
            }
        }
        __aicore__ inline void threshold_bin(vector_u16 h0, vector_u16 h1, vector_u16& rank,
                                            vector_u16& bin, vector_u16& residual) {
            vector_u16 i0, i1, c0, c1, previous0, previous1, zero, missing;
            vector_bool ge0, ge1, before0, before1;
            auto full = pset_b16(PAT_ALL);
            vci((vector_s16&)i0, int16_t(0), INC_ORDER);
            vci((vector_s16&)i1, int16_t(128), INC_ORDER);
            vbr(zero, uint16_t(0));
            vbr(missing, uint16_t(65535));
            vcmp_ge(ge0, h0, rank, full);
            vcmp_ge(ge1, h1, rank, full);
            vsel(c0, i0, missing, ge0);
            vsel(c1, i1, missing, ge1);
            vmin(c0, c0, c1, full, MODE_ZEROING);
            vcmin(c1, c0, full, MODE_ZEROING);
            vdup(bin, c1, full, POS_LOWEST, MODE_ZEROING);
            vcmp_lt(before0, i0, bin, full);
            vcmp_lt(before1, i1, bin, full);
            vsel(previous0, h0, zero, before0);
            vsel(previous1, h1, zero, before1);
            vmax(previous0, previous0, previous1, full, MODE_ZEROING);
            vcmax(previous1, previous0, full, MODE_ZEROING);
            vdup(previous0, previous1, full, POS_LOWEST, MODE_ZEROING);
            vcmp_eq(ge0, i0, bin, full);
            vcmp_eq(ge1, i1, bin, full);
            vsel(c0, h0, zero, ge0);
            vsel(c1, h1, zero, ge1);
            vmax(c0, c0, c1, full, MODE_ZEROING);
            vcmax(c1, c0, full, MODE_ZEROING);
            vdup(c0, c1, full, POS_LOWEST, MODE_ZEROING);
            vsub(residual, c0, rank, full, MODE_ZEROING);
            vsub(rank, rank, previous0, full, MODE_ZEROING);
        }
        __aicore__ inline void find_threshold(__ubuf__ uint16_t* keys, int width,
                                              __ubuf__ uint16_t* scratch, bool cached,
                                              __ubuf__ uint16_t* cached_hist,
                                              vector_u16& key_cutoff, vector_u16& key_boundary) {
            vector_u16 h0, h1, h2, h3, rank, bin, residual, cutoff;
            vector_u8 lo, hi, prefix;
            vector_bool selected;
            auto full = pset_b16(PAT_ALL);
            if (cached) {
                vlds(h0, cached_hist, 0, NORM);
                vlds(h1, cached_hist, 128, NORM);
            } else {
                vbr(h0, uint16_t(0));
                vbr(h1, uint16_t(0));
                vbr(h2, uint16_t(0));
                vbr(h3, uint16_t(0));
                uint32_t remaining = width;
                for (uint16_t c = 0; c < (width + 255) / 256; c += 2) {
                    auto mask = CreatePredicate<uint8_t>(remaining);
                    vlds(lo, hi, (__ubuf__ uint8_t*)keys, c * 512, DINTLV_B8);
                    chistv2(h0, hi, mask, Bin_N0);
                    chistv2(h1, hi, mask, Bin_N1);
                    if (c + 1 < (width + 255) / 256) {
                        mask = CreatePredicate<uint8_t>(remaining);
                        vlds(lo, hi, (__ubuf__ uint8_t*)keys, (c + 1) * 512, DINTLV_B8);
                        chistv2(h2, hi, mask, Bin_N0);
                        chistv2(h3, hi, mask, Bin_N1);
                    }
                }
                vadd(h0, h0, h2, full, MODE_ZEROING);
                vadd(h1, h1, h3, full, MODE_ZEROING);
            }
            vbr(rank, uint16_t(width - 511));
            threshold_bin(h0, h1, rank, bin, residual);
            vpack(prefix, bin, LOWER);
            auto full8 = pset_b8(PAT_ALL);
            vdup(prefix, prefix, full8, POS_LOWEST, MODE_ZEROING);
            vshls(cutoff, bin, 8, full, MODE_ZEROING);
            vbr(h0, uint16_t(0));
            vbr(h1, uint16_t(0));
            vbr(h2, uint16_t(0));
            vbr(h3, uint16_t(0));
            uint32_t remaining = width;
            for (uint16_t c = 0; c < (width + 255) / 256; c += 2) {
                auto mask = CreatePredicate<uint8_t>(remaining);
                vlds(lo, hi, (__ubuf__ uint8_t*)keys, c * 512, DINTLV_B8);
                vcmp_eq(selected, hi, prefix, mask);
                chistv2(h0, lo, selected, Bin_N0);
                chistv2(h1, lo, selected, Bin_N1);
                if (c + 1 < (width + 255) / 256) {
                    mask = CreatePredicate<uint8_t>(remaining);
                    vlds(lo, hi, (__ubuf__ uint8_t*)keys, (c + 1) * 512, DINTLV_B8);
                    vcmp_eq(selected, hi, prefix, mask);
                    chistv2(h2, lo, selected, Bin_N0);
                    chistv2(h3, lo, selected, Bin_N1);
                }
            }
            vadd(h0, h0, h2, full, MODE_ZEROING);
            vadd(h1, h1, h3, full, MODE_ZEROING);
            threshold_bin(h0, h1, rank, bin, residual);
            vor(cutoff, cutoff, bin, full, MODE_ZEROING);
            vector_u16 key, lane, squeezed;
            vector_align align;
            auto out = scratch;
            sprclr(SPR_AR);
            remaining = width;
            for (uint16_t c = 0; c < (width + 127) / 128; ++c) {
                auto mask = CreatePredicate<uint16_t>(remaining);
                vlds(key, keys, c * 128, NORM);
                vci((vector_s16&)lane, int16_t(c * 128), INC_ORDER);
                vcmp_eq(selected, key, cutoff, mask);
                vsqz(squeezed, lane, selected, MODE_STORED);
                vstur(align, squeezed, out, POST_UPDATE);
            }
            vstar(align, out);
            mem_bar(VST_VLD);
            vector_u32 index, shift, packed;
            auto full32 = pset_b32(PAT_ALL);
            vcvt(index, residual, full, PART_EVEN);
            vbr(shift, 1u);
            vand(shift, shift, index, full32, MODE_ZEROING);
            vshls(shift, shift, 4, full32, MODE_ZEROING);
            vshrs(index, index, 1, full32, MODE_ZEROING);
            vgather2(packed, (__ubuf__ uint32_t*)scratch, index, full32);
            vshr(packed, packed, (vector_s32&)shift, full32, MODE_ZEROING);
            vpack(lane, packed, LOWER);
            key_cutoff = cutoff;
            vdup(key_boundary, lane, full, POS_LOWEST, MODE_ZEROING);
        }
        __tf__ __aicore__ void vector_threshold(__ubuf__ uint16_t* keys, int width,
                                               __ubuf__ uint16_t* scratch, __ubuf__ uint16_t* result) {
            __VEC_SCOPE__ {
                vector_u16 cutoff, boundary;
                find_threshold(keys, width, scratch, false, (__ubuf__ uint16_t*)0, cutoff, boundary);
                auto full = pset_b16(PAT_ALL);
                vsts(cutoff, result, 0, NORM_B16, full);
                vsts(boundary, result, 128, NORM_B16, full);
            }
        }
        __aicore__ inline void packed_threshold(__ubuf__ uint16_t* keys, int width, int scratch, int hist_base,
        uint32_t& cutoff, int& boundary) {
            cutoff = 0x007f;
            boundary = -1;
            if (width>512) {
                vector_threshold(keys, width, ub<uint16_t>(scratch), ub<uint16_t>(hist_base));
                set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
                cutoff = ub<uint16_t>(hist_base)[0];
                boundary = ub<uint16_t>(hist_base)[128];
                set_flag(PIPE_S, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
            }
        }
        __tf__ __aicore__ void packed_select(__ubuf__ uint16_t* outk, __ubuf__ uint32_t* outi,
        __ubuf__ uint16_t* src, __ubuf__ uint32_t* ids, __ubuf__ uint32_t* table,
        int width, uint32_t cutoff, int boundary, bool gathered, bool map_output) {
            auto result = outi;
            __VEC_SCOPE__ {
                vector_u32 key, lane, logical, tmp, page, squeezed, invalid, shift;
                vector_u16 key16, zero16;
                vector_bool eq, strict, early, keep, valid, packed_mask;
                vector_align align;
                auto full = pset_b32(PAT_ALL);
                auto full16 = pset_b16(PAT_ALL);
                vbr(invalid, 0xffffffffu);
                vbr(zero16, uint16_t(0));
                for (uint16_t c = 0;c<8;++c)vsts(invalid, outi, c*64, NORM_B32, full);
                for (uint16_t c = 0;c<4;++c)vsts(zero16, outk, c*128, NORM_B16, full16);
                mem_bar(VST_VST);
                sprclr(SPR_AR);
                uint32_t remaining = width;
                for (uint16_t c = 0;c<(width+63)/64;++c) {
                    auto mask = CreatePredicate<uint32_t>(remaining);
                    vlds((vector_u16&)key, src, c*64, UNPK_B16);
                    vci((vector_s32&)lane, int32_t(c*64), INC_ORDER);
                    vcmps_gt(strict, key, cutoff, mask);
                    vcmps_eq(eq, key, cutoff, mask);
                    vcmps_le(early, (vector_s32&)lane, boundary, mask);
                    pand(eq, eq, early, mask);
                    por(keep, strict, eq, mask);
                    vcmps_gt(valid, key, 0x007fu, mask);
                    pand(keep, keep, valid, mask);
                    {
                        vlds(logical, ids, c*64, NORM);
                    }
                    vsqz(squeezed, logical, keep, MODE_STORED);
                    vstur(align, squeezed, outi, POST_UPDATE);
                }
                vstar(align, outi);
                // Map only the selected rows, retaining their logical order.
                if (map_output) {
                    mem_bar(VST_VLD);
                    for (uint16_t c = 0; c < 8; ++c) {
                        vlds(logical, result, c * 64, NORM);
                        vcmps_ne(valid, logical, 0xffffffffu, full);
                        if (gathered) {
                            vgather2(page, table, logical, valid);
                            logical = page;
                        } else {
                            vshrs(tmp, logical, 7, full, MODE_ZEROING);
                            vgather2(page, table, tmp, valid);
                            vshls(page, page, 7, full, MODE_ZEROING);
                            vbr(tmp, 127u);
                            vand(tmp, logical, tmp, full, MODE_ZEROING);
                            vadd(logical, page, tmp, full, MODE_ZEROING);
                        }
                        vsel(logical, logical, invalid, valid);
                        vsts(logical, result, c * 64, NORM_B32, full);
                    }
                }
            }
        }
        __aicore__ inline void select_local(__ubuf__ uint16_t* outk, __ubuf__ uint32_t* outi,
                                            __ubuf__ uint16_t* src, int width, vector_u16& cutoff,
                                            vector_u16& boundary, int logical_begin) {
            vector_u16 key, lane, squeezed, zero;
            vector_bool eq, strict, early, keep, valid;
            vector_align align;
            auto full = pset_b16(PAT_ALL);
            auto full32 = pset_b32(PAT_ALL);
            auto packed = (__ubuf__ uint16_t*)0;
            vbr(zero, uint16_t(65535));
            for (uint16_t c = 0; c < 4; ++c) vsts(zero, packed, c * 128, NORM_B16, full);
            mem_bar(VST_VST);
            sprclr(SPR_AR);
            uint32_t remaining = width;
            for (uint16_t c = 0; c < (width + 127) / 128; ++c) {
                auto mask = CreatePredicate<uint16_t>(remaining);
                vlds(key, src, c * 128, NORM);
                vci((vector_s16&)lane, int16_t(c * 128), INC_ORDER);
                vcmp_gt(strict, key, cutoff, mask);
                vcmp_eq(eq, key, cutoff, mask);
                vcmp_le(early, (vector_s16&)lane, (vector_s16&)boundary, mask);
                pand(eq, eq, early, mask);
                por(keep, strict, eq, mask);
                vcmps_gt(valid, key, uint16_t(0x007f), mask);
                pand(keep, keep, valid, mask);
                vsqz(squeezed, lane, keep, MODE_STORED);
                vstur(align, squeezed, packed, POST_UPDATE);
            }
            vstar(align, packed);
            mem_bar(VST_VLD);
            vector_u16 interleaved, unused;
            vector_u32 index, local, gathered, shift, invalid;
            vbr(invalid, 0xffffffffu);
            for (uint16_t c = 0; c < 8; ++c) {
                vlds(lane, (__ubuf__ uint16_t*)0, c * 64, NORM);
                vintlv(interleaved, unused, lane, lane);
                vcvt(local, interleaved, full, PART_EVEN);
                vcmps_ne(valid, local, 65535u, full32);
                vadds(index, local, uint32_t(logical_begin), full32, MODE_ZEROING);
                vsel(index, index, invalid, valid);
                vsts(index, outi, c * 64, NORM_B32, full32);
                vbr(shift, 1u);
                vand(shift, local, shift, full32, MODE_ZEROING);
                vshls(shift, shift, 4, full32, MODE_ZEROING);
                vshrs(local, local, 1, full32, MODE_ZEROING);
                vbr(gathered, 0u);
                vgather2(gathered, (__ubuf__ uint32_t*)src, local, valid);
                vshr(gathered, gathered, (vector_s32&)shift, full32, MODE_ZEROING);
                vpack(key, gathered, LOWER);
                vector_bool packed_mask;
                ppack(packed_mask, full32, LOWER);
                vsts(key, outk, c * 64, NORM_B16, packed_mask);
            }
        }
        __tf__ __aicore__ void local_select(__ubuf__ uint16_t* outk, __ubuf__ uint32_t* outi,
                                           __ubuf__ uint16_t* keys, int width, int logical_begin,
                                           __ubuf__ uint16_t* cached_hist) {
            __VEC_SCOPE__ {
                vector_u16 cutoff, boundary;
                if (width > 512) {
                    find_threshold(keys, width, (__ubuf__ uint16_t*)0, true, cached_hist, cutoff, boundary);
                    mem_bar(VLD_VST);
                } else {
                    vbr(cutoff, uint16_t(0x007f));
                    vbr(boundary, uint16_t(65535));
                }
                select_local(outk, outi, keys, width, cutoff, boundary, logical_begin);
            }
        }
        __aicore__ inline void local_topk(__gm__ int64_t* args, int begin, int count, int logical, int width, int shard) {
            int shards = args[21];
            for (int row = 0;row<3;++row) {
                int group_offset = get_subblockid()*3+row;
                if (group_offset >= count)continue;
                auto keys = ub<uint16_t>(98304+row*51200);
                CandidateKeys selected_keys;
                CandidateIndices selected_indices;
                TASSIGN(selected_keys, 51200 + row * 3072);
                TASSIGN(selected_indices, 52224 + row * 3072);
                local_select(selected_keys.data(), selected_indices.data(), keys, width, logical,
                             ub<uint16_t>(251904 + row * 512));
            }
            // Publish all local query rows through one strided DMA per output.
            int first = get_subblockid() * 3;
            int rows = minimum(3, count - first);
            if (rows <= 0) return;
            Tile<TileType::Vec, uint16_t, 3, 1536, BLayout::RowMajor, DYNAMIC, 512> keys(rows);
            Tile<TileType::Vec, uint32_t, 3, 768, BLayout::RowMajor, DYNAMIC, 512> indices(rows);
            TASSIGN(keys, 51200);
            TASSIGN(indices, 52224);
            int index = ((begin + first) * shards + shard) * 512;
            GM<uint16_t, DYNAMIC, 512, DYNAMIC, DYNAMIC> kg(data<uint16_t>(args, 10) + index,
                TileShape2D<uint16_t, DYNAMIC, 512>(rows, 512), BaseShape2D<uint16_t, DYNAMIC, DYNAMIC>(rows, shards * 512));
            GM<uint32_t, DYNAMIC, 512, DYNAMIC, DYNAMIC> ig(data<uint32_t>(args, 11) + index,
                TileShape2D<uint32_t, DYNAMIC, 512>(rows, 512), BaseShape2D<uint32_t, DYNAMIC, DYNAMIC>(rows, shards * 512));
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(kg, keys);
            TSTORE(ig, indices);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
        __tf__ __aicore__ void compact_keys(__ubuf__ uint16_t* out, __ubuf__ uint16_t* keys,
                                            int width, uint32_t cutoff, int boundary) {
            __VEC_SCOPE__ {
                vector_u16 key, selected;
                vector_s16 lane;
                vector_bool strict, equal, early, keep, valid;
                vector_align align;
                sprclr(SPR_AR);
                uint32_t remaining = width;
                for (uint16_t c = 0; c < (width + 127) / 128; ++c) {
                    auto mask = CreatePredicate<uint16_t>(remaining);
                    vlds(key, keys, c * 128, NORM);
                    vci(lane, int16_t(c * 128), INC_ORDER);
                    vcmps_gt(strict, key, uint16_t(cutoff), mask);
                    vcmps_eq(equal, key, uint16_t(cutoff), mask);
                    vcmps_le(early, lane, int16_t(boundary), mask);
                    pand(equal, equal, early, mask);
                    por(keep, strict, equal, mask);
                    vcmps_gt(valid, key, uint16_t(0x007f), mask);
                    pand(keep, keep, valid, mask);
                    vsqz(selected, key, keep, MODE_STORED);
                    vstur(align, selected, out, POST_UPDATE);
                }
                vstar(align, out);
            }
        }
        __tf__ __aicore__ void unpack_scores(__ubuf__ bfloat16_t* dst, __ubuf__ uint16_t* src) {
            __VEC_SCOPE__ {
                vector_u16 key, positive, negative, bits, sign;
                vector_bool mask;
                auto full = pset_b16(PAT_ALL);
                vbr(sign, uint16_t(0x8000));
                for (uint16_t c = 0; c < 2; ++c) {
                    vlds(key, src, c * 128, NORM);
                    vcmps_ge(mask, key, uint16_t(0x8000), full);
                    vxor(positive, key, sign, full, MODE_ZEROING);
                    vnot(negative, key, full, MODE_ZEROING);
                    vsel(bits, positive, negative, mask);
                    vsts(bits, (__ubuf__ uint16_t*)dst, c * 128, NORM_B16, full);
                }
            }
        }
        __tf__ __aicore__ void block_maxima(__ubuf__ float* dst, __ubuf__ uint16_t* src, int width) {
            __VEC_SCOPE__ {
                vector_u32 keys, maxima, positive, negative, bits, sign;
                vector_bool positive_mask;
                auto full = pset_b32(PAT_ALL);
                uint32_t count = 8;
                auto reduced = CreatePredicate<uint32_t>(count);
                vbr(sign, 0x8000u);
                for (uint16_t c = 0; c < (width + 63) / 64; ++c) {
                    vlds((vector_u16&)keys, src, c * 64, UNPK_B16);
                    vcgmax(maxima, keys, full);
                    vcmps_ge(positive_mask, maxima, 0x8000u, reduced);
                    vxor(positive, maxima, sign, reduced, MODE_ZEROING);
                    vnot(negative, maxima, reduced, MODE_ZEROING);
                    vsel(bits, positive, negative, positive_mask);
                    vshls(bits, bits, 16, reduced, MODE_ZEROING);
                    vsts((vector_f32&)bits, dst, c * 8, NORM_B32, reduced);
                }
            }
        }
        __aicore__ inline void store_scores(__gm__ int64_t* args, int begin, int count,
                                            int logical, int cached_width) {
            int width = args[15];
            Vec<bfloat16_t, 1, 256> rounded;
            Vec<float, 1, 256> values;
            TASSIGN(rounded, 0);
            TASSIGN(values, 512);
            for (int row = 0; row < 3; ++row) {
                int offset = get_subblockid() * 3 + row;
                if (offset >= count) continue;
                if (args[20] == 2) {
                    block_maxima(ub<float>(4096), ub<uint16_t>(98304 + row * 51200), cached_width);
                } else {
                    for (int column = 0; column < cached_width; column += 256) {
                    unpack_scores(rounded.data(), ub<uint16_t>(98304 + row * 51200) + column);
                    TCVT(values, rounded, RoundMode::CAST_RINT);
                    {
                        int valid = minimum(256, cached_width - column);
                        Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, DYNAMIC> output(valid);
                        TASSIGN(output, 512);
                        GM<float, 1, DYNAMIC, 1, DYNAMIC> gm(
                            data<float>(args, 14) + (begin + offset) * width + logical + column,
                            TileShape2D<float, 1, DYNAMIC>(1, valid), BaseShape2D<float, 1, DYNAMIC>(1, width));
                        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                        TSTORE(gm, output);
                        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                    }
                    }
                }
                if (args[20] == 2) {
                    int blocks = (cached_width + 7) / 8;
                    Tile<TileType::Vec, float, 1, 3200, BLayout::RowMajor, 1, DYNAMIC> maxima(blocks);
                    TASSIGN(maxima, 4096);
                    GM<float, 1, DYNAMIC, 1, DYNAMIC> gm(
                        data<float>(args, 14) + (begin + offset) * width + logical / 8,
                        TileShape2D<float, 1, DYNAMIC>(1, blocks), BaseShape2D<float, 1, DYNAMIC>(1, width));
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    TSTORE(gm, maxima);
                    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                }
            }
        }
        __tf__ __aicore__ void short_page(__ubuf__ uint32_t* output, int base, int count) {
            __VEC_SCOPE__ {
                vector_u32 lane, physical, invalid, result;
                vector_bool valid;
                auto full = pset_b32(PAT_ALL);
                vbr(invalid, 0xffffffffu);
                for (uint16_t chunk = 0; chunk < 2; ++chunk) {
                    vci((vector_s32&)lane, int32_t(chunk * 64), INC_ORDER);
                    vadds(physical, lane, uint32_t(base), full, MODE_ZEROING);
                    vcmps_lt(valid, (vector_s32&)lane, count, full);
                    vsel(result, physical, invalid, valid);
                    vsts(result, output, chunk * 64, NORM_B32, full);
                }
            }
        }

        __aicore__ inline void write_short_topk(__gm__ int64_t* args, int token,
                                               int length, CandidateIndices& indices) {
            int table_width = args[17];
            int cache_pages = reinterpret_cast<__gm__ Tensor*>(args[3])->shapes[0];
            int request = read(data<int32_t>(args, 6) + token);
            for (int block = 0; block < 4; ++block) {
                int count = minimum(128, length - block * 128);
                int page = -1;
                if (count > 0 && block < table_width) {
                    page = read(data<int32_t>(args, 5) + request * table_width + block);
                }
                if (page < 0 || page >= cache_pages) {
                    page = 0;
                    count = 0;
                }
                short_page(indices.data() + block * 128, page * 128, count);
            }
            GM<uint32_t, 1, 512, 1, 512> out(data<uint32_t>(args, 8) + token * 512);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(out, indices);
            pipe_barrier(PIPE_ALL);
        }

        __aicore__ inline void merge(__gm__ int64_t* args) {
            int width = args[15], stride = args[16], table_width = args[17], tokens = args[18];
            int workers = get_block_num(args), worker = get_block_idx(args) * 2 + get_subblockid();
            int gathered_rows = args[19], shards = args[21];
            int panels = (width + 255) / 256, total = read(data<int32_t>(args, 9) + 3 * stride) * panels;
            CandidateKeys selected_keys;
            CandidateIndices selected_indices;
            TASSIGN(selected_keys, 131584);
            TASSIGN(selected_indices, 132608);
            for (int token = worker; token < tokens; token += workers * 2) {
                if (args[22]) {
                    int length = read(data<int32_t>(args, 7) + token);
                    if (length <= 512) {
                        write_short_topk(args, token, length, selected_indices);
                        continue;
                    }
                }
                GM<uint32_t, 1, 512, 1, 512> out(data<uint32_t>(args, 8) + token * 512);
                if (total == 0) {
                    TEXPANDS(selected_indices, 0xffffffffu);
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    TSTORE(out, selected_indices);
                    pipe_barrier(PIPE_ALL);
                    continue;
                }
                int group = read(data<int32_t>(args, 9) + 3 * stride + token + 1);
                int owners = minimum(shards, total);
                int first = ((group * panels + 1) * owners - 1) / total;
                int last = (((group + 1) * panels) * owners - 1) / total;
                int carry = 0;
                for (int start = first; start <= last;) {
                    int batch = minimum(last - start + 1, 32 - carry / 512);
                    int loaded = batch * 512, candidates = carry + loaded;
                    int index = (token * shards + start) * 512;
                    using KeyInput = Tile<TileType::Vec, uint16_t, 1, 16384, BLayout::RowMajor, 1, DYNAMIC>;
                    using IndexInput = Tile<TileType::Vec, uint32_t, 1, 16384, BLayout::RowMajor, 1, DYNAMIC>;
                    KeyInput ki(loaded);
                    IndexInput ii(loaded);
                    TASSIGN(ki, carry * 2);
                    TASSIGN(ii, 65536 + carry * 4);
                    GM<uint16_t, 1, DYNAMIC, 1, DYNAMIC> kg(data<uint16_t>(args, 10) + index,
                        TileShape2D<uint16_t, 1, DYNAMIC>(1, loaded), BaseShape2D<uint16_t, 1, DYNAMIC>(1, loaded));
                    GM<uint32_t, 1, DYNAMIC, 1, DYNAMIC> ig(data<uint32_t>(args, 11) + index,
                        TileShape2D<uint32_t, 1, DYNAMIC>(1, loaded), BaseShape2D<uint32_t, 1, DYNAMIC>(1, loaded));
                    TLOAD(ki, kg);
                    TLOAD(ii, ig);
                    bool final = start + batch > last;
                    auto mapping = ub<uint32_t>(134656);
                    if (final && gathered_rows) {
                        Tile<TileType::Vec, uint32_t, 1, 24576, BLayout::RowMajor, 1, DYNAMIC> rows(width);
                        TASSIGN(rows, 143360);
                        GM<uint32_t, 1, DYNAMIC, 1, DYNAMIC> pg(data<uint32_t>(args, 13) + token * gathered_rows,
                            TileShape2D<uint32_t, 1, DYNAMIC>(1, width), BaseShape2D<uint32_t, 1, DYNAMIC>(1, width));
                        TLOAD(rows, pg);
                        mapping = rows.data();
                    } else if (final) {
                        int request = read(data<int32_t>(args, 6) + token), page_count = (width + 127) / 128;
                        Tile<TileType::Vec, uint32_t, 1, 16384, BLayout::RowMajor, 1, DYNAMIC> pages(page_count);
                        TASSIGN(pages, 134656);
                        GM<uint32_t, 1, DYNAMIC, 1, DYNAMIC> pg(data<uint32_t>(args, 5) + request * table_width,
                            TileShape2D<uint32_t, 1, DYNAMIC>(1, page_count), BaseShape2D<uint32_t, 1, DYNAMIC>(1, table_width));
                        TLOAD(pages, pg);
                    }
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    uint32_t cutoff;
                    int boundary;
                    packed_threshold(ub<uint16_t>(0), candidates, 32768, 131072, cutoff, boundary);
                    packed_select(selected_keys.data(), selected_indices.data(), ub<uint16_t>(0),
                        ub<uint32_t>(65536), mapping, candidates, cutoff, boundary, gathered_rows != 0, final);
                    if (final) {
                        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                        TSTORE(out, selected_indices);
                    } else {
                        compact_keys(selected_keys.data(), ub<uint16_t>(0), candidates, cutoff, boundary);
                        CandidateKeys retained_keys;
                        CandidateIndices retained_indices;
                        TASSIGN(retained_keys, 0);
                        TASSIGN(retained_indices, 65536);
                        TMOV(retained_keys, selected_keys);
                        TMOV(retained_indices, selected_indices);
                        carry = 512;
                    }
                    pipe_barrier(PIPE_ALL);
                    start += batch;
                }
            }
        }
#endif
        __aicore__ inline void run(__gm__ int64_t* args) {
            int width = args[15], stride = args[16], table_width = args[17];
            int gathered_rows = args[19];
            int cache_pages = reinterpret_cast<__gm__ Tensor*>(args[3])->shapes[0];
            int groups = read(data<int32_t>(args, 9) + 3 * stride);
            int panels = (width + 255) / 256;
            int block = get_block_idx(args), workers = get_block_num(args);
            int owners = minimum(int(args[21]), groups * panels);
            int transferred = 0;
            for (int shard = block; shard < owners; shard += workers) {
                int first = groups * panels * shard / owners;
                int end = groups * panels * (shard + 1) / owners;
                for (int cursor = first; cursor < end;) {
                    int group_id = cursor / panels;
                    int finish = minimum(end, (group_id + 1) * panels);
                    int first_panel = cursor % panels;
                    int panel_count = finish - cursor;
                    int begin = read(data<int32_t>(args, 9) + group_id);
                    int count = read(data<int32_t>(args, 9) + stride + group_id) - begin;
                    int maximum = read(data<int32_t>(args, 9) + 2 * stride + group_id);
                    int request = read(data<int32_t>(args, 6) + begin);
#if defined(__DAV_CUBE__)
                    Tile<TileType::Mat, F4, 192, 128, BLayout::ColMajor, DYNAMIC, 128, SLayout::RowMajor> query(count * 32);
                    TASSIGN(query, 0);
                    GM<F4, DYNAMIC, 128, DYNAMIC, 128> qdata(data<F4>(args, 0) + begin * 2048,
                        TileShape2D<F4, DYNAMIC, 128>(count * 32, 128),
                        BaseShape2D<F4, DYNAMIC, 128>(count * 32, 128));
                    // Query and scale L1 slots are reused across fragments.
                    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
                    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
                    TLOAD(query, qdata);
                    QueryScale query_scale(count * 32);
                    TASSIGN(query_scale, 83200);
                    using QS = TileShape2D<S, DYNAMIC, 4, Layout::MX_A_ND>;
                    using QD = BaseShape2D<S, DYNAMIC, 4, Layout::MX_A_ND>;
                    GlobalTensor<S, QS, QD, Layout::MX_A_ND> qg(data<S>(args, 1) + begin * 128,
                    QS(count * 32, 4), QD(count * 32, 4));
                    TLOAD(query_scale, qg);
                    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                    TileLeft<F4, 96, 128> left0, left1;
                    TileLeftScale<S, 96, 4> scale0, scale1;
                    TASSIGN(left0, 0);
                    TASSIGN(left1, 16384);
                    TASSIGN(scale0, GetScaleAddr(left0.data()));
                    TASSIGN(scale1, GetScaleAddr(left1.data()));
                    // Both query halves must finish MMAD before their L0A slots are replaced.
                    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
                    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
                    TEXTRACT(left0, query, 0, 0);
                    TEXTRACT(left1, query, 96, 0);
                    TEXTRACT(scale0, query_scale, 0, 0);
                    TEXTRACT(scale1, query_scale, 96, 0);
#else
                    prepare_weights(args, begin, count);
                    int offset = get_subblockid() * 3;
                    int visible0 = offset < count ? (gathered_rows ? maximum : minimum(read(data<int32_t>(args, 7) + begin + offset), width)) : 0;
                    int visible1 = offset + 1 < count ? (gathered_rows ? maximum : minimum(read(data<int32_t>(args, 7) + begin + offset + 1), width)) : 0;
                    int visible2 = offset + 2 < count ? (gathered_rows ? maximum : minimum(read(data<int32_t>(args, 7) + begin + offset + 2), width)) : 0;
#endif
                    for (int panel = 0; panel < panel_count; ++panel) {
                        int logical = (first_panel + panel) * 256;
                        int page0 = !gathered_rows && logical < maximum ? read(data<int32_t>(args, 5) + request * table_width + logical / 128) : -1;
                        int page1 = !gathered_rows && logical + 128 < maximum ? read(data<int32_t>(args, 5) + request * table_width + logical / 128 + 1) : -1;
                        if (gathered_rows) {
                            page0 = logical < maximum ? (begin * gathered_rows + logical) / 128 : -1;
                            page1 = logical + 128 < maximum ? (begin * gathered_rows + logical) / 128 + 1 : -1;
                        }
                        if (!gathered_rows) {
                            if (page0 >= cache_pages) page0 = -1;
                            if (page1 >= cache_pages) page1 = -1;
                        }
                        bool have_dot = page0 >= 0 || page1 >= 0;
#if defined(__DAV_CUBE__)
                        if (have_dot) {
                            using Key = Tile<TileType::Mat, F4, 128, 256, BLayout::RowMajor, 128, 256, SLayout::ColMajor>;
                            using KeyPart = Tile<TileType::Mat, F4, 128, 256, BLayout::RowMajor, 128, 128, SLayout::ColMajor>;
                            using Scale = Tile<TileType::Mat, S, 4, 256, BLayout::ColMajor, 4, 256, SLayout::ColMajor, 32>;
                            Key key;
                            Scale key_scale;
                            TileRight<F4, 128, 256> right;
                            TileRightScale<S, 4, 256> right_scale;
                            TileAcc<float, 96, 256> dot0, dot1;
                            int key_slot = transferred % 2;
                            int key_event = key_slot + 2;
                            TASSIGN(key, 65536 + key_slot * 65536);
                            TASSIGN(key_scale, 81920 + key_slot * 65536);
                            TASSIGN(right, key_slot * 16384);
                            TASSIGN(right_scale, GetScaleAddr(right.data()));
                            TASSIGN(dot0, 0);
                            TASSIGN(dot1, 98304);
                            if (transferred >= 2) wait_flag(PIPE_MTE1, PIPE_MTE2, key_event);
                            for (int piece = 0; piece < 2; ++piece) {
                                int page = piece == 0 ? page0 : page1;
                                if (page < 0) page = page0 >= 0 ? page0 : page1;
                                KeyPart part;
                                KeyScale part_scale;
                                TASSIGN(part, 65536 + key_slot * 65536 + piece * 4096);
                                TASSIGN(part_scale, 81920 + key_slot * 65536 + piece * 512);
                                GM<F4, 128, 128, 128, 128, Layout::DN> kg(data<F4>(args, 3) + static_cast<int64_t>(page) * 8192);
                                GM<S, 4, 128, 4, 128, Layout::MX_B_DN> sg(data<S>(args, 4) + page * 512);
                                TLOAD(part, kg);
                                TLOAD(part_scale, sg);
                            }
                            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                            if (transferred >= 2) wait_flag(PIPE_M, PIPE_MTE1, key_event);
                            TMOV(right, key);
                            TMOV(right_scale, key_scale);
                            set_flag(PIPE_MTE1, PIPE_MTE2, key_event);
                            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                            int slot = transferred % 2;
                            using Received = Tile<TileType::Vec, bfloat16_t, 96, 256, BLayout::RowMajor, 96, 256>;
                            using DotPart = Tile<TileType::Acc, float, 96, 256, BLayout::ColMajor, 96, 256, SLayout::RowMajor>;
                            if (transferred > 0) wait_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
                            TMATMUL_MX(dot0, left0, scale0, right, right_scale);
                            set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                            if (transferred > 0) wait_flag(PIPE_FIX, PIPE_M, EVENT_ID3);
                            TMATMUL_MX(dot1, left1, scale1, right, right_scale);
                            set_flag(PIPE_M, PIPE_MTE1, key_event);
                            set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
                            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                            if (transferred >= 2) wait_intra_block(PIPE_FIX, slot * 2 + 1);
                            {
                                Received received;
                                DotPart half;
                                TASSIGN(received, slot * 49152);
                                TASSIGN(half, 0);
                                TMOV<Received, DotPart, AccToVecMode::SingleModeVec0>(received, half);
                            }
                            set_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
                            set_intra_block(PIPE_FIX, slot * 2);
                            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
                            if (transferred >= 2) wait_intra_block(PIPE_FIX, slot * 2 + 17);
                            {
                                Received received;
                                DotPart half;
                                TASSIGN(received, slot * 49152);
                                TASSIGN(half, 98304);
                                TMOV<Received, DotPart, AccToVecMode::SingleModeVec1>(received, half);
                            }
                            set_flag(PIPE_FIX, PIPE_M, EVENT_ID3);
                            set_intra_block(PIPE_FIX, slot * 2 + 16);
                            ++transferred;
                        }
#else
                        if (have_dot) wait_intra_block(PIPE_V, (transferred % 2) * 2);
                        if (gathered_rows) {
                            Vec<int32_t, 1, 256> raw_valid;
                            Vec<uint16_t, 1, 256> pool_valid;
                            TASSIGN(raw_valid, 253440);
                            TASSIGN(pool_valid, 254464);
                            GM<int32_t, 1, 256, 1, 256> vg(data<int32_t>(args, 12) + begin * gathered_rows + logical);
                            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                            TLOAD(raw_valid, vg);
                            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                            TCVT(pool_valid, raw_valid, RoundMode::CAST_NONE);
                        }
                        int n0 = visible0 - logical;
                        int n1 = visible1 - logical;
                        int n2 = visible2 - logical;
                        if (n0<0)n0 = 0;
                        if (n1<0)n1 = 0;
                        if (n2<0)n2 = 0;
                        int v00 = page0 >= 0?minimum(n0, 128):0, v01 = page1 >= 0&&n0>128?minimum(n0-128, 128):0;
                        int v10 = page0 >= 0?minimum(n1, 128):0, v11 = page1 >= 0&&n1>128?minimum(n1-128, 128):0;
                        int v20 = page0 >= 0?minimum(n2, 128):0, v21 = page1 >= 0&&n2>128?minimum(n2-128, 128):0;
                        int span0 = minimum(128, width-logical), span1 = minimum(128, width-logical-128);
                        if (span0<0)span0 = 0;
                        if (span1<0)span1 = 0;
                        score_cache(ub<bfloat16_t>((transferred%2)*49152), ub<bfloat16_t>(257280+offset*64),
                        ub<uint16_t>(98304), ub<uint16_t>(251904), panel*256, span0, span1, v00, v01, v10, v11, v20, v21, have_dot, ub<uint16_t>(254464), gathered_rows != 0);
                        if (have_dot) {
                            set_intra_block(PIPE_V, (transferred%2)*2+1);
                            ++transferred;
                        }
#endif
                    }
#if defined(__DAV_CUBE__)
                    wait_intra_block(PIPE_FIX, 6);
                    wait_intra_block(PIPE_FIX, 22);
#else
                    int cached_width = minimum(panel_count * 256, width - first_panel * 256);
                    local_topk(args, begin, count, first_panel * 256, cached_width, shard);
                    if (args[20]) store_scores(args, begin, count, first_panel * 256, cached_width);
                    set_intra_block(PIPE_MTE3, 6);
#endif
                    cursor = finish;
                }
            }
#if defined(__DAV_CUBE__)
            if (transferred > 0) {
                for (int slot = 0; slot < minimum(2, transferred); ++slot) {
                    wait_flag(PIPE_MTE1, PIPE_MTE2, slot + 2);
                    wait_flag(PIPE_M, PIPE_MTE1, slot + 2);
                }
                wait_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
                wait_flag(PIPE_FIX, PIPE_M, EVENT_ID3);
            }
            for (int slot = 0; slot < minimum(2, transferred); ++slot) {
                wait_intra_block(PIPE_FIX, slot * 2 + 1);
                wait_intra_block(PIPE_FIX, slot * 2 + 17);
            }
#endif
        }
    }
    // namespace fused
}
// namespace
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    set_atomic_none();
#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
#else
    set_ctrl(sbitset0(get_ctrl(), pto::MAD_MODE_BIT));
#endif
    fused::run(args);
    SYNCALL<SyncCoreType::AIVOnly>();
#if defined(__DAV_VEC__)
    fused::merge(args);
#endif
}
