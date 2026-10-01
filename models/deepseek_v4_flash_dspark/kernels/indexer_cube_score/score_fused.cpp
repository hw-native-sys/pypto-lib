/* Copyright (c) PyPTO Contributors.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
// Mixed compact-score transport and leaf-local TopK; Cube math lives in a separate header.
#include <cstdint>
#if defined(__DAV_C220_CUBE__) && !defined(__DAV_CUBE__)
#define __DAV_CUBE__
#endif
#if defined(__DAV_C220_VEC__) && !defined(__DAV_VEC__)
#define __DAV_VEC__
#endif
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"
#include "vector_topk.hpp"
#if defined(__DAV_CUBE__)
#include "cube_score_exact.hpp"
using dspark_exact::CubeScoreComputer;
#endif
using namespace pto;

template <typename T>
__aicore__ inline __gm__ T* tensor_data(__gm__ int64_t* args, int index) {
    auto desc = reinterpret_cast<__gm__ Tensor*>(args[index]);
    return reinterpret_cast<__gm__ T*>(desc->buffer.addr) + desc->start_offset;
}
__aicore__ inline int64_t min64(int64_t a, int64_t b) { return a < b ? a : b; }
__aicore__ inline int64_t max64(int64_t a, int64_t b) { return a > b ? a : b; }

// Tensor-first ABI: pairs, pipe, Q, coeff_hi, coeff_lo, K, Kscale, table,
// positions, lengths, constants. Scalars: max_leaves, table_stride.
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    constexpr int kLeaf = 8192;
    constexpr int kMaxCandidates = 262144;
    constexpr int kPairRowsPerQuery = 64;
    constexpr int kCoefficientStride = 64;
    const int worker = get_block_idx(args);
    const int workers = get_block_num(args);
    const auto position_desc = reinterpret_cast<__gm__ Tensor*>(args[8]);
    const int64_t query_count = position_desc->shapes[0];
    const int64_t max_leaves = args[11];
    const int64_t table_stride = args[12];
    auto pairs = tensor_data<float>(args, 0);
    auto pipe_gm = tensor_data<float>(args, 1) + worker * 2048;
    auto query_i8 = tensor_data<int8_t>(args, 2);
    auto coefficient_hi = tensor_data<half>(args, 3);
    auto coefficient_lo = tensor_data<half>(args, 4);
    auto keys = tensor_data<int8_t>(args, 5);
    auto scales = tensor_data<float>(args, 6);
    auto table = tensor_data<int32_t>(args, 7);
    auto positions = tensor_data<int32_t>(args, 8);
    auto lengths = tensor_data<int32_t>(args, 9);
    auto constants = tensor_data<float>(args, 10);
    for (int64_t q = 0; q < query_count; q += 16) {
        dcci(reinterpret_cast<__gm__ void*>(positions + q), cache_line_t::SINGLE_CACHE_LINE);
    }
    if (query_count) dcci(reinterpret_cast<__gm__ void*>(positions + query_count - 1), cache_line_t::SINGLE_CACHE_LINE);
    for (int64_t b = 0; b < query_count / 8; b += 16) {
        dcci(reinterpret_cast<__gm__ void*>(lengths + b), cache_line_t::SINGLE_CACHE_LINE);
    }
    if (query_count >= 8) dcci(reinterpret_cast<__gm__ void*>(lengths + query_count / 8 - 1), cache_line_t::SINGLE_CACHE_LINE);
    using Pipe = TPipe<0, Direction::DIR_C2V, 1024, 8, 2, false>;
    Pipe pipe(pipe_gm, 0, 0);
#if defined(__DAV_CUBE__)
    CubeScoreComputer computer(keys, constants);
#else
    const int lane = get_sub_block_id(args);
    set_mask_norm();
    set_vector_mask(-1, -1);
#endif
    // Query-major leaf assignment matches the existing forest and is identical on all paired cores.
    for (int64_t item = worker; item < query_count * max_leaves; item += workers) {
        const int64_t query = item / max_leaves;
        const int64_t leaf = item % max_leaves;
        const int64_t batch = query / 8;
        const int64_t visible = max64(0, min64(kMaxCandidates,
            min64(lengths[batch] / 4, (positions[query] + 1) / 4)));
        const int64_t leaf_begin = leaf * kLeaf;
        if (leaf_begin >= visible) continue;
        const int valid = min64(kLeaf, visible - leaf_begin);
        const int lane_span = ((valid + 255) / 256) * 128;
        // The block table is immutable during the task. Invalidate each
        // addressed cache line once per leaf, rather than once per tile/page.
        const int64_t first_page = batch * table_stride + leaf_begin / 32;
        const int64_t page_count = (valid + 31) / 32;
        const uint64_t first_line = reinterpret_cast<uint64_t>(table + first_page) & ~uint64_t(63);
        const uint64_t last_line = reinterpret_cast<uint64_t>(table + first_page + page_count - 1) & ~uint64_t(63);
        for (uint64_t line = first_line; line <= last_line; line += 64) {
            dcci(reinterpret_cast<__gm__ void*>(line), cache_line_t::SINGLE_CACHE_LINE);
        }
#if defined(__DAV_CUBE__)
        computer.set_query(query_i8 + query * 8192,
                           coefficient_hi + query * kCoefficientStride,
                           coefficient_lo + query * 128,
                           coefficient_lo + query * 128 + 64);
        int64_t first_rows[8];
        for (int page=0; page<8; ++page) {
            const int local=min64((page/4)*lane_span+(page%4)*32,((valid-1)/32)*32);
            const int64_t index=batch*table_stride+(leaf_begin+local)/32;
            first_rows[page]=static_cast<int64_t>(table[index])*32;
        }
        computer.prefetch(first_rows,0);

#else
        dspark_fused::clear_leaf_scores();
#endif
        for (int offset=0; offset<lane_span; offset+=128) {
#if defined(__DAV_CUBE__)
            const bool has_next=offset+128<lane_span;
            int64_t next_rows[8];
            if (has_next) {
                for (int page=0; page<8; ++page) {
                    const int local=min64(offset+128+(page/4)*lane_span+(page%4)*32,
                                          ((valid-1)/32)*32);
                    const int64_t index=batch*table_stride+(leaf_begin+local)/32;
                    next_rows[page]=static_cast<int64_t>(table[index])*32;
                }
            }
            auto& score=computer.compute((offset/128)&1,has_next ? next_rows : nullptr);
            set_flag(PIPE_M,PIPE_FIX,EVENT_ID7);
            wait_flag(PIPE_M,PIPE_FIX,EVENT_ID7);
            TPUSH<Pipe,dspark_exact::ScoreAcc,TileSplitAxis::TILE_LEFT_RIGHT>(pipe,score);
            set_flag(PIPE_FIX,PIPE_M,EVENT_ID7);
            wait_flag(PIPE_FIX,PIPE_M,EVENT_ID7);
#else
            int64_t physical_rows[8];
            for (int page=lane*4; page<lane*4+4; ++page) {
                const int local=min64(offset+(page/4)*lane_span+(page%4)*32,
                                      ((valid-1)/32)*32);
                const int64_t index=batch*table_stride+(leaf_begin+local)/32;
                physical_rows[page]=static_cast<int64_t>(table[index])*32;
            }
            const int lane_valid=max64(0,min64(128,valid-offset-lane*lane_span));
            dspark_fused::consume_score_tile(pipe,lane,offset,lane_valid,physical_rows,scales);
#endif
        }
#if defined(__DAV_CUBE__)
        computer.finish_leaf();
#endif
#if defined(__DAV_VEC__)
        const int64_t pair_row = query * kPairRowsPerQuery + leaf * 2 + lane;
        dspark_fused::publish_top512(pairs + pair_row * 1024,
                                      leaf_begin + lane * lane_span,
                                      max64(0, min64(lane_span, valid - lane * lane_span)));
#endif
    }
    pipe_barrier(PIPE_ALL);
}
