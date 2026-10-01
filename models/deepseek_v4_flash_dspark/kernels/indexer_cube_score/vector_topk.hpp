/* Copyright (c) PyPTO Contributors.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#pragma once

// UB ownership: pipe [0,2048), scales [2048,2560), accumulated leaf [4096,20480),
// indices [20480,36864), two pair buffers [36864,102400), merge scratch thereafter.
namespace dspark_fused {
using namespace pto;
constexpr float kNegativeInfinity = -3.4028234663852886e38f;
constexpr int kLeafScoreUb = 4096;
constexpr int kIndexUb = 20480;
constexpr int kPairAUb = 36864;
constexpr int kPairBUb = 69632;
constexpr int kIndexTmpUb = 102400;
constexpr int kMergeTmpUb = 103424;
constexpr int kMergeOutUb = 111616;

template <int Width, typename T = float>
using Vec = pto::Tile<pto::TileType::Vec, T, 1, Width, pto::BLayout::RowMajor,
                      1, Width, pto::SLayout::NoneBox>;
template <int Width>
using GlobalRow = pto::GlobalTensor<float, pto::Shape<1,1,1,1,Width>,
                                   pto::Stride<Width,Width,Width,Width,1>>;

#if defined(__DAV_VEC__)
__aicore__ inline void clear_leaf_scores() {
    Vec<4096> scores;
    TASSIGN(scores, kLeafScoreUb);
    TEXPANDS(scores, kNegativeInfinity);
    pipe_barrier(PIPE_V);
}

// Same TSORT32 and 4-way merge sequence as indexer_topk_half_leaf.
// A full 4096-candidate row is padded once during tile collection.
template <int Width>
__aicore__ inline void publish_top512_width(__gm__ float* output, int logical_begin) {
    Vec<Width> scores;
    Vec<Width, int32_t> indices_i;
    Vec<Width, uint32_t> indices_u;
    Vec<192> index_tmp;
    Vec<2 * Width> pairs_a, pairs_b;
    TASSIGN(scores, kLeafScoreUb);
    TASSIGN(indices_i, kIndexUb);
    TASSIGN(indices_u, kIndexUb);
    TASSIGN(index_tmp, kIndexTmpUb);
    TASSIGN(pairs_a, kPairAUb);
    TASSIGN(pairs_b, kPairBUb);
    TCI<Vec<Width, int32_t>, Vec<192>, int32_t, 0>(indices_i, logical_begin, index_tmp);
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
    pipe_barrier(PIPE_V);
    TSORT32(pairs_a, scores, indices_u);
    pipe_barrier(PIPE_V);
    TMRGSORT(pairs_b, pairs_a, 64);
    pipe_barrier(PIPE_V);
    TMRGSORT(pairs_a, pairs_b, 256);
    pipe_barrier(PIPE_V);
    constexpr int sorted_base = Width >= 2048 ? kPairBUb : kPairAUb;
    if constexpr (Width >= 2048) {
        TMRGSORT(pairs_b, pairs_a, 1024);
        pipe_barrier(PIPE_V);
    }
    constexpr bool needs_final_merge = Width == 1024 || Width == 4096;
    if constexpr (needs_final_merge) {
        Vec<1024> left, right;
        Vec<2048> merged, merge_tmp;
        TASSIGN(left, sorted_base);
        TASSIGN(right, sorted_base + Width * sizeof(float));
        TASSIGN(merged, kMergeOutUb);
        TASSIGN(merge_tmp, kMergeTmpUb);
        pto::MrgSortExecutedNumList consumed;
        TMRGSORT<Vec<2048>, Vec<2048>, Vec<1024>, Vec<1024>, false>(
            merged, consumed, merge_tmp, left, right);
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    Vec<1024> top;
    TASSIGN(top, needs_final_merge ? kMergeOutUb : sorted_base);
    GlobalRow<1024> out(output);
    TSTORE(out, top);
    // The next leaf may reuse the merge output and all UB staging.
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
}

__aicore__ inline void publish_top512(__gm__ float* output, int logical_begin, int valid) {
    if (valid <= 512) publish_top512_width<512>(output, logical_begin);
    else if (valid <= 1024) publish_top512_width<1024>(output, logical_begin);
    else if (valid <= 2048) publish_top512_width<2048>(output, logical_begin);
    else publish_top512_width<4096>(output, logical_begin);
}

template <typename Pipe>
__aicore__ inline void consume_score_tile(Pipe& pipe, int lane, int tile_offset,
                                          int valid, const int64_t physical_rows[8],
                                          __gm__ float* key_scale) {
    Vec<128> raw;
    TPOP<Pipe, Vec<128>, pto::TileSplitAxis::TILE_LEFT_RIGHT>(pipe, raw, lane);
    // TPOP emits MTE2->paired-AIC free credits automatically for tile payloads.
    // Keep its UB slot alive until this tile's vector multiplication is done.
    if (valid > 0) {
        for (int page = 0; page < 4; ++page) {
            Vec<32> scale_page;
            TASSIGN(scale_page, 2048 + page * 128);
            GlobalRow<32> global_scale(key_scale + physical_rows[lane * 4 + page]);
            TLOAD(scale_page, global_scale);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        Vec<128> scale, destination;
        TASSIGN(scale, 2048);
        TASSIGN(destination, kLeafScoreUb + tile_offset * sizeof(float));
        TMUL(destination, raw, scale);
        if (valid < 128) {
            pipe_barrier(PIPE_V);
            using Partial = pto::Tile<pto::TileType::Vec, float, 1, 128,
                                     pto::BLayout::RowMajor, 1, -1,
                                     pto::SLayout::NoneBox, 512, pto::PadValue::Null>;
            Partial partial(valid);
            TASSIGN(partial, kLeafScoreUb + tile_offset * sizeof(float));
            using Padded = pto::Tile<pto::TileType::Vec, float, 1, 128,
                                    pto::BLayout::RowMajor, 1, 128,
                                    pto::SLayout::NoneBox, 512, pto::PadValue::Min>;
            Padded padded;
            TASSIGN(padded, kLeafScoreUb + tile_offset * sizeof(float));
            TFILLPAD<pto::TFillPadMode::InPlace>(padded, partial);
            // Match the DSL half-leaf helper's finite sentinel. ISA Min padding
            // is IEEE -inf, while TopK publishes exactly 512 score/index pairs.
            pipe_barrier(PIPE_V);
            TMAXS(destination, destination, kNegativeInfinity);
        }
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
    } else {
        // Drain the issued TPOP before a later tile reuses the same UB slot.
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
    }
}
#endif
}  // namespace dspark_fused
