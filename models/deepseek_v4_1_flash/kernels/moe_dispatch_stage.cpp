// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// MoE payload staging and MX_A_ZZ scale unpacking.
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"

using namespace pto;

namespace {
constexpr int D = 5120;
constexpr int ScaleColumns = D / 32;
constexpr int StageRows = 8;

template <typename T, int R, int C>
using Vec = Tile<TileType::Vec, T, R, C>;
template <typename T, int R, int C>
using GM = GlobalTensor<T, TileShape2D<T, R, C>, BaseShape2D<T, R, C>>;

template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int index) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

#if defined(__DAV_VEC__) && !defined(__CPU_SIM)
template <typename Output, typename Packed>
__tf__ __aicore__ inline void unpack_scales(
    typename Output::TileDType __out__ output,
    typename Packed::TileDType __in__ packed, uint16_t row_base, uint16_t rows) {
    auto src = (__ubuf__ uint16_t*)__cce_get_tile_ptr(packed);
    auto dst = (__ubuf__ uint16_t*)__cce_get_tile_ptr(output);
    __VEC_SCOPE__ {
        RegTensor<uint16_t> lane, index, values;
        uint32_t pairs = ScaleColumns / 2;
        MaskReg mask = CreatePredicate<uint16_t>(pairs);
        vci((RegTensor<int16_t>&)lane, (int16_t)0, INC_ORDER);
        vshls(lane, lane, (uint16_t)4, mask, MODE_ZEROING);
        for (uint16_t row = 0; row < rows; ++row) {
            // Each 16-bit lane retains both adjacent scale bytes.
            vadds(index, lane, (uint16_t)(row_base + row), mask, MODE_ZEROING);
            vgather2(values, src, index, mask);
            vsts(values, dst, row * (ScaleColumns / 2), NORM_B16, mask);
        }
    }
}

__aicore__ inline void stage(__gm__ int64_t* args) {
    int first = get_block_idx(args) * StageRows;
    int capacity = reinterpret_cast<__gm__ Tensor*>(args[0])->shapes[0];
    int active = args[4];
    active = active < 0 ? 0 : (active > capacity ? capacity : active);
    int rows = active - first;
    rows = rows < 0 ? 0 : (rows > StageRows ? StageRows : rows);

    Vec<uint8_t, StageRows, D> payload;
    Vec<uint8_t, 1, 16 * ScaleColumns> packed;
    using Output = Tile<TileType::Vec, uint8_t, StageRows, ScaleColumns,
                        BLayout::RowMajor, DYNAMIC, ScaleColumns>;
    Output output(rows);
    TASSIGN(payload, 0);
    TASSIGN(packed, StageRows * D);
    TASSIGN(output, StageRows * D + 16 * ScaleColumns);

    GM<uint8_t, StageRows, D> input_gm(data<uint8_t>(args, 0) + first * D);
    GM<uint8_t, StageRows, D> payload_gm(data<uint8_t>(args, 2) + first * D);
    TLOAD(payload, input_gm);
    if (rows > 0) {
        GM<uint8_t, 1, 16 * ScaleColumns> packed_gm(
            data<uint8_t>(args, 1) + (first / 16) * 16 * ScaleColumns);
        TLOAD(packed, packed_gm);
    }
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(payload_gm, payload);

    if (rows > 0) {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        unpack_scales<Output, decltype(packed)>(output.data(), packed.data(), first % 16, rows);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        using Shape = TileShape2D<uint8_t, DYNAMIC, ScaleColumns>;
        using Stride = BaseShape2D<uint8_t, StageRows, ScaleColumns>;
        GlobalTensor<uint8_t, Shape, Stride> output_gm(
            data<uint8_t>(args, 3) + first * ScaleColumns, Shape(rows, ScaleColumns), Stride());
        TSTORE(output_gm, output);
    }
    pipe_barrier(PIPE_ALL);
}
#endif
} // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_VEC__) && !defined(__CPU_SIM)
    stage(args);
#endif
}
