// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Sharded paged scores, local TopK, and candidate merge in one MIX kernel.
// Scalar simulation of the native BF16 score/FMA and stable TopK contract.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"

namespace index_score_sim {
using namespace pto;
template <typename T>
inline T* data(int64_t* args, int index) {
    auto* tensor = reinterpret_cast<Tensor*>(args[index]);
    return reinterpret_cast<T*>(tensor->buffer.addr) + tensor->start_offset;
}
inline float unpack(uint8_t byte, int index) {
    constexpr float grid[] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
    const unsigned code = (byte >> (4 * (index & 1))) & 15;
    return code & 8 ? -grid[code & 7] : grid[code & 7];
}
inline float scale(uint8_t code) {
    return code == 255 ? std::numeric_limits<float>::quiet_NaN() : std::ldexp(1.0f, int(code) - 127);
}
inline void run(int64_t* args) {
#if defined(__DAV_VEC__)
    const int width = args[15], runStride = args[16], tableWidth = args[17];
    const int tokens = args[18], gathered = args[19];
    const int pages = reinterpret_cast<Tensor*>(args[3])->shapes[0];
    const int workers = get_block_num(args) * 2;
    const int worker = get_block_idx(args) * 2 + get_subblockid();
    for (int token = worker; token < tokens; token += workers) {
        auto* output = data<int32_t>(args, 8) + token * 512;
        std::fill(output, output + 512, -1);
        const int length = std::clamp(data<int32_t>(args, 7)[token], 0, width);
        const int request = data<int32_t>(args, 6)[token];
        if (args[22] && length <= 512) {
            for (int logical = 0; logical < length; ++logical) {
                const int page = data<int32_t>(args, 5)[request * tableWidth + logical / 128];
                if (page >= 0 && page < pages) output[logical] = page * 128 + logical % 128;
            }
            if (!args[20]) continue;
        }
        const int group = data<int32_t>(args, 9)[3 * runStride + token + 1];
        const int begin = data<int32_t>(args, 9)[group];
        std::array<float, 32 * 128> query;
        std::array<float, 32> weights;
        for (int head = 0; head < 32; ++head) {
            weights[head] = static_cast<float>(static_cast<bfloat16_t>(
                static_cast<float>(data<bfloat16_t>(args, 2)[token * 32 + head]) * 0.015625f));
            for (int d = 0; d < 128; ++d) {
                const int index = (token * 32 + head) * 128 + d;
                query[head * 128 + d] = unpack(data<uint8_t>(args, 0)[index / 2], d) *
                    scale(data<uint8_t>(args, 1)[(token * 32 + head) * 4 + d / 32]);
            }
        }
        std::vector<float> scores(width, -std::numeric_limits<float>::infinity());
        std::vector<int> physical(width, -1), selected;
        for (int logical = 0; logical < (gathered ? width : length); ++logical) {
            int row;
            if (gathered) {
                if (data<int32_t>(args, 12)[begin * gathered + logical] != 1) continue;
                row = begin * gathered + logical;
                physical[logical] = data<int32_t>(args, 13)[token * gathered + logical];
            } else {
                const int page = data<int32_t>(args, 5)[request * tableWidth + logical / 128];
                if (page < 0 || page >= pages) continue;
                row = page * 128 + logical % 128;
                physical[logical] = row;
            }
            std::array<float, 128> key;
            for (int d = 0; d < 128; ++d) {
                key[d] = unpack(data<uint8_t>(args, 3)[row * 64 + d / 2], d) *
                    scale(data<uint8_t>(args, 4)[row * 4 + d / 32]);
            }
            bfloat16_t sum = static_cast<bfloat16_t>(0.0f);
            for (int head = 0; head < 32; ++head) {
                float dot = 0;
                for (int d = 0; d < 128; ++d) dot += query[head * 128 + d] * key[d];
                const float rounded = static_cast<float>(static_cast<bfloat16_t>(dot));
                // Both BF16 operands have an exact FP32 product. Keep one
                // BF16 rounding after adding the preceding BF16 partial.
                sum = static_cast<bfloat16_t>(std::fma(double(std::max(rounded, 0.0f)), double(weights[head]), double(sum)));
            }
            scores[logical] = static_cast<float>(sum);
            if (std::isfinite(scores[logical])) selected.push_back(logical);
        }
        if (args[20]) std::copy(scores.begin(), scores.end(), data<float>(args, 14) + token * width);
        if (args[22] && length <= 512) continue;
        const auto order = [&](int a, int b) {
            return scores[a] > scores[b] || (scores[a] == scores[b] && a < b);
        };
        const int keep = std::min<int>(512, selected.size());
        std::partial_sort(selected.begin(), selected.begin() + keep, selected.end(), order);
        selected.resize(keep);
        std::sort(selected.begin(), selected.end());
        for (int i = 0; i < keep; ++i) output[i] = physical[selected[i]];
    }
#endif
}
} // namespace index_score_sim
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    index_score_sim::run(args);
}
