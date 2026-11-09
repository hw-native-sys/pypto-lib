/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "expert_routed_comm.hpp"

namespace {
constexpr int TOPK = 6;
constexpr int PACKETS = (ROW_BYTES + 479) / 480;
constexpr int PACKED_COLS = PACKETS * 256;
constexpr int PACKED_BYTES = PACKED_COLS * 2;
constexpr int RAW_SLOT_BYTES = PACKETS * 480;
constexpr int RAW_BASE = 32768;
constexpr int PACKED_BASE = RAW_BASE + SLOTS * RAW_SLOT_BYTES;
constexpr int DESCRIPTOR_TILE = 1024;
constexpr int ARRIVED = 0x3f800000;
using Acc = Tile<TileType::Vec, float, 1, WIDTH>;
using Packed = Tile<TileType::Vec, bfloat16_t, 1, PACKED_COLS>;
using GlobalPacked = GlobalTensor<bfloat16_t, Shape<1, 1, 1, 1, PACKED_COLS>,
                                 pto::Stride<PACKED_COLS, PACKED_COLS, PACKED_COLS, PACKED_COLS, 1>>;
using Flags = Tile<TileType::Vec, int32_t, TOPK * PACKETS, 8>;
using GlobalFlags = GlobalTensor<int32_t, Shape<1, 1, 1, TOPK * PACKETS, 8>,
                                pto::Stride<1, 1, 1, 128, 1>>;

AICORE inline void packRow(__ubuf__ uint32_t* input, __ubuf__ uint32_t* output)
{
    __VEC_SCOPE__ {
        vector_bool full = pset_b32(PAT_ALL);
        uint32_t count = 56;
        vector_bool tail = plt_b32(count, POST_UPDATE);
        for (uint16_t packet = 0; packet < static_cast<uint16_t>(PACKETS); ++packet) {
            vector_u32 first;
            vlds(first, input, packet * 120, NORM);
            vsts(first, output, packet * 128, NORM_B32, full);
        }
        for (uint16_t packet = 0; packet < static_cast<uint16_t>(PACKETS); ++packet) {
            vector_u32 second;
            vlds(second, input, packet * 120 + 64, NORM);
            vsts(second, output, packet * 128 + 64, NORM_B32, tail);
        }
    }
}

AICORE inline void loadSendRow(__gm__ bfloat16_t* source, int slot)
{
    Row row;
    TASSIGN(row, RAW_BASE + slot * RAW_SLOT_BYTES);
    GlobalRow global(source);
    const auto event = static_cast<event_t>(slot);
    wait_flag(PIPE_V, PIPE_MTE2, event);
    TLOAD(row, global);
    set_flag(PIPE_MTE2, PIPE_V, event);
}

AICORE inline void packSendRow(__gm__ bfloat16_t* target, int slot)
{
    const auto event = static_cast<event_t>(slot);
    wait_flag(PIPE_MTE2, PIPE_V, event);
    wait_flag(PIPE_MTE3, PIPE_V, event);
    Packed packed;
    TASSIGN(packed, PACKED_BASE + slot * PACKED_BYTES);
    packRow(reinterpret_cast<__ubuf__ uint32_t*>(RAW_BASE + slot * RAW_SLOT_BYTES),
            reinterpret_cast<__ubuf__ uint32_t*>(packed.data()));
    set_flag(PIPE_V, PIPE_MTE2, event);
    set_flag(PIPE_V, PIPE_MTE3, event);
    wait_flag(PIPE_V, PIPE_MTE3, event);
    copy_ubuf_to_gm_align_v2(reinterpret_cast<__gm__ void*>(target),
                            reinterpret_cast<__ubuf__ void*>(packed.data()),
                            0, 1, PACKED_BYTES, 0, PACKED_BYTES, PACKED_BYTES);
    set_flag(PIPE_MTE3, PIPE_V, event);
}

// Compact descriptors contain source rank, original token and top-k slot.
AICORE inline void prepareDescriptors(__gm__ int64_t* args, __ubuf__ int32_t* counts,
                                     __ubuf__ int32_t* metadata, int begin, int end)
{
    const int experts = args[8];
    const int stride = args[9];
    const int ranks = args[10];
    auto* routesGM = tensorData<int32_t>(args, 2);
    auto* descriptors = reinterpret_cast<__ubuf__ int32_t*>(20480);
    int compactBase = 0;
    for (int expert = 0; expert < experts; ++expert) {
        const int count = counts[expert];
        if (count > 0 && compactBase < end && compactBase + count > begin) {
            const int firstRow = begin > compactBase ? begin - compactBase : 0;
            const int lastRow = end < compactBase + count ? end - compactBase : count;
            auto* routes = loadMetadata<1024>(routesGM + expert * stride + firstRow, lastRow - firstRow, 14336);
            int sourceBase = 0;
            for (int peer = 0; peer < ranks; ++peer) {
                const int sourceEnd = sourceBase + metadata[peer * experts + expert];
                const int first = firstRow > sourceBase ? firstRow : sourceBase;
                const int last = lastRow < sourceEnd ? lastRow : sourceEnd;
                if (first < last) {
                    for (int row = first; row < last; ++row) {
                        const int index = (compactBase + row - begin) * 3;
                        const int route = routes[row - firstRow];
                        descriptors[index] = peer;
                        descriptors[index + 1] = route / TOPK;
                        descriptors[index + 2] = route % TOPK;
                    }
                }
                sourceBase = sourceEnd;
            }
            set_flag(PIPE_S, PIPE_MTE2, EVENT_ID4);
            wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID4);
        }
        compactBase += count;
    }
}

AICORE inline void sendRows(__gm__ int64_t* args, int worker, int senders, int receivers, int rank)
{
    const int experts = args[8];
    const int ranks = args[10];
    auto* counts = loadMetadata<384>(tensorData<int32_t>(args, 0), experts, 0);
    auto* metadata = loadMetadata<3072>(tensorData<int32_t>(args, 1), experts * ranks, 2048);
    auto* output = tensorData<bfloat16_t>(args, 3);
    auto* target = tensorData<bfloat16_t>(args, 6);
    auto* context = reinterpret_cast<__gm__ uint64_t*>(args[14]);
    int total = 0;
    for (int expert = 0; expert < experts; ++expert) total += counts[expert];
    const int quotient = total / senders;
    const int remainder = total % senders;
    const int begin = worker * quotient + (worker < remainder ? worker : remainder);
    const int rows = quotient + (worker < remainder ? 1 : 0);
    if (rows == 0) return;
    uint64_t windows[8];
    for (int peer = 0; peer < ranks; ++peer) {
        dcci(context + COMM_WINDOWS_WORD + peer, SINGLE_CACHE_LINE);
        windows[peer] = reinterpret_cast<uint64_t>(target) + context[COMM_WINDOWS_WORD + peer] - context[COMM_WINDOWS_WORD + rank];
    }
    set_loop_size_ubtoout((1u << 21) | 1u);
    using MarkerTile = Tile<TileType::Vec, int32_t, 1, SLOTS * PACKED_BYTES / 4,
                            BLayout::RowMajor, 1, DYNAMIC>;
    MarkerTile marker;
    TASSIGN(marker, PACKED_BASE);
    marker.SetValidCol((rows < SLOTS ? rows : SLOTS) * PACKED_BYTES / 4);
    TEXPANDS(marker, int32_t(ARRIVED));
    for (int slot = 0; slot < SLOTS; ++slot) {
        set_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(slot));
        set_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(slot));
    }
    const int chunks = (rows + DESCRIPTOR_TILE - 1) / DESCRIPTOR_TILE;
    if (chunks == 1) prepareDescriptors(args, counts, metadata, begin, begin + rows);
    auto* descriptors = reinterpret_cast<__ubuf__ int32_t*>(20480);
    int issued = 0;
    __gm__ bfloat16_t* previousTarget = target;
    // Send the rows needed by early receiver rounds before later token rounds.
    const int batches = (static_cast<int>(args[11]) + receivers - 1) / receivers;
    for (int batch = 0; batch < batches && issued < rows; ++batch) {
        const int firstChunk = (rank + batch) % chunks;
        for (int loop = 0; loop < chunks; ++loop) {
            const int chunk = (firstChunk + loop) % chunks;
            const int offset = chunk * DESCRIPTOR_TILE;
            const int count = rows - offset < DESCRIPTOR_TILE ? rows - offset : DESCRIPTOR_TILE;
            if (chunks > 1) prepareDescriptors(args, counts, metadata, begin + offset, begin + offset + count);
            int permStride = 1;
            if (count > 2) {
                permStride = count / 2 + 1;
                if ((count % 2 == 0) && (permStride % 2 == 0)) ++permStride;
            }
            int index = (rank * count / ranks) % count;
            for (int loopRow = 0; loopRow < count; ++loopRow) {
                const int token = descriptors[index * 3 + 1];
                if (token >= batch * receivers && token < (batch + 1) * receivers) {
                    const int peer = descriptors[index * 3];
                    const int route = token * TOPK + descriptors[index * 3 + 2];
                    auto* destination = reinterpret_cast<__gm__ bfloat16_t*>(windows[peer]) + route * PACKED_COLS;
                    loadSendRow(output + (begin + offset + index) * WIDTH, issued % SLOTS);
                    if (issued > 0) packSendRow(previousTarget, (issued - 1) % SLOTS);
                    previousTarget = destination;
                    ++issued;
                }
                index += permStride;
                if (index >= count) index -= count;
            }
        }
    }
    if (issued > 0) packSendRow(previousTarget, (issued - 1) % SLOTS);
    for (int slot = 0; slot < SLOTS; ++slot) {
        wait_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(slot));
        wait_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(slot));
    }
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
}

AICORE inline int readyPrefix(__gm__ bfloat16_t* source, int begin)
{
    constexpr int FLAG_CAPACITY = 1152;
    using FlagRows = Tile<TileType::Vec, int32_t, TOPK * PACKETS, 8,
                          BLayout::RowMajor, DYNAMIC, 8>;
    using FlagVector = Tile<TileType::Vec, int32_t, 1, FLAG_CAPACITY,
                            BLayout::RowMajor, 1, DYNAMIC>;
    using CompareBits = Tile<TileType::Vec, uint8_t, 1, 160>;
    using Global = GlobalTensor<int32_t, Shape<1, 1, 1, DYNAMIC, 8>, pto::Stride<1, 1, 1, 128, 1>>;
    const int flagCount = (TOPK - begin) * PACKETS * 8;
    const int compareCount = (flagCount + 63) / 64 * 64;
    FlagRows flags;
    FlagVector values;
    CompareBits bits;
    TASSIGN(flags, 122880);
    TASSIGN(values, 122880);
    TASSIGN(bits, 131072);
    flags.SetValidRow((TOPK - begin) * PACKETS);
    if (compareCount > flagCount) {
        using FlagTail = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, DYNAMIC>;
        FlagTail tail;
        TASSIGN(tail, 122880 + flagCount * sizeof(int32_t));
        tail.SetValidCol(compareCount - flagCount);
        TEXPANDS(tail, int32_t(0));
    }
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
    Global global(reinterpret_cast<__gm__ int32_t*>(source + begin * PACKED_COLS) + 120,
                  Global::Shape(1, 1, 1, (TOPK - begin) * PACKETS, 8));
    TLOAD(flags, global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
    values.SetValidCol(compareCount);
    TCMPS(bits, values, int32_t(ARRIVED), CmpMode::EQ);
    set_flag(PIPE_V, PIPE_S, EVENT_ID6);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID6);
    auto* masks = reinterpret_cast<__ubuf__ uint64_t*>(bits.data());
    int arrived = flagCount;
    for (int word = 0; word < compareCount / 64; ++word) {
        const int64_t firstInvalid = sff0(masks[word]);
        if (firstInvalid != -1) {
            arrived = word * 64 + firstInvalid;
            if (arrived > flagCount) arrived = flagCount;
            break;
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID6);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID6);
    return begin + arrived / (PACKETS * 8);
}

AICORE inline void clearToken(__gm__ bfloat16_t* source)
{
    Flags zeros;
    TASSIGN(zeros, 143360);
    GlobalFlags global(reinterpret_cast<__gm__ int32_t*>(source) + 120);
    TSTORE(global, zeros);
}

AICORE inline void loadContribution(__gm__ bfloat16_t* source, int slot)
{
    using Unpacked = Tile<TileType::Vec, bfloat16_t, PACKETS, 240>;
    using Global = GlobalTensor<bfloat16_t, Shape<1, 1, 1, PACKETS, 240>, pto::Stride<1, 1, 1, 256, 1>>;
    Unpacked input;
    TASSIGN(input, slot * PACKED_BYTES);
    Global global(source);
    const auto event = static_cast<event_t>(slot);
    wait_flag(PIPE_V, PIPE_MTE2, event);
    TLOAD(input, global);
    set_flag(PIPE_MTE2, PIPE_V, event);
}
AICORE inline void receiveTokens(__gm__ int64_t* args, int worker, int workers, int first, int last)
{
    if (first + worker >= last) return;
    auto* shared = tensorData<bfloat16_t>(args, 4);
    auto* result = tensorData<bfloat16_t>(args, 5);
    auto* target = tensorData<bfloat16_t>(args, 6);
    const int active = args[12];
    Flags zeros;
    TASSIGN(zeros, 143360);
    TEXPANDS(zeros, int32_t(0));
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
    Row sharedTile, outputTile;
    Acc temporary;
    TASSIGN(sharedTile, 24576);
    TASSIGN(outputTile, 40960);
    TASSIGN(temporary, 98304);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    int tokens[2] = {-1, -1};
    int consumed[2] = {0, 0};
    int next = first + worker;
    while (next < last || tokens[0] >= 0 || tokens[1] >= 0) {
        for (int lane = 0; lane < 2; ++lane) {
            Acc acc;
            TASSIGN(acc, 57344 + lane * WIDTH * sizeof(float));
            if (tokens[lane] < 0 && next < last) {
                tokens[lane] = next;
                next += workers;
                if (tokens[lane] < active) {
                    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
                    GlobalRow sharedGlobal(shared + tokens[lane] * WIDTH);
                    TLOAD(sharedTile, sharedGlobal);
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
                    TCVT(acc, sharedTile, RoundMode::CAST_NONE);
                    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
                    consumed[lane] = 0;
                } else {
                    consumed[lane] = TOPK;
                }
            }
        }
        for (int lane = 0; lane < 2; ++lane) {
            Acc acc;
            TASSIGN(acc, 57344 + lane * WIDTH * sizeof(float));
            const int token = tokens[lane];
            if (token < 0) continue;
            if (consumed[lane] < TOPK) {
                const int begin = consumed[lane];
                const int end = readyPrefix(target + token * TOPK * PACKED_COLS, begin);
                if (end == begin) continue;
                loadContribution(target + (token * TOPK + begin) * PACKED_COLS, begin % 2);
                for (int slot = begin; slot < end; ++slot) {
                    if (slot + 1 < end) loadContribution(target + (token * TOPK + slot + 1) * PACKED_COLS, (slot + 1) % 2);
                    Row input;
                    TASSIGN(input, slot % 2 * PACKED_BYTES);
                    const auto event = static_cast<event_t>(slot % 2);
                    wait_flag(PIPE_MTE2, PIPE_V, event);
                    TCVT(temporary, input, RoundMode::CAST_NONE);
                    TADD(acc, acc, temporary);
                    set_flag(PIPE_V, PIPE_MTE2, event);
                }
                consumed[lane] = end;
            }
            if (consumed[lane] == TOPK) {
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
                if (token < active) {
                    TCVT(outputTile, acc, RoundMode::CAST_RINT);
                    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID6);
                    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID6);
                    clearToken(target + token * TOPK * PACKED_COLS);
                } else {
                    TEXPANDS(outputTile, static_cast<bfloat16_t>(0));
                }
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
                GlobalRow out(result + token * WIDTH);
                TSTORE(out, outputTile);
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
                tokens[lane] = -1;
            }
        }
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args)
{
#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    const int block = get_block_idx(args);
    const int workers = get_block_num(args);
    const int tokens = args[11];
    const int active = args[12];
    int receivers = workers / 2 < 32 ? workers / 2 : 32;
    if (receivers > tokens) receivers = tokens;
    if (receivers == 0) receivers = 1;
    const int senders = workers - receivers;
    auto* context = reinterpret_cast<__gm__ uint64_t*>(args[14]);
    dcci(context, SINGLE_CACHE_LINE);
    const int rank = static_cast<uint32_t>(context[COMM_RANK_WORD]);
    dcci(context + COMM_WINDOWS_WORD + rank, SINGLE_CACHE_LINE);
    int early = active;
    if (active > receivers) {
        const int denominator = workers * receivers + senders * senders;
        early = (active * workers * receivers + denominator - 1) / denominator;
        early = ((early + receivers - 1) / receivers) * receivers;
        if (early > active) early = active;
    }
    if (block < senders) {
        sendRows(args, block, senders, receivers, rank);
        receiveTokens(args, block, senders, early, active);
    } else {
        receiveTokens(args, block - senders, receivers, 0, early);
    }
    // Inactive physical rows have no incoming contributions.
    if (active + block < tokens) {
        pipe_barrier(PIPE_ALL);
        Row zeros;
        TASSIGN(zeros, 0);
        TEXPANDS(zeros, static_cast<bfloat16_t>(0));
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        auto* result = tensorData<bfloat16_t>(args, 5);
        for (int token = active + block; token < tokens; token += workers) {
            GlobalRow out(result + token * WIDTH);
            TSTORE(out, zeros);
        }
        pipe_barrier(PIPE_ALL);
    }
#endif
}
