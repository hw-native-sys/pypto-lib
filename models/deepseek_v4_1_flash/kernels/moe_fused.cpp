/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#define ASCENDC_DUMP 0
#define ASCENDC_TILING_KEY_VAR_IMPL 0
#include <utils/debug/asc_printf.h>
#include <utils/debug/asc_assert.h>
// Keep N256 resident for the 384-row decode entry, including short expert groups.
#define MOE_DECODE_K256 (MOE_TOKENS == 384)
#define MOE_STREAM_W2_SCALES (!MOE_PACKED_ROUTED_WEIGHTS && MOE_TOKENS <= 80 && \
    1ULL * MOE_TOKENS * (MOE_TOPK + 2) * MOE_HIDDEN * 2 <= 32ULL * 1024 * 1024)
/*
 * Operation-private DeepSeek V4.1 MoE pipeline port.
 *
 * The MIT notice below applies only to sections marked "DeepSeek MIT port"
 * in this file. Those sections are adapted from
 * https://github.com/deepseek-ai/DeepGEMM-Ascend at revision
 * 3a35326271b069b8b5bf1a754435acd6cc91575b.
 * Sections marked "PyPTO adapter" retain the CANN Open Software License
 * Agreement Version 2.0 and are outside this MIT notice's scope.
 *
 * MIT License
 *
 * Copyright (c) 2026 DeepSeek
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */


// BEGIN DeepSeek MIT port: include/deep_gemm/ascend.hpp

#include <c_api/asc_simd.h>
// BEGIN DeepSeek MIT port: include/deep_gemm/common.hpp

// CLion uses a regular C++ frontend, so load the CCE shim before any CANN header.
// The real device compiler does not define __CLION_IDE__ and is unaffected.
#ifdef __CLION_IDE__
#define __CCE_STUB_DISABLE_ADDRESS_SPACE_QUALIFIERS__
#define __CCE_STUB_USE_LOCAL_WRAPPER__
#define __CCE_AICORE_SUPPORT_SIMT__
#define __clang__
#include <cce_stubs/cce_stubs.h>
#endif

// Shared definitions used by BOTH host (csrc/) and device (deep_gemm/include) code.
// This header is pure C++: no <kernel_operator.h>, no intrinsics, so it can be included
// from host translation units. Keep it dependency-free.

#include <cstdint>
#include <utility>

// The device compiler (bisheng) defines __CCE__ and provides __aicore__ / __gm__. On the
// host these are not keywords, so define them away — this lets a handful of small types
// (e.g. gm_ptr below) be shared verbatim between host and device translation units.
#ifndef __CCE__
#define __aicore__
#define __gm__
#endif

// Keep both mixed-kernel paths visible to the host indexer.
#ifdef __CLION_IDE__
#define DG_IS_AIC constexpr (true)
#define DG_IS_AIV constexpr (true)
#else
#define DG_IS_AIC ASCEND_IS_AIC
#define DG_IS_AIV ASCEND_IS_AIV
#endif

namespace deep_gemm {

#ifdef __CCE__
template <typename T1, typename T2>
__aicore__ constexpr auto ceil_div(T1 a, T2 b) -> decltype(a + b) { return (a + b - 1) / b; }

template <typename T1, typename T2>
__aicore__ constexpr auto align(T1 a, T2 b) -> decltype(a + b) { return (a + b - 1) / b * b; }
#endif

template <typename T1, typename T2>
__aicore__ constexpr auto aligned(T1 a, T2 b) -> decltype(a + b) { return (a + b - 1) / b * b; }

// Workspace capacity is independent of the runtime-admitted core count.
inline constexpr uint32_t kNumAICores = 32;

template <bool kFullTopology>
struct CoreTopology {
    static constexpr uint32_t kMinCores = kFullTopology ? kNumAICores : 1;
    uint32_t active_cores;

    constexpr __aicore__ uint32_t cores() const {
        return kFullTopology ? kNumAICores : active_cores;
    }

    constexpr __aicore__ uint32_t vectors() const {
        return 2 * cores();
    }
};

inline constexpr uint32_t L1SizeBytes = 512 * 1024;   // L1 per core
inline constexpr uint32_t L0ASizeBytes = 64 * 1024;   // L0A
inline constexpr uint32_t L0BSizeBytes = 64 * 1024;   // L0B
inline constexpr uint32_t L0CSizeBytes = 256 * 1024;  // L0C (fp32 accumulator)
inline constexpr uint32_t UBSizeBytes = 256 * 1024;   // UB
inline constexpr uint32_t kDCacheLineBytes = 64;
inline constexpr uint32_t kL2SectorBytes = 128;
inline constexpr uint32_t kNumPipeEvents = 8;

inline constexpr uint32_t FRAC_MN = 16;      // NZ fractal M/N rows
inline constexpr uint32_t MX_ADDR_DIV = 16;  // MX dst address encoding divisor
inline constexpr uint32_t MX_SF_DIVISOR = 64;  // K-elements per SF pair

#ifdef __CCE__
// only bisheng has float4_e2m1x2_t defined; host code does not
template <typename T>
__aicore__ constexpr bool is_fp4() { return std::is_same_v<T, float4_e2m1x2_t>; }

template <typename T>
__aicore__ constexpr uint32_t get_frac_k() {
    if constexpr (is_fp4<T>()) {
        return 64;  // one MX SF pair
    } else {
        return 32 / sizeof(T);  // 32 bytes per C0 / K-elements per byte
    }
}
#endif

enum class Major : uint8_t { K, MN };

__aicore__ constexpr bool is_k_major(Major m) { return m == Major::K; }
__aicore__ constexpr bool is_mn_major(Major m) { return m == Major::MN; }

enum class GemmType : uint8_t {
    Normal,
    MGroupedContiguousWithPsumLayout,
    KGroupedContiguousWithPsumLayout,
    Batched,
};

// Runtime state shared by host classes and device operators.
namespace epilogue {
struct Args {
    float alpha = 1.0f;
    uintptr_t sfd = 0;
    uint64_t sfd_stride = 0;
    uint32_t shape_n = 0;
    uintptr_t bias = 0;
};
} // namespace epilogue

__aicore__ constexpr bool is_m_grouped(GemmType t) {
    return t == GemmType::MGroupedContiguousWithPsumLayout;
}
__aicore__ constexpr bool is_k_grouped(GemmType t) {
    return t == GemmType::KGroupedContiguousWithPsumLayout;
}
__aicore__ constexpr bool is_grouped(GemmType t) {
    return is_m_grouped(t) || is_k_grouped(t);
}
__aicore__ constexpr bool is_batched(GemmType t) {
    return t == GemmType::Batched;
}

#ifndef __CCE__
// Mirror C API enum names for host JIT formatting without Bisheng headers.
enum class asc_load_l2_cache_mode : uint8_t {
    NORMAL_FIRST_VICTIM = 0,
    NORMAL_LAST_VICTIM = 1,
    NORMAL_PERSISTENT = 2,
    NOTALLOC_KEEP = 4,
    NOTALLOC_CLEAN = 5,
    NOTALLOC_DROP = 6,
};

enum class asc_store_l2_cache_mode : uint8_t {
    NORMAL_FIRST_VICTIM = 0,
    NORMAL_LAST_VICTIM = 1,
    NORMAL_PERSISTENT = 2,
    NOTALLOC_CLEAN = 4,
};
#endif

template <typename T, Major kMajor = Major::K>
struct gm_ptr {
    uintptr_t addr;
    uint64_t stride_outer;
    uint64_t stride_batch;

    using ptr_t = __gm__ T*;
    using dtype_t = T;
    static constexpr Major major = kMajor;

    constexpr __aicore__ gm_ptr(uint64_t stride_outer, uintptr_t addr = 0, uint64_t stride_batch = 0) :
        addr(addr), stride_outer(stride_outer), stride_batch(stride_batch) {}

#ifdef __CCE__
    template <typename dst_dtype_t = dtype_t>
    __aicore__ __gm__ dst_dtype_t* ptr() const {
        return reinterpret_cast<__gm__ dst_dtype_t*>(addr);
    }

    constexpr __aicore__ gm_ptr index(uint64_t mn_idx, uint64_t k_idx) const {
        constexpr uint64_t pack_size = is_fp4<dtype_t>() ? 2 : 1;
        gm_ptr r = *this;
        if constexpr (kMajor == Major::K) {
            r.addr += sizeof(dtype_t) * (mn_idx * stride_outer + k_idx / pack_size);
        } else {
            r.addr += sizeof(dtype_t) * (k_idx * stride_outer + mn_idx / pack_size);
        }
        return r;
    }

    constexpr __aicore__ gm_ptr group(uint64_t group_idx, uint64_t shape_mn, uint64_t shape_k) const {
        constexpr uint64_t pack_size = is_fp4<dtype_t>() ? 2 : 1;
        gm_ptr r = *this;
        r.addr += group_idx * shape_mn * shape_k / pack_size * sizeof(dtype_t);
        return r;
    }
#endif

    constexpr __aicore__ gm_ptr operator+(uint64_t elems) const {
        gm_ptr r = *this;
        r.addr += elems * sizeof(dtype_t);
        return r;
    }

    constexpr __aicore__ gm_ptr batch(uint64_t idx) const {
        gm_ptr r = *this;
        r.addr += idx * stride_batch * sizeof(dtype_t);
        return r;
    }
};

} // namespace deep_gemm
// END DeepSeek MIT port: include/deep_gemm/common.hpp


// Device primitives, split by abstraction layer:
//   tile_ptr — operand_ptr / result_ptr (shape-carrying device pointers) + l*_ptr aliases
//   copy     — GM/L1/L0/UB data movement, scale-factor load, mad, quant-mode inference
//   sync     — intra-block / flag / ffts synchronization
// BEGIN DeepSeek MIT port: include/deep_gemm/ascend/tile_ptr.hpp

#include <c_api/asc_simd.h>

namespace deep_gemm {

template <typename T, typename dst_t> struct cvt_pointee { using type = dst_t*; };
template <typename T, typename dst_t> struct cvt_pointee<__gm__ T*, dst_t> { using type = __gm__ dst_t*; };
template <typename T, typename dst_t> struct cvt_pointee<__cbuf__ T*, dst_t> { using type = __cbuf__ dst_t*; };
template <typename T, typename dst_t> struct cvt_pointee<__ca__ T*, dst_t> { using type = __ca__ dst_t*; };
template <typename T, typename dst_t> struct cvt_pointee<__cb__ T*, dst_t> { using type = __cb__ dst_t*; };
template <typename T, typename dst_t> struct cvt_pointee<__cc__ T*, dst_t> { using type = __cc__ dst_t*; };
template <typename T, typename dst_t> struct cvt_pointee<__ubuf__ T*, dst_t> { using type = __ubuf__ dst_t*; };
template <typename ptr_t, typename dst_t>
using cvt_pointee_t = typename cvt_pointee<ptr_t, dst_t>::type;

template <typename ptr_t, typename elem_t, Major MAJOR = Major::K>
struct operand_ptr {
    uintptr_t addr;
    const uint64_t shape_mn, shape_k;  // GEMM operand dimensions (logical, layout-invariant)
    static constexpr Major major = MAJOR;

    using dtype_t = elem_t;

    // K-elements per type-slot: fp4 (float4_e2m1x2_t) packs 2, all others 1.
    static constexpr uint32_t k_per_slot = is_fp4<dtype_t>() ? 2u : 1u;

    constexpr __aicore__ operand_ptr(uint64_t shape_mn, uint64_t shape_k, uintptr_t addr = 0) :
        addr(addr), shape_mn(shape_mn), shape_k(shape_k) {}

    template <typename dst_dtype_t = dtype_t>
    __aicore__ cvt_pointee_t<ptr_t, dst_dtype_t> ptr() const {
        return reinterpret_cast<cvt_pointee_t<ptr_t, dst_dtype_t>>(addr);
    }

    constexpr __aicore__ uint64_t size_per_stage() const { return shape_mn * shape_k / k_per_slot * sizeof(dtype_t); }
    constexpr __aicore__ uintptr_t offset(uint32_t n) const { return addr + n * size_per_stage(); }
    constexpr __aicore__ operand_ptr<ptr_t, elem_t, MAJOR> operator[](uint32_t i) const {
        operand_ptr<ptr_t, elem_t, MAJOR> r = *this;
        r.addr = offset(i);
        return r;
    }

    // Byte-granular addressing for NZ fractal layouts (element/stage strides do not apply).
    constexpr __aicore__ operand_ptr<ptr_t, elem_t, MAJOR> offset_bytes(uint64_t bytes) const {
        operand_ptr<ptr_t, elem_t, MAJOR> r = *this;
        r.addr += bytes;
        return r;
    }

    constexpr __aicore__ operand_ptr<ptr_t, elem_t, MAJOR> as_mad_aligned(uint64_t shape_mn, uint64_t shape_k) const {
        constexpr auto FRAC_K = get_frac_k<dtype_t>();
        return operand_ptr<ptr_t, elem_t, MAJOR>(aligned(shape_mn, FRAC_MN), aligned(shape_k, FRAC_K), addr);
    }
};

template <typename ptr_t, typename elem_t>
struct result_ptr {
    uintptr_t addr;
    const uint64_t shape_m, shape_n;  // GEMM output dimensions

    using dtype_t = elem_t;

    constexpr __aicore__ result_ptr(uint64_t shape_m, uint64_t shape_n, uintptr_t addr = 0) :
        addr(addr), shape_m(shape_m), shape_n(shape_n) {}

    template <typename dst_dtype_t = dtype_t>
    __aicore__ cvt_pointee_t<ptr_t, dst_dtype_t> ptr() const {
        return reinterpret_cast<cvt_pointee_t<ptr_t, dst_dtype_t>>(addr);
    }

    constexpr __aicore__ uint64_t size_per_stage() const { return shape_m * shape_n * sizeof(dtype_t); }
    constexpr __aicore__ uintptr_t offset(uint32_t n) const { return addr + n * size_per_stage(); }
    constexpr __aicore__ result_ptr<ptr_t, elem_t> operator[](uint32_t i) const {
        result_ptr<ptr_t, elem_t> r = *this;
        r.addr = offset(i);
        return r;
    }

    constexpr __aicore__ result_ptr<ptr_t, elem_t> offset_nz_m(uint32_t m) const {
        static_assert(std::is_same_v<ptr_t, __cc__ elem_t*>);
        result_ptr<ptr_t, elem_t> r = *this;
        r.addr += m * FRAC_MN * sizeof(dtype_t);
        return r;
    }

    constexpr __aicore__ result_ptr<ptr_t, elem_t> as_mad_aligned(uint64_t shape_m, uint64_t shape_n) const {
        return result_ptr<ptr_t, elem_t>(aligned(shape_m, FRAC_MN), aligned(shape_n, FRAC_MN), addr);
    }
};

template <typename T, Major M = Major::K> using l1_ptr = operand_ptr<__cbuf__ T*, T, M>;
template <typename T> using l0a_ptr = operand_ptr<__ca__ T*, T>;
template <typename T> using l0b_ptr = operand_ptr<__cb__ T*, T>;
template <typename T> using l0c_ptr = result_ptr<__cc__ T*, T>;
template <typename T> using ub_ptr = result_ptr<__ubuf__ T*, T>;

// gm_ptr<T, Major> (the GM operand handle) lives in common.hpp so the host
// can build one and pass it as a kernel launch argument.

} // namespace deep_gemm
// END DeepSeek MIT port: include/deep_gemm/ascend/tile_ptr.hpp

// BEGIN DeepSeek MIT port: include/deep_gemm/ascend/copy.hpp

#include <c_api/asc_simd.h>

namespace deep_gemm {

__aicore__ inline void set_l0c_nz2dn(uint64_t src_matrix_elems, uint64_t dst_matrix_elems) {
    asc_set_l0c2gm_nz2nd(1, src_matrix_elems / FRAC_MN, dst_matrix_elems);
    asc_set_l0c2gm_channel_para(static_cast<uint64_t>(1) << 48);
}

__aicore__ inline void set_mte2_nz(
    uint16_t num_batch, uint16_t dst_nz_n_stride, uint16_t dst_nz_c0_stride,
    uint16_t dst_batch_stride_blocks = 0
) {
    asc_set_gm2l1_nz_para(static_cast<uint64_t>(num_batch) | (static_cast<uint64_t>(dst_nz_n_stride) << 16) |
                          (static_cast<uint64_t>(dst_nz_c0_stride) << 32) |
                          (static_cast<uint64_t>(dst_batch_stride_blocks) << 48));
}

template <typename dtype_t>
__aicore__ inline void copy_ub_to_gm(
    __gm__ dtype_t* dst_ptr, __ubuf__ dtype_t* src_ptr,
    uint64_t shape_m, uint64_t shape_n,
    uint64_t dst_stride_bytes,
    asc_store_l2_cache_mode l2_ctrl = asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM
) {
    asc_copy_ub2gm_align(dst_ptr, src_ptr,
                         /* burst_num */ shape_m,
                         /* burst_len */ shape_n * sizeof(dtype_t),
                         l2_ctrl,
                         /* burst_dst_stride */ dst_stride_bytes,
                         /* burst_src_stride */ shape_n * sizeof(dtype_t));
}

template <typename T>
__aicore__ inline void copy_gm_to_ub(
    ub_ptr<T> dst, __gm__ T* src,
    uint32_t n_rows, uint32_t n_cols_bytes,
    uint64_t src_stride_bytes, uint32_t dst_stride_bytes,
    asc_load_l2_cache_mode l2_ctrl = asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
    uint8_t right_padding = 0,
    bool enable_constant_pad = false
) {
    asc_copy_gm2ub_align(dst.ptr(), src, n_rows, n_cols_bytes, 0, right_padding, enable_constant_pad, l2_ctrl,
                         src_stride_bytes, dst_stride_bytes);
}

template <typename T, Major M>
__aicore__ inline void copy_gm_to_ub(
    ub_ptr<T> dst, gm_ptr<T, M> src,
    uint32_t valid_m, uint32_t valid_k,
    asc_load_l2_cache_mode l2_ctrl = asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
    bool enable_constant_pad = false
) {
    const uint32_t rows = M == Major::K ? valid_m : valid_k;
    const uint32_t cols = M == Major::K ? valid_k : valid_m;
    constexpr uint32_t kBlockElems = 32 / sizeof(T);
    const uint8_t right_padding = static_cast<uint8_t>((kBlockElems - cols % kBlockElems) % kBlockElems);
    copy_gm_to_ub(dst, src.ptr(), rows, cols * sizeof(T), src.stride_outer * sizeof(T), dst.shape_n * sizeof(T),
                  l2_ctrl, right_padding, enable_constant_pad);
}

template <typename dtype_t>
__aicore__ inline void copy_gm_to_ub(
    __ubuf__ dtype_t* dst_ptr, __gm__ dtype_t* src_ptr,
    uint64_t shape_m, uint64_t shape_n,
    uint64_t src_stride_bytes,
    asc_load_l2_cache_mode l2_ctrl = asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM
) {
    asc_copy_gm2ub_align(dst_ptr, src_ptr,
                         /* burst_num */ shape_m,
                         /* burst_len */ shape_n * sizeof(dtype_t),
                         /* left_padding */ 0,
                         /* right_padding */ 0,
                         /* data_select */ false,
                         /* l2_cache_ctl */ l2_ctrl,
                         /* burst_src_stride */ src_stride_bytes,
                         /* burst_dst_stride */ shape_n * sizeof(dtype_t));
}

template <typename T>
__aicore__ inline void copy_gm_to_ub(
    ub_ptr<T> dst, __gm__ T* src,
    uint64_t src_stride_bytes,
    asc_load_l2_cache_mode l2_ctrl = asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM
) {
    asc_copy_gm2ub_align(dst.ptr(), src,
                         /* burst_num */ dst.shape_m,
                         /* burst_len */ dst.shape_n * sizeof(T),
                         /* left_padding */ 0,
                         /* right_padding */ 0,
                         /* data_select */ false,
                         /* l2_cache_ctl */ l2_ctrl,
                         /* burst_src_stride */ src_stride_bytes,
                         /* burst_dst_stride */ dst.shape_n * sizeof(T));
}

template <bool kSrcIsNZ = false, typename T, Major M>
__aicore__ inline void copy_gm_to_l1(
    l1_ptr<T, M> dst, gm_ptr<T, M> src,
    uint32_t actual_mn, uint32_t actual_k,
    asc_load_l2_cache_mode l2_ctrl = asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
    uint16_t num_batch = 1, uint16_t dst_batch_stride_blocks = 0,
    uint64_t src_batch_stride_bytes = 0
) {
    constexpr uint32_t pack_factor = is_fp4<T>() ? 2 : 1;
    using copy_dtype_t = std::conditional_t<is_fp4<T>(), int8_t, T>;
    if constexpr (kSrcIsNZ) {
        static_assert(M == Major::K, "Direct NZ copy requires K-major GM storage");
        const uint32_t page_size = dst_batch_stride_blocks;
        const uint32_t num_fragments = actual_k / (32 * pack_factor);
        asc_set_gm2l1_loop_size(num_batch, 1);
        asc_set_gm2l1_loop1_stride(src_batch_stride_bytes, page_size * 32);
        asc_set_gm2l1_loop2_stride(0, 0);
        asc_copy_gm2l1_align(dst.template ptr<uint8_t>(), src.template ptr<uint8_t>(), num_fragments, actual_mn * 32, 0,
                             0, false, l2_ctrl, page_size * 32, dst.shape_mn * 32);
    } else if constexpr (M == Major::K) {
        set_mte2_nz(num_batch, 1, dst.shape_mn, dst_batch_stride_blocks);
        asc_copy_gm2l1_nd2nz(dst.template ptr<copy_dtype_t>(), src.template ptr<copy_dtype_t>(),
                             src.stride_outer * sizeof(T), l2_ctrl, actual_mn, actual_k / pack_factor,
                             src_batch_stride_bytes, false);
    } else {
        set_mte2_nz(num_batch, 1, dst.shape_k, dst_batch_stride_blocks);
        asc_copy_gm2l1_nd2nz(dst.template ptr<copy_dtype_t>(), src.template ptr<copy_dtype_t>(),
                             src.stride_outer * sizeof(T), l2_ctrl, actual_k, actual_mn / pack_factor,
                             src_batch_stride_bytes, false);
    }
}

template <typename T, Major M>
__aicore__ inline void copy_l1_to_l0a(l0a_ptr<T> dst, l1_ptr<T, M> src, uint64_t mn_idx, uint64_t k_idx) {
    constexpr uint32_t FRAC_K = get_frac_k<T>();
    if constexpr (M == Major::K) {
        asc_copy_l12l0a(dst.ptr(), src.ptr(), mn_idx / FRAC_MN, k_idx / FRAC_K, dst.shape_mn / FRAC_MN,
                        dst.shape_k / FRAC_K, src.shape_mn / FRAC_MN, dst.shape_mn / FRAC_MN);
    } else {
        uint32_t m_step = dst.shape_k / FRAC_MN;
        if constexpr (sizeof(T) == 1) m_step = (m_step + 1) / 2 * 2;
        asc_copy_l12l0a_transpose(dst.ptr(), src.ptr(), k_idx / FRAC_MN, mn_idx / FRAC_K, m_step,
                                  ceil_div(dst.shape_mn, FRAC_K), src.shape_k / FRAC_MN, dst.shape_mn / FRAC_MN);
    }
}

template <typename T, Major M>
__aicore__ inline void copy_l1_to_l0b(l0b_ptr<T> dst, l1_ptr<T, M> src, uint64_t mn_idx, uint64_t k_idx) {
    constexpr uint32_t FRAC_K = get_frac_k<T>();
    if constexpr (M == Major::K) {
        asc_copy_l12l0b(dst.ptr(), src.ptr(), mn_idx / FRAC_MN, k_idx / FRAC_K, dst.shape_mn / FRAC_MN,
                        dst.shape_k / FRAC_K, src.shape_mn / FRAC_MN, dst.shape_mn / FRAC_MN);
    } else {
        uint32_t m_step = dst.shape_k / FRAC_MN;
        if constexpr (sizeof(T) == 1) m_step = (m_step + 1) / 2 * 2;
        asc_copy_l12l0b_transpose(dst.ptr(), src.ptr(), k_idx / FRAC_MN, mn_idx / FRAC_K, m_step,
                                  ceil_div(dst.shape_mn, FRAC_K), src.shape_k / FRAC_MN, dst.shape_mn / FRAC_MN);
    }
}

template <bool kSrcIsNZ = false, Major M>
__aicore__ inline void copy_gm_to_l1_mx(
    l1_ptr<int16_t> dst, gm_ptr<int16_t, M> src,
    uint32_t mn_elems, uint32_t sf_pairs,
    asc_load_l2_cache_mode l2_ctrl = asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
    uint16_t num_batch = 1, uint16_t dst_batch_stride_blocks = 0,
    uint64_t src_batch_stride_bytes = 0
) {
    if constexpr (kSrcIsNZ) {
        const uint32_t page_bytes = dst_batch_stride_blocks * 32;
        const uint32_t valid_bytes = ceil_div(mn_elems, FRAC_MN) * sf_pairs * 32;
        asc_set_gm2l1_loop_size(1, 1);
        asc_set_gm2l1_loop1_stride(0, 0);
        asc_set_gm2l1_loop2_stride(0, 0);
        asc_copy_gm2l1_align(dst.ptr(), src.ptr(), num_batch, valid_bytes, 0, 0, false, l2_ctrl, src_batch_stride_bytes,
                             page_bytes);
    } else {
        set_mte2_nz(num_batch, 1, dst.shape_k, dst_batch_stride_blocks);
        if constexpr (M == Major::K) {
            asc_copy_gm2l1_dn2nz(dst.ptr(), src.ptr(),
                                 /* stride */ src.stride_outer * sizeof(int16_t),
                                 /* l2ctrl */ l2_ctrl,
                                 /* n (rows) */ sf_pairs,
                                 /* d (cols) */ mn_elems,
                                 /* batch_stride */ src_batch_stride_bytes,
                                 /* small */ false);
        } else {
            asc_copy_gm2l1_nd2nz(dst.ptr(), src.ptr(),
                                 /* stride */ src.stride_outer * sizeof(int16_t),
                                 /* l2ctrl */ l2_ctrl,
                                 /* n (rows) */ sf_pairs,
                                 /* d (cols) */ mn_elems,
                                 /* batch_stride */ src_batch_stride_bytes,
                                 /* small */ false);
        }
    }
}

template <typename T>
__aicore__ inline void copy_l1_to_l0a_mx(
    l0a_ptr<T> l0a, l1_ptr<int16_t> sf_l1,
    uint64_t m_idx, uint16_t k_idx,
    uint32_t eff_k = 0
) {
    uint16_t sf_pairs = (eff_k ? eff_k : l0a.shape_k) / MX_SF_DIVISOR;
    asc_copy_l12l0a_mx(l0a.addr / MX_ADDR_DIV, sf_l1.ptr<fp8_e8m0_t>(),
                       /* x_start */ m_idx / FRAC_MN,
                       /* y_start */ k_idx,
                       /* x_step */ l0a.shape_mn / FRAC_MN,
                       /* y_step */ sf_pairs,
                       /* src_stride */ sf_l1.shape_k,
                       /* dst_stride */ sf_pairs);
}

template <typename T>
__aicore__ inline void copy_l1_to_l0b_mx(
    l0b_ptr<T> l0b, l1_ptr<int16_t> sf_l1,
    uint64_t m_idx, uint16_t k_idx,
    uint32_t eff_k = 0
) {
    uint16_t sf_pairs = (eff_k ? eff_k : l0b.shape_k) / MX_SF_DIVISOR;
    asc_copy_l12l0b_mx(l0b.addr / MX_ADDR_DIV, sf_l1.ptr<fp8_e8m0_t>(),
                       /* x_start */ m_idx / FRAC_MN,
                       /* y_start */ k_idx,
                       /* x_step */ l0b.shape_mn / FRAC_MN,
                       /* y_step */ sf_pairs,
                       /* src_stride */ sf_l1.shape_k,
                       /* dst_stride */ sf_pairs);
}

template <typename T, typename ST = float>
constexpr __aicore__ asc_quant_mode infer_quant_mode(bool scalar = false) {
    if constexpr (std::is_same_v<ST, float>) {
        if constexpr (std::is_same_v<T, float>) {
            return asc_quant_mode::NoQuant;
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            return asc_quant_mode::F322BF16;
        } else if constexpr (std::is_same_v<T, half>) {
            return asc_quant_mode::F322F16;
        } else if constexpr (std::is_same_v<T, float8_e4m3_t>) {
            return scalar ? asc_quant_mode::QF322FP8_PRE : asc_quant_mode::VQF322FP8_PRE;
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            return scalar ? asc_quant_mode::QF322B8_PRE : asc_quant_mode::VQF322B8_PRE;
        } else if constexpr (std::is_same_v<T, int4x2_t>) {
            return scalar ? asc_quant_mode::QF322S4_PRE : asc_quant_mode::VQF322S4_PRE;
        } else if constexpr (std::is_same_v<T, hifloat8_t>) {
            return scalar ? asc_quant_mode::QF322HIF8_PRE : asc_quant_mode::VQF322HIF8_PRE;
        } else {
            static_assert(!std::is_same_v<T, T>, "Unsupported quantization type");
        }
    } else if constexpr (std::is_same_v<ST, int32_t>) {
        if constexpr (std::is_same_v<T, int32_t>) {
            return asc_quant_mode::NoQuant;
        } else if constexpr (std::is_same_v<T, int8_t>) {
            return scalar ? asc_quant_mode::REQ8 : asc_quant_mode::VREQ8;
        } else if constexpr (std::is_same_v<T, int4x2_t>) {
            return scalar ? asc_quant_mode::REQ4 : asc_quant_mode::VREQ4;
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            return scalar ? asc_quant_mode::DEQF16 : asc_quant_mode::VDEQF16;
        } else {
            static_assert(!std::is_same_v<T, T>, "Unsupported quantization type");
        }
    }
}

template <typename T, typename ST>
__aicore__ inline void copy_l0c_to_ub(
    ub_ptr<T> dst, l0c_ptr<ST> src,
    uint32_t m_elems, uint32_t n_elems,
    asc_dual_dst_mode dual_dst_ctrl,
    uint32_t dst_subblock_id = 0,
    asc_quant_mode quant_mode = infer_quant_mode<T, ST>(),
    bool relu = false,
    bool transpose_out = false,  // false: write row-major [M, N]; true: col-major [N, M]
    asc_unit_flag_mode unit_flag = asc_unit_flag_mode::ENABLE_UPDATE
) {
    if (transpose_out) {
        set_l0c_nz2dn(src.shape_m * src.shape_n, static_cast<uint64_t>(n_elems) * dst.shape_n);
    } else {
        asc_set_l0c2gm_nz2nd(1, 0, 0);
    }
    asc_copy_l0c2ub(dst.ptr(), src.ptr(), n_elems, m_elems, dst.shape_n, src.shape_m, dst_subblock_id, dual_dst_ctrl,
                    unit_flag, quant_mode, relu ? asc_relu_pre_mode::NORMAL : asc_relu_pre_mode::NONE, false,
                    !transpose_out, transpose_out, false);
}

template <typename T, typename ST>
__aicore__ inline void copy_l0c_to_gm(
    __gm__ T* dst, l0c_ptr<ST> src,
    uint32_t m_elems, uint32_t n_elems,
    uint64_t dst_stride_elems,
    asc_store_l2_cache_mode l2_ctrl = asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
    asc_quant_mode quant_mode = infer_quant_mode<T, ST>(),
    bool relu = false,
    bool transpose_out = false,  // false: write row-major [M, N]; true: col-major [N, M]
    asc_unit_flag_mode unit_flag = asc_unit_flag_mode::ENABLE_UPDATE
) {
    if (transpose_out) {
        set_l0c_nz2dn(src.shape_m * src.shape_n, static_cast<uint64_t>(n_elems) * dst_stride_elems);
    } else {
        asc_set_l0c2gm_nz2nd(1, 0, 0);
    }
    asc_copy_l0c2gm(dst, src.ptr(), n_elems, m_elems, dst_stride_elems, src.shape_m, l2_ctrl, unit_flag, quant_mode,
                    relu ? asc_relu_pre_mode::NORMAL : asc_relu_pre_mode::NONE, false, !transpose_out, transpose_out,
                    false);
}

template <typename T>
__aicore__ inline void copy_ub_to_gm(
    __gm__ T* dst, ub_ptr<T> src,
    uint32_t n_rows, uint32_t n_cols,
    uint64_t dst_stride_bytes,
    asc_store_l2_cache_mode l2_ctrl = asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM
) {
    asc_copy_ub2gm_align(dst, src.ptr(), n_rows, n_cols * sizeof(T), l2_ctrl, dst_stride_bytes,
                         src.shape_n * sizeof(T));
}

template <typename T>
__aicore__ inline void fill_l1(
    __cbuf__ T* dst, T value,
    uint32_t num_rows, uint32_t col_blks_32b,
    uint32_t dst_gap_32b
) {
    asc_fill_value_config config;
    config.repeat = num_rows;
    config.blk_num = col_blks_32b;
    config.dst_gap = dst_gap_32b;
    using value_t = std::conditional_t<std::is_integral_v<T>, uint32_t, T>;
    asc_fill_l1(dst, static_cast<value_t>(value), config);
}

template <typename T, Major M>
__aicore__ inline void fill_tail_block(l1_ptr<T, M> ptr, uint32_t actual_k) {
    static_assert(sizeof(T) == 1, "Expect 8-bit (fp8) or packed-4bit (fp4) data type");
    constexpr uint32_t FRAC_K = get_frac_k<T>();
    const uint32_t k64 = aligned(actual_k, 64);
    if constexpr (M == Major::K) {
        const uint32_t nz32 = aligned(actual_k, 32);
        if (nz32 < k64) {
            fill_l1(reinterpret_cast<__cbuf__ int16_t*>(ptr.ptr() + (nz32 / 32) * ptr.shape_mn * 32),
                    static_cast<int16_t>(0),
                    /* num_rows */ 1,
                    /* col_blks_32b */ ptr.shape_mn,
                    /* dst_gap_32b */ 0);
        }
    } else {
        if (actual_k < k64) {
            fill_l1(reinterpret_cast<__cbuf__ int16_t*>(ptr.ptr() + actual_k * 32),
                    static_cast<int16_t>(0),
                    /* num_rows */ ptr.shape_mn / FRAC_K,
                    /* col_blks_32b */ k64 - actual_k,
                    /* dst_gap_32b */ ptr.shape_k - (k64 - actual_k));
        }
    }
}

template <typename T>
__aicore__ inline void fill_mn_tail_block(l1_ptr<T, Major::K> ptr, uint32_t actual_mn, uint32_t padded_mn) {
    static_assert(sizeof(T) == 1, "Expect 8-bit (fp8) or packed-4bit (fp4) data type");
    // ptr.shape_mn remains the physical NZ stride; padded_mn only bounds the
    // region consumed by the following L1->L0 transfer.
    if (actual_mn >= padded_mn) return;
    const uint32_t k_blocks = ptr.shape_k / get_frac_k<T>();  // one 32B row per C0
    fill_l1(reinterpret_cast<__cbuf__ int16_t*>(ptr.ptr() + actual_mn * 32), static_cast<int16_t>(0),
            /* num_rows */ k_blocks, /* col_blks_32b */ padded_mn - actual_mn,
            /* dst_gap_32b */ ptr.shape_mn - (padded_mn - actual_mn));
}

template <typename TA, typename TB, typename TC>
__aicore__ inline void mad(l0c_ptr<TC> l0c, l0a_ptr<TA> l0a, l0b_ptr<TB> l0b,
                           uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                           asc_unit_flag_mode unit_flag, bool clear_C = false) {
    if constexpr (sizeof(TA) == 1 || sizeof(TB) == 1) {
        asc_mmad_mx(l0c.ptr(), l0a.ptr(), l0b.ptr(), shape_m, shape_k, shape_n, static_cast<uint8_t>(unit_flag), true,
                    false, clear_C);
    } else {
        asc_mmad(l0c.ptr(), l0a.ptr(), l0b.ptr(), shape_m, shape_k, shape_n, static_cast<uint8_t>(unit_flag), true,
                 false, clear_C);
    }
}

} // namespace deep_gemm
// END DeepSeek MIT port: include/deep_gemm/ascend/copy.hpp

// BEGIN DeepSeek MIT port: include/deep_gemm/ascend/sync.hpp

#include <c_api/asc_simd.h>

namespace deep_gemm {

// New typed PIPE interface; existing kernels will migrate incrementally
// One directed PIPE event; base + stage_idx selects a staged event
// The parameter type checks the PIPE directions written at each call
template <pipe_t kSrc, pipe_t kDst>
struct PipeEvent {
    uint32_t base;
};

template <pipe_t kSrc, pipe_t kDst, uint32_t kNumStages = 1>
__aicore__ __forceinline__ void notify(PipeEvent<kSrc, kDst> event, uint32_t stage_idx = 0) {
    #pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++i)
        asc_sync_notify(kSrc, kDst, static_cast<event_t>(event.base + stage_idx + i));
}

template <pipe_t kSrc, pipe_t kDst, uint32_t kNumStages = 1>
__aicore__ __forceinline__ void wait(PipeEvent<kSrc, kDst> event, uint32_t stage_idx = 0) {
    #pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++i)
        asc_sync_wait(kSrc, kDst, static_cast<event_t>(event.base + stage_idx + i));
}

__aicore__ inline void set_intra_block(pipe_t pipe, uint8_t subblock_id, uint8_t flag_id) {
    asc_sync_intra_arrive(pipe, ((subblock_id & 1) << 4) | (flag_id & 0xf));
}

__aicore__ inline void wait_intra_block(pipe_t pipe, uint8_t subblock_id, uint8_t flag_id) {
    // AIV ignores subblock_id for intra-block waiting.
    if ASCEND_IS_AIV {
        asc_sync_intra_wait(pipe, flag_id & 0xf);
    } else {
        asc_sync_intra_wait(pipe, ((subblock_id & 1) << 4) | (flag_id & 0xf));
    }
}

// New intra-block interface; existing callers will migrate incrementally
// kSubblock selects the AIC/AIV0 or AIC/AIV1 connection; kPipe is local
template <uint32_t kSubblock>
struct IntraBlockEvent {
    uint32_t base;
};

template <pipe_t kPipe, uint32_t kSubblock, uint32_t kNumStages = 1>
__aicore__ __forceinline__ void notify_intra_block(IntraBlockEvent<kSubblock> event, uint32_t stage_idx = 0) {
    #pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++i)
        set_intra_block(kPipe, kSubblock, event.base + stage_idx + i);
}

template <pipe_t kPipe, uint32_t kSubblock, uint32_t kNumStages = 1>
__aicore__ __forceinline__ void wait_intra_block(IntraBlockEvent<kSubblock> event, uint32_t stage_idx = 0) {
    #pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++i)
        wait_intra_block(kPipe, kSubblock, event.base + stage_idx + i);
}

template <pipe_t pipe, pipe_t tpipe, uint32_t size, uint32_t offset = 0>
__aicore__ inline void set_flags() {
    for (uint32_t i = 0; i < size; i++)
        asc_sync_notify(pipe, tpipe, static_cast<event_t>(offset + i));
}

template <pipe_t pipe, pipe_t tpipe, uint32_t size, uint32_t offset = 0>
__aicore__ inline void wait_flags() {
    for (uint32_t i = 0; i < size; i++)
        asc_sync_wait(pipe, tpipe, static_cast<event_t>(offset + i));
}

template <pipe_t pipe, uint32_t size, uint32_t offset = 0>
__aicore__ inline void set_intra_blocks(uint8_t subblock_id) {
    for (uint32_t i = 0; i < size; i++)
        set_intra_block(pipe, subblock_id, offset + i);
}

template <pipe_t pipe, uint32_t size, uint32_t offset = 0>
__aicore__ inline void wait_intra_blocks(uint8_t subblock_id) {
    for (uint32_t i = 0; i < size; i++)
        wait_intra_block(pipe, subblock_id, offset + i);
}

} // namespace deep_gemm
// END DeepSeek MIT port: include/deep_gemm/ascend/sync.hpp

// END DeepSeek MIT port: include/deep_gemm/ascend.hpp

// BEGIN DeepSeek MIT port: include/deep_gemm/comm/mega_moe.hpp

// BEGIN DeepSeek MIT port: include/deep_gemm/layout/mega_moe.hpp

#include <cstdint>
#include <type_traits>

// BEGIN DeepSeek MIT port: include/deep_gemm/layout/sym_buffer.hpp

#include <cstdint>


namespace deep_gemm::layout {

// Communication kernels encode rank indices in 8 bits
constexpr uint32_t kNumMaxRanks = 1u << 8;

template <uint32_t kNumRanks = kNumMaxRanks>
struct SymBuffer {
    int64_t base;
    uint32_t rank_idx;
    // Keep rank and the active offsets near base in the launch data
    int64_t offsets[kNumMaxRanks];

    static_assert(kNumRanks <= kNumMaxRanks);
    SymBuffer() = default;

    template <typename Container>
    explicit SymBuffer(const Container& c, uint32_t rank_idx): base(c[rank_idx]), rank_idx(rank_idx) {
        for (uint32_t i = 0; i < kNumMaxRanks; ++ i)
            offsets[i] = i < c.size() ? c[i] - base : 0;
    }

#if defined(__CCE__) || defined(__CLION_IDE__)
    template <typename ptr_t = __gm__ void*>
    constexpr __aicore__ __attribute__((always_inline)) ptr_t get_base_ptr() const {
        return __builtin_bit_cast(ptr_t, static_cast<uint64_t>(base));
    }

    // SIMT uses a UB copy of the rank offsets
    template <typename ptr_t, typename offset_ptr_t>
    static constexpr __aicore__ __attribute__((always_inline)) ptr_t map(ptr_t ptr, uint32_t dst_rank_idx, offset_ptr_t offsets) {
        if constexpr (kNumRanks == 1)
            return ptr;
        const uint64_t address = __builtin_bit_cast(uint64_t, ptr) + offsets[dst_rank_idx];
        return __builtin_bit_cast(ptr_t, address);
    }

    template <typename ptr_t>
    constexpr __aicore__ __attribute__((always_inline)) ptr_t map(ptr_t ptr, uint32_t dst_rank_idx) const {
        return map(ptr, dst_rank_idx, offsets);
    }
#endif
};

} // namespace deep_gemm::layout
// END DeepSeek MIT port: include/deep_gemm/layout/sym_buffer.hpp


namespace deep_gemm::mega_moe::layout {

// Logical blocks keep the GM stride and task count independent of the MAD shape
constexpr uint32_t BLOCK_M = 256, BLOCK_N = 256;

using deep_gemm::layout::SymBuffer;

// GM buffer alignment/count stride, not the 128-byte L2 cache sector
constexpr uint64_t kNumBufferAlignmentBytes = 512;

// SIMT threads per metadata worker and routes per batch
constexpr uint32_t kNumDispatchThreads = MOE_TOKENS > 256 ? 1024 : 512;

// Packed count: arrivals [31:24], value [23:0]
constexpr uint32_t kCountArrivalBit = 1u << 24;
constexpr uint32_t kCountValueMask = kCountArrivalBit - 1;

// Routed progress counts workers or GEMM N blocks
enum ProgressStage : uint32_t {
    kDispatched,    // Linear1 inputs ready
    kLinear1Ready,  // Quantized Linear2 inputs ready
    kLinear2Ready,  // BF16 expert outputs ready
    kCombineSent,   // Sends complete; output ring reusable
    kNumProgressStages
};

// Fixed shape and registered token capacity
struct Config {
    uint32_t num_ranks;
    uint32_t num_experts;
    uint32_t num_shared_experts;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_topk;
    uint32_t hidden;
    uint32_t intermediate_hidden;

    const uint32_t shared_intermediate_hidden = num_shared_experts * intermediate_hidden;
    const uint32_t num_input_sf_pairs = hidden / MX_SF_DIVISOR;
};

// Typed GM range with a fixed byte stride
template <typename dtype_t, uint64_t kStrideBytes = sizeof(dtype_t)>
struct Buffer {
    static_assert(kStrideBytes >= sizeof(dtype_t) and kStrideBytes % alignof(dtype_t) == 0);
    uint64_t offset;

    constexpr __aicore__ Buffer(uint64_t& next_offset, uint64_t num_elements, uint64_t alignment = kStrideBytes):
        offset(aligned(next_offset, alignment)) { next_offset = offset + num_elements * kStrideBytes; }

    // entry_idx uses kStrideBytes even when view_t changes
    // Buffer<uint32_t>::get_ptr<uint64_t>(base, i) advances by i * 4 bytes
    template <typename view_t = dtype_t, typename base_t>
    constexpr __aicore__ __attribute__((always_inline)) __gm__ view_t* get_ptr(base_t base, uint64_t entry_idx = 0) const {
        static_assert(std::is_pointer_v<base_t> or std::is_same_v<base_t, uint64_t>,
                      "Buffer base must be a pointer or a uint64_t address");

        // Preserve the base type through CANN lowering
        const auto address = __builtin_bit_cast(uintptr_t, base) + offset + entry_idx * kStrideBytes;
        return __builtin_bit_cast(__gm__ view_t*, address);
    }
};

struct LinearWeights {
    __gm__ const uint8_t* data;
    __gm__ const int16_t* sf;
};

struct Weights {
    LinearWeights l1;
    LinearWeights l2;
    LinearWeights shared_l1;
    LinearWeights shared_l2;
    LinearWeights l1_up;
    LinearWeights shared_l1_up;
};
static_assert(sizeof(Weights) == 12 * sizeof(uint64_t));

// Preserve the public PyPTO weight layouts and pointer order.
__aicore__ inline Weights make_original_weights(
    __gm__ const uint8_t* w1, __gm__ const uint8_t* s1,
    __gm__ const uint8_t* w3, __gm__ const uint8_t* s3,
    __gm__ const uint8_t* w2, __gm__ const uint8_t* s2,
    __gm__ const uint8_t* shared_w1, __gm__ const uint8_t* shared_s1,
    __gm__ const uint8_t* shared_w3, __gm__ const uint8_t* shared_s3,
    __gm__ const uint8_t* shared_w2, __gm__ const uint8_t* shared_s2) {
    return {{w1, reinterpret_cast<__gm__ const int16_t*>(s1)},
            {w2, reinterpret_cast<__gm__ const int16_t*>(s2)},
            {shared_w1, reinterpret_cast<__gm__ const int16_t*>(shared_s1)},
            {shared_w2, reinterpret_cast<__gm__ const int16_t*>(shared_s2)},
            {w3, reinterpret_cast<__gm__ const int16_t*>(s3)},
            {shared_w3, reinterpret_cast<__gm__ const int16_t*>(shared_s3)}};
}

// GM workspace layout; all offsets are relative to the allocation base
struct Workspace {
private:
    // Host and device use the same storage layout
#ifdef __CCE__
    using fp8_t = float8_e4m3_t;
    using bf16_t = bfloat16_t;
#else
    using fp8_t = uint8_t;
    using bf16_t = uint16_t;
#endif

    // Keep capacity calculations independent of include order
    template <typename lhs_t, typename rhs_t>
    static constexpr __aicore__ auto ceil_div(lhs_t value, rhs_t divisor) -> decltype(value + divisor) {
        return (value + divisor - 1) / divisor;
    }

public:
    // Rank-barrier ABI; alternate signal slots by epoch
    struct RankBarrier {
        uint64_t epoch;
        int32_t signals[2];
    };

    struct FP8Buffer {
        Buffer<fp8_t> data;
        Buffer<int16_t> sf;

        constexpr __aicore__ FP8Buffer(uint64_t& offset, uint64_t num_rows, uint32_t num_cols):
            data{offset, num_rows * num_cols, kNumBufferAlignmentBytes},
            sf{offset, num_rows * ceil_div(num_cols, MX_SF_DIVISOR), kNumBufferAlignmentBytes} {}
    };

    Config config;

    // Reserve one packed index/weight entry per source route.
    const uint32_t num_src_slots_per_expert = config.num_max_tokens_per_rank * config.num_topk;
    const uint64_t num_src_indices = static_cast<uint64_t>(config.num_experts) * num_src_slots_per_expert;
    const uint32_t num_padded_shared_rows = aligned(config.num_max_tokens_per_rank, BLOCK_M);

    const uint32_t num_max_shared_m_blocks = config.num_shared_experts > 0 ? ceil_div(config.num_max_tokens_per_rank, BLOCK_M) : 0;
    const uint32_t num_progress_counters_per_stage = num_max_shared_m_blocks + get_num_max_m_blocks();
    const uint64_t num_full_pool_rows = get_num_max_m_blocks() * BLOCK_M;

    // Size from the registered limit, not active T
    const uint64_t num_ring_rows = static_cast<uint64_t>(get_num_ring_blocks()) * BLOCK_M;

    // Field order fixes the symmetric GM placement
    uint64_t next_offset = 0;
    struct {
        Buffer<uint32_t> grid_sync_count;
        Buffer<RankBarrier> rank_barriers;
    } sync{
        .grid_sync_count = {next_offset, 3, kNumBufferAlignmentBytes},
        .rank_barriers = {next_offset, 2, kNumBufferAlignmentBytes},
    };

    struct {
        Buffer<uint32_t> worker_expert_count;
        Buffer<uint32_t> total_expert_count;
        // Completed producer M-blocks; all ones means this rank has finished its stores
        Buffer<uint64_t, kL2SectorBytes> combine_ready_prefix;
        Buffer<uint32_t> expert_recv_count;
        Buffer<uint32_t, kNumBufferAlignmentBytes> expert_recv_count_sum;

        // Packed AIV0 arrivals/M-block count is independent of expert counters
        Buffer<uint32_t> num_m_blocks_certificate;
        Buffer<uint64_t> src_token_metadata;
        Buffer<uint32_t> staged_token_topk_idx;
    } metadata{
        .worker_expert_count = {next_offset, static_cast<uint64_t>(kNumAICores) * config.num_experts, kNumBufferAlignmentBytes},
        .total_expert_count = {next_offset, config.num_experts},
        .combine_ready_prefix = {next_offset, config.num_ranks},
        .expert_recv_count = {next_offset, config.num_experts},
        .expert_recv_count_sum = {next_offset, config.num_experts / config.num_ranks},
        .num_m_blocks_certificate = {next_offset, 1, kL2SectorBytes},
        .src_token_metadata = {next_offset, num_src_indices, kNumBufferAlignmentBytes},
        .staged_token_topk_idx = {next_offset, num_src_indices, kNumBufferAlignmentBytes},
    };

    struct {
        FP8Buffer acts;
        Buffer<int64_t> topk_idx;
        Buffer<float> topk_weights;
    } input{
        .acts = {next_offset, config.num_max_tokens_per_rank, config.hidden},
        .topk_idx = {next_offset, static_cast<uint64_t>(config.num_max_tokens_per_rank) * config.num_topk, kNumBufferAlignmentBytes},
        .topk_weights = {next_offset, static_cast<uint64_t>(config.num_max_tokens_per_rank) * config.num_topk, kNumBufferAlignmentBytes},
    };

    // Clear these progress arrays together on each launch
    struct {
        Buffer<uint32_t, kL2SectorBytes> blocks;
        Buffer<uint8_t, kL2SectorBytes> num_issued_gemms;
        Buffer<uint8_t, kL2SectorBytes> metadata_ready;
    } progress{
        .blocks = {next_offset, static_cast<uint64_t>(kNumProgressStages) * num_progress_counters_per_stage},
        .num_issued_gemms = {next_offset, kNumAICores},
        .metadata_ready = {next_offset, kNumAICores},
    };
    const uint64_t num_progress_bytes = next_offset - progress.blocks.offset;

    struct {
        FP8Buffer l1_acts;

        // Metadata uses logical rows and never wraps
        Buffer<float> topk_weights;
        Buffer<uint32_t> src_metadata;

        // Input SF is token-major; quantized Linear2 SF is K-major within each M-block
        FP8Buffer l2_acts;
        Buffer<bf16_t> output;
    } routed{
        .l1_acts = {next_offset, num_ring_rows, config.hidden},
        .topk_weights = {next_offset, num_full_pool_rows, kNumBufferAlignmentBytes},
        .src_metadata = {next_offset, num_full_pool_rows, kNumBufferAlignmentBytes},
        .l2_acts = {next_offset, num_ring_rows, config.intermediate_hidden},
        .output = {next_offset, num_ring_rows * config.hidden, kNumBufferAlignmentBytes},
    };

    // One BF16 slot per route, plus one combined shared slot
    Buffer<bf16_t> combine_input{
        next_offset, static_cast<uint64_t>(config.num_topk + (config.num_shared_experts > 0)) * config.num_max_tokens_per_rank * config.hidden,
        kNumBufferAlignmentBytes};

    // Shared Linear2 inputs; Linear1 reads original tokens
    FP8Buffer shared_l2_acts{next_offset, config.num_shared_experts > 0 ? num_padded_shared_rows : 0, config.shared_intermediate_hidden};

    // Producer M-block for each input route; dispatch fills this before publishing readiness
    Buffer<uint32_t> route_m_blocks{
        next_offset, static_cast<uint64_t>(config.num_max_tokens_per_rank) * config.num_topk, kNumBufferAlignmentBytes};

    constexpr __aicore__ uint64_t get_num_max_m_blocks() const {
        return ceil_div(static_cast<uint64_t>(config.num_ranks) * config.num_max_tokens_per_rank * config.num_topk, BLOCK_M) + config.num_experts / config.num_ranks;
    }

    constexpr __aicore__ uint32_t get_num_ring_blocks() const {
        // Measured overlap window; the deadlock-safe minimum can still lose throughput
        const uint32_t num_windows = config.hidden / BLOCK_N <= kNumAICores / 2 ? 4 : 1;
        const uint32_t num_window_blocks = num_windows * kNumAICores;
        return get_num_max_m_blocks() < num_window_blocks ? static_cast<uint32_t>(get_num_max_m_blocks()) : num_window_blocks;
    }

    // Coordinates within the registered GM layout
    constexpr __aicore__ __attribute__((always_inline)) uint32_t get_progress_idx(ProgressStage stage, uint32_t progress_slot_idx) const {
        return stage * num_progress_counters_per_stage + progress_slot_idx;
    }

    constexpr __aicore__ __attribute__((always_inline)) uint64_t get_src_metadata_idx(
        uint32_t local_expert_idx, uint32_t src_rank_idx, uint32_t src_slot_idx) const {
        return (static_cast<uint64_t>(local_expert_idx) * config.num_ranks + src_rank_idx) * num_src_slots_per_expert + src_slot_idx;
    }

    constexpr __aicore__ __attribute__((always_inline)) uint64_t get_expert_recv_idx(uint32_t src_rank_idx, uint32_t local_expert_idx) const {
        return static_cast<uint64_t>(src_rank_idx) * (config.num_experts / config.num_ranks) + local_expert_idx;
    }

    constexpr __aicore__ __attribute__((always_inline)) uint64_t get_combine_m_idx(uint64_t slot_idx, uint64_t token_idx) const {
        return slot_idx * config.num_max_tokens_per_rank + token_idx;
    }

    constexpr __aicore__ uint64_t get_num_bytes() const {
        // HCCL registration uses 2 MiB alignment
        return aligned(next_offset, 2 * 1024 * 1024);
    }
};

template <uint32_t kNumRanks, uint32_t kNumExperts, uint32_t kNumSharedExperts,
          uint32_t kNumMaxTokensPerRank, uint32_t kNumTopk, uint32_t kHidden, uint32_t kIntermediateHidden>
inline constexpr Workspace kWorkspace{{
    kNumRanks, kNumExperts, kNumSharedExperts, kNumMaxTokensPerRank, kNumTopk, kHidden, kIntermediateHidden}};

template <const auto& workspace>
constexpr __aicore__ uint32_t get_ring_block_idx(uint32_t pool_block_idx) {
    constexpr uint32_t kNumRingBlocks = workspace.get_num_ring_blocks();
    return kNumRingBlocks == workspace.get_num_max_m_blocks() ? pool_block_idx : pool_block_idx & (kNumRingBlocks - 1);
}

template <const auto& workspace>
constexpr __aicore__ uint64_t get_ring_m_idx(uint32_t pool_block_idx, uint32_t m_idx_in_block = 0) {
    return static_cast<uint64_t>(get_ring_block_idx<workspace>(pool_block_idx)) * BLOCK_M + m_idx_in_block;
}
} // namespace deep_gemm::mega_moe::layout
// END DeepSeek MIT port: include/deep_gemm/layout/mega_moe.hpp


#include <c_api/asc_simd.h>
#include <simt_api/device_functions.h>

namespace deep_gemm::mega_moe::comm {

// The predicate may advance work; true means the wait is complete
template <typename pred_t, typename print_timeout_t>
__aicore__ __forceinline__ void wait_until(const pred_t& pred, const print_timeout_t& print_timeout) {
    // 60 seconds at 1 GHz on Ascend 950DT
    constexpr uint64_t kNumTimeoutCycles = 60ull * 1'000'000'000;
    const auto start_cycle = static_cast<uint64_t>(asc_get_system_cycle());
    while (not pred()) {
        if (static_cast<uint64_t>(asc_get_system_cycle()) - start_cycle >= kNumTimeoutCycles) {
            print_timeout();
            trap();
        }
    }
}

__aicore__ inline void print_local_timeout(uint32_t rank_idx) {
    __asc_aicore::printf("MegaMoE local wait timeout: rank=%u core=%u\n", rank_idx, static_cast<uint32_t>(block_idx));
}

__aicore__ inline void print_peer_timeout(uint32_t rank_idx, uint32_t aiv_idx, uint32_t peer_idx,
                                         uint64_t prefix, bool wait_done) {
    __asc_aicore::printf("MegaMoE peer wait timeout: rank=%u aiv=%u peer=%u prefix=0x%llx wait_done=%u\n",
                        rank_idx, aiv_idx, peer_idx, static_cast<unsigned long long>(prefix), static_cast<uint32_t>(wait_done));
}

template <uint32_t kGridSyncIdx>
__simt_callee__ __forceinline__ void grid_sync(__gm__ uint32_t* counter_ptr, uint32_t rank_idx,
                                               uint32_t worker_idx, uint32_t num_workers) {
    // Worker deltas sum to the phase bit; low bits return to zero
    constexpr uint32_t kFinishSumTag = 0x80000000u;
    // 60 seconds of vector cycles at 1.65 GHz on Ascend 950DT
    constexpr uint64_t kNumTimeoutCycles = 60ull * 1'650'000'000;

    asc_threadfence();
    asc_syncthreads();
    if (num_workers == 1)
        return;
    if (threadIdx.x == 0) {
        const uint32_t old_value = asc_atomic_add(
            counter_ptr, worker_idx == 0 ? kFinishSumTag - (num_workers - 1) : 1u);
        const uint32_t expected_tag = (old_value ^ kFinishSumTag) & kFinishSumTag;
        const auto start_cycle = static_cast<uint64_t>(clock64());
        // SIMT cannot use the Scalar wait_until lambda
        while ((*reinterpret_cast<volatile __gm__ uint32_t*>(counter_ptr) & kFinishSumTag) != expected_tag) {
            if (static_cast<uint64_t>(clock64()) - start_cycle >= kNumTimeoutCycles) {
                const uint32_t current = *reinterpret_cast<volatile __gm__ uint32_t*>(counter_ptr);
                __asc_simt_vf::printf("MegaMoE grid sync timeout: rank=%u worker=%u grid=%u workers=%u current=0x%x expected_tag=0x%x\n",
                                     rank_idx, worker_idx, kGridSyncIdx, num_workers, current, expected_tag);
                __trap();
            }
        }
    }
    asc_syncthreads();
}

template <const auto& workspace, uint32_t kBarrierIdx = 0>
__aicore__ __forceinline__ void rank_barrier(__gm__ uint8_t* buffer, const layout::SymBuffer<>& sym_buffer, uint32_t vec_core_idx) {
    constexpr auto& config = workspace.config;
    const uint32_t rank_idx = sym_buffer.rank_idx;
    // Native event waits still rely on the device watchdog
    constexpr uint32_t kAllAIVSyncID = 14;
    asc_sync_pipe(PIPE_ALL);
    asc_sync_inter_arrive(PIPE_MTE3, kAllAIVSyncID);
    asc_sync_inter_wait(PIPE_S, kAllAIVSyncID);
    if constexpr (config.num_ranks == 1)
        return;
    if (vec_core_idx == 0) {
        const auto barrier = workspace.sync.rank_barriers.get_ptr(buffer, kBarrierIdx);
        const uint64_t epoch = asc_load_dev(&barrier->epoch);
        const int delta = epoch & 2 ? -1 : 1;
        const int target = delta < 0 ? 0 : static_cast<int>(config.num_ranks);
        asc_store_dev(&barrier->epoch, epoch + 1);
        const auto signal_ptr = &barrier->signals[epoch & 1];
        for (uint32_t dst_rank_idx = 0; dst_rank_idx < config.num_ranks; ++ dst_rank_idx) {
            const auto remote_base = sym_buffer.map(buffer, dst_rank_idx);
            const auto remote_barrier = workspace.sync.rank_barriers.get_ptr(remote_base, kBarrierIdx);
            asc_atomic_add(&remote_barrier->signals[epoch & 1], delta);
        }
        asc_sync_data_barrier(DSB_DDR);

        wait_until([&]() __aicore__ { return asc_load_dev(signal_ptr) == target; }, [&]() __aicore__ {
            __asc_aicore::printf("MegaMoE rank barrier timeout: rank=%u barrier=%u epoch=%llu signal=%d target=%d\n",
                   rank_idx, kBarrierIdx, static_cast<unsigned long long>(epoch), asc_load_dev(signal_ptr), target);
        });
    }
    asc_sync_pipe(PIPE_ALL);
    asc_sync_inter_arrive(PIPE_MTE3, kAllAIVSyncID);
    asc_sync_inter_wait(PIPE_S, kAllAIVSyncID);
}

template <const auto& workspace>
__aicore__ inline bool is_block_ready(__gm__ uint8_t* buffer, layout::ProgressStage stage,
                                      uint32_t progress_slot_idx, uint32_t num_expected_completions) {
    // Counts fit in the low byte of the atomic uint32 counter
    const auto completion = workspace.progress.blocks.template get_ptr<uint8_t>(buffer, workspace.get_progress_idx(stage, progress_slot_idx));
    return asc_load_dev(completion) >= num_expected_completions;
}

template <const auto& workspace>
__aicore__ inline void wait_for_block(__gm__ uint8_t* buffer, uint32_t rank_idx, layout::ProgressStage stage,
                                      uint32_t progress_slot_idx, uint32_t num_expected_completions) {
    wait_until([&]() __aicore__ {
        return is_block_ready<workspace>(buffer, stage, progress_slot_idx, num_expected_completions);
    }, [&]() __aicore__ { print_local_timeout(rank_idx); });
}

// Compact rank-local waves consume model route records directly. In both
// layouts an invalid expert code compares outside the global expert range.
using RouteId = uint32_t;

// Build expert-contiguous dispatch metadata
// A small route table fits in a few passes by one SIMT block. Each source
// rank owns a disjoint metadata range within every destination expert.
template <const auto& workspace>
constexpr bool kDirectMetadata =
    workspace.config.num_max_tokens_per_rank * workspace.config.num_topk <= 4 * layout::kNumDispatchThreads;

template <const auto& workspace>
__simt_vf__ __aicore__ __launch_bounds__(layout::kNumDispatchThreads)
void build_direct_metadata(__gm__ uint8_t* buffer, __gm__ const RouteId* ids,
                           __gm__ uint32_t* weights, uint32_t tokens, uint32_t rank,
                           __ubuf__ const int64_t* offsets) {
    constexpr auto& config = workspace.config;
    constexpr uint32_t local_experts = config.num_experts / config.num_ranks;
    constexpr auto ub_offset = aligned(config.num_ranks * sizeof(int64_t), 32);
    static_assert(ub_offset + config.num_experts * sizeof(uint32_t) <= UBSizeBytes);
    auto* counts = reinterpret_cast<__ubuf__ uint32_t*>(ub_offset);
    const uint32_t tid = threadIdx.x;
    for (uint32_t e = tid; e < config.num_experts; e += layout::kNumDispatchThreads)
        counts[e] = 0;
    asc_syncthreads();
    for (uint32_t i = tid; i < tokens * config.num_topk; i += layout::kNumDispatchThreads) {
        const auto e = static_cast<uint64_t>(asc_ldcg(const_cast<__gm__ RouteId*>(ids + i)));
        if (e >= config.num_experts)
            continue;
        const auto slot = asc_atomic_add(counts + e, 1u);
        auto* peer = layout::SymBuffer<>::map(buffer, e / local_experts, offsets);
        const auto dst = workspace.get_src_metadata_idx(e % local_experts, rank, slot);
        *workspace.metadata.src_token_metadata.get_ptr(peer, dst) =
            (static_cast<uint64_t>(asc_ldcg(weights + i)) << 32) | i;
    }
    // Complete every writer's peer stores before publishing any expert count.
    asc_threadfence();
    asc_syncthreads();
    for (uint32_t e = tid; e < config.num_experts; e += layout::kNumDispatchThreads) {
        auto* peer = layout::SymBuffer<>::map(buffer, e / local_experts, offsets);
        const auto local = e % local_experts;
        const auto count = counts[e];
        *workspace.metadata.expert_recv_count.get_ptr(peer, workspace.get_expert_recv_idx(rank, local)) = count;
        asc_threadfence();
        asc_atomic_add(workspace.metadata.expert_recv_count_sum.get_ptr(peer, local), layout::kCountArrivalBit | count);
    }
    asc_threadfence();
}

template <const auto& workspace, bool kOverlapReduce, bool kFullTopology>
__simt_vf__ __aicore__ __launch_bounds__(layout::kNumDispatchThreads)
void build_dispatch_metadata(__gm__ uint8_t* buffer, __gm__ const RouteId* topk_idx,
                             __gm__ uint32_t* route_weights,
                             uint32_t num_tokens, uint32_t rank_idx, uint32_t worker_idx, uint32_t num_workers,
                             __ubuf__ const int64_t* offsets) {
    constexpr auto& config = workspace.config;
    constexpr uint32_t kNumExpertsPerRank = config.num_experts / config.num_ranks;

    const uint32_t num_threads_per_round =
        (kFullTopology ? kNumAICores : num_workers) * layout::kNumDispatchThreads;

    const auto grid_sync_ptr = workspace.sync.grid_sync_count.get_ptr(buffer);
    const auto expert_count = workspace.metadata.worker_expert_count.get_ptr(buffer);
    const auto total_expert_count = workspace.metadata.total_expert_count.get_ptr(buffer);
    const auto staged_token_topk_idx = workspace.metadata.staged_token_topk_idx.get_ptr(buffer);
    // Metadata owns this UB scratch until the VF completes
    auto* expert_token_count = reinterpret_cast<__ubuf__ uint32_t*>(aligned(config.num_ranks * sizeof(int64_t), 32));
    static_assert(config.num_experts * sizeof(uint32_t) + aligned(config.num_ranks * sizeof(int64_t), 32) <= UBSizeBytes, "Metadata UB overflow");

    const uint32_t thread_idx = threadIdx.x;
    const uint32_t num_token_topk = num_tokens * config.num_topk;
    // Controlled experiment: mutable metadata is republished between logical
    // invocations in one persistent launch. Bypass SIMT L1 on these acquired
    // GM reads to test whether a prior invocation's cache lines can survive.

    // Count expert selections in UB
    for (uint32_t expert_idx = thread_idx; expert_idx < config.num_experts; expert_idx += layout::kNumDispatchThreads)
        expert_token_count[expert_idx] = 0;
    asc_syncthreads();

    // With overlap reduce, assign contiguous route batches to each worker
    const uint32_t worker_stride = not kOverlapReduce ? layout::kNumDispatchThreads :
        (num_token_topk + num_threads_per_round - 1) / num_threads_per_round * layout::kNumDispatchThreads;
    const uint32_t worker_begin = worker_idx * worker_stride;
    const uint32_t batch_stride = (kOverlapReduce ? 1 : num_workers) * layout::kNumDispatchThreads;
    const uint32_t worker_end = worker_begin + worker_stride;
    const uint32_t token_topk_end = kOverlapReduce and worker_end < num_token_topk ? worker_end : num_token_topk;
    for (uint32_t token_topk_idx = worker_begin + thread_idx; token_topk_idx < token_topk_end; token_topk_idx += batch_stride) {
        // Clear route mappings before counts allow any peer to dispatch these inputs
        if constexpr (kOverlapReduce)
            *workspace.route_m_blocks.get_ptr(buffer, token_topk_idx) = ~uint32_t{0};
        const auto expert_idx = static_cast<uint64_t>(asc_ldcg(const_cast<__gm__ RouteId*>(topk_idx + token_topk_idx)));
        if (expert_idx < config.num_experts)
            asc_atomic_add(expert_token_count + expert_idx, 1u);
    }
    asc_syncthreads();

    // Save counts, then reuse UB counters as write offsets
    for (uint32_t expert_idx = thread_idx; expert_idx < config.num_experts; expert_idx += layout::kNumDispatchThreads)
        expert_count[worker_idx * config.num_experts + expert_idx] = expert_token_count[expert_idx];
    grid_sync<0>(grid_sync_ptr, rank_idx, worker_idx, num_workers);

    for (uint32_t expert_idx = thread_idx; expert_idx < config.num_experts; expert_idx += layout::kNumDispatchThreads) {
        uint32_t slot_begin = 0;
        for (uint32_t prev_worker_idx = 0; prev_worker_idx < worker_idx; ++ prev_worker_idx)
            slot_begin += asc_ldcg(expert_count + prev_worker_idx * config.num_experts + expert_idx);
        if (worker_idx == num_workers - 1)
            total_expert_count[expert_idx] = slot_begin + expert_token_count[expert_idx];
        expert_token_count[expert_idx] = slot_begin;
    }
    asc_syncthreads();

    // Stage one route per thread in kNumDispatchThreads-route batches
    for (uint32_t batch_base = worker_begin; batch_base < token_topk_end; batch_base += batch_stride) {
        const uint32_t token_topk_idx = batch_base + thread_idx;
        if (token_topk_idx < token_topk_end) {
            const auto expert_idx = static_cast<uint64_t>(asc_ldcg(const_cast<__gm__ RouteId*>(topk_idx + token_topk_idx)));
            if (expert_idx < config.num_experts) {
                const uint32_t dst_slot_idx = asc_atomic_add(expert_token_count + expert_idx, 1u);
                staged_token_topk_idx[static_cast<uint64_t>(expert_idx) * workspace.num_src_slots_per_expert + dst_slot_idx] = token_topk_idx;
            }
        }

        // Keep route batches ordered across iterations
        if (kOverlapReduce and batch_base + batch_stride < token_topk_end)
            asc_syncthreads();
    }
    grid_sync<1>(grid_sync_ptr + 1, rank_idx, worker_idx, num_workers);

    // Distribute expert ranges across warps
    constexpr uint32_t kNumThreadsPerWarp = 32;
    const uint32_t warp_idx = thread_idx / kNumThreadsPerWarp;
    const uint32_t lane_idx = thread_idx % kNumThreadsPerWarp;
    const uint32_t global_warp_idx = warp_idx * num_workers + worker_idx;

    for (uint32_t expert_idx = global_warp_idx; expert_idx < config.num_experts; expert_idx += num_workers * (layout::kNumDispatchThreads / kNumThreadsPerWarp)) {
        const uint32_t count = asc_ldcg(total_expert_count + expert_idx);
        const uint32_t dst_rank_idx = expert_idx / kNumExpertsPerRank;
        const uint32_t local_expert_idx = expert_idx % kNumExpertsPerRank;
        const auto remote_base = layout::SymBuffer<>::map(buffer, dst_rank_idx, offsets);
        const uint64_t src_idx = static_cast<uint64_t>(expert_idx) * workspace.num_src_slots_per_expert;
        const uint64_t dst_idx = workspace.get_src_metadata_idx(local_expert_idx, rank_idx, 0);
        const auto remote_metadata = workspace.metadata.src_token_metadata.get_ptr(remote_base, dst_idx);
        const auto staged_indices = workspace.metadata.staged_token_topk_idx.get_ptr(buffer, src_idx);

        // Ship route weights with indices to avoid per-route remote Scalar reads.
        for (uint32_t slot_idx = lane_idx; slot_idx < count; slot_idx += kNumThreadsPerWarp) {
            const uint32_t token_topk_idx = asc_ldcg(staged_indices + slot_idx);

            const uint32_t weight_bits = asc_ldcg(route_weights + token_topk_idx);
            remote_metadata[slot_idx] = (static_cast<uint64_t>(weight_bits) << 32) | token_topk_idx;
        }
    }
    grid_sync<2>(grid_sync_ptr + 2, rank_idx, worker_idx, num_workers);

    // Publish counts after the metadata copy barrier
    if (worker_idx == 0) {
        for (uint32_t expert_idx = thread_idx; expert_idx < config.num_experts; expert_idx += layout::kNumDispatchThreads) {
            const uint32_t count = asc_ldcg(total_expert_count + expert_idx);
            const uint32_t dst_rank_idx = expert_idx / kNumExpertsPerRank;
            const uint32_t local_expert_idx = expert_idx % kNumExpertsPerRank;

            // Reset the peer prefix before publishing local expert 0's count
            if (kOverlapReduce and local_expert_idx == 0) {
                *workspace.metadata.combine_ready_prefix.get_ptr(buffer, dst_rank_idx) = 0;
                asc_threadfence();
            }

            const auto remote_base = layout::SymBuffer<>::map(buffer, dst_rank_idx, offsets);
            const uint64_t recv_count_idx = workspace.get_expert_recv_idx(rank_idx, local_expert_idx);
            *workspace.metadata.expert_recv_count.get_ptr(remote_base, recv_count_idx) = count;
            const auto remote_count = workspace.metadata.expert_recv_count_sum.get_ptr(remote_base, local_expert_idx);
            // Release the count store before publishing its arrival
            asc_threadfence();
            asc_atomic_add(remote_count, layout::kCountArrivalBit | count);
        }
        asc_threadfence();
    }
}

template <const auto& workspace, bool kFullTopology>
__aicore__ __forceinline__ uint32_t wait_dispatch_metadata(__gm__ uint8_t* buffer, uint32_t rank_idx, uint32_t core_idx,
                                                           __gm__ int32_t* cumulative_local_expert_recv_stats, CoreTopology<kFullTopology> topology) {
    constexpr auto& config = workspace.config;
    const auto num_m_blocks_certificate = workspace.metadata.num_m_blocks_certificate.get_ptr(buffer);

    // Wait for source counts and sum local M-blocks
    uint32_t num_contributed_m_blocks = 0;
    for (uint32_t expert_idx = core_idx; expert_idx < config.num_experts / config.num_ranks; expert_idx += topology.cores()) {
        const auto count_ptr = workspace.metadata.expert_recv_count_sum.get_ptr(buffer, expert_idx);
        uint32_t count;
        wait_until([&]() __aicore__ {
            return ((count = asc_load_dev(count_ptr)) & ~layout::kCountValueMask) == config.num_ranks * layout::kCountArrivalBit;
        }, [&]() __aicore__ {
            __asc_aicore::printf("MegaMoE metadata count timeout: rank=%u local_expert=%u arrivals=%u expected=%u\n",
                   rank_idx, expert_idx, count / layout::kCountArrivalBit, config.num_ranks);
        });
        count &= layout::kCountValueMask;
        // Accumulate received routes before padding; shared experts are excluded
        if (cumulative_local_expert_recv_stats != nullptr)
            asc_atomic_add(cumulative_local_expert_recv_stats + expert_idx, static_cast<int32_t>(count));
        num_contributed_m_blocks += ceil_div(count, layout::BLOCK_M);
    }
    asc_sync_data_barrier(DSB_DDR);

    // All AIV0 cores contribute, including idle cores
    asc_atomic_add(num_m_blocks_certificate, layout::kCountArrivalBit + num_contributed_m_blocks);
    uint32_t num_total_m_blocks;
    wait_until([&]() __aicore__ {
        return ((num_total_m_blocks = asc_load_dev(num_m_blocks_certificate)) & ~layout::kCountValueMask) == topology.cores() * layout::kCountArrivalBit;
    }, [&]() __aicore__ { print_local_timeout(rank_idx); });
    asc_sync_data_barrier(DSB_DDR);
    return num_total_m_blocks & layout::kCountValueMask;
}

} // namespace deep_gemm::mega_moe::comm
// END DeepSeek MIT port: include/deep_gemm/comm/mega_moe.hpp

// BEGIN DeepSeek MIT port: include/deep_gemm/scheduler/mega_moe.hpp


namespace deep_gemm::mega_moe::sched {

// kNone means no current task
enum class BlockPhase : uint32_t {kNone, kLinear1, kLinear2, kSharedLinear1, kSharedLinear2};

struct LinearTaskState {
    uint32_t shared_l1_task_base = 0;
    uint32_t l1_task_base = 0;
    uint32_t l2_task_base = 0;
    uint32_t shared_l2_task_base = 0;
};

// Map logical M-blocks to nonempty local experts
template <const auto& workspace>
struct ExpertCursor {
    uint32_t pool_block_end = 0;  // Exclusive M-block end
    uint32_t local_expert_idx = ~uint32_t{0};  // First increment wraps to expert 0
    uint32_t num_expert_tokens = 0;

    __aicore__ void next_expert(uint32_t num_tokens) {
        ++ local_expert_idx;
        num_expert_tokens = num_tokens;
        pool_block_end += ceil_div(num_tokens, layout::BLOCK_M);
    }

    // Balance FRAC_MN groups; expert M is compact, pool blocks keep the BLOCK_M stride
    // Return valid_m and write the expert-relative M start
    __aicore__ uint32_t get_m_block_range(uint32_t pool_block_idx, uint32_t& expert_m_begin) const {
        const uint32_t num_expert_m_blocks = ceil_div(num_expert_tokens, layout::BLOCK_M);
        const uint32_t expert_m_block_idx = pool_block_idx + num_expert_m_blocks - pool_block_end;
        const uint32_t num_m_groups = ceil_div(num_expert_tokens, FRAC_MN);
        const uint32_t num_groups_per_block = num_m_groups / num_expert_m_blocks;
        const uint32_t num_larger_m_blocks = num_m_groups % num_expert_m_blocks;

        expert_m_begin = (expert_m_block_idx * num_groups_per_block + min(expert_m_block_idx, num_larger_m_blocks)) * FRAC_MN;
        const uint32_t aligned_valid_m = (num_groups_per_block + (expert_m_block_idx < num_larger_m_blocks)) * FRAC_MN;
        return min(aligned_valid_m, num_expert_tokens - expert_m_begin);
    }

    // Advance the expert cursor and return valid rows
    __aicore__ uint32_t advance_to(__gm__ uint8_t* buffer, uint32_t pool_block_idx) {
        while (pool_block_idx >= pool_block_end)
            next_expert(asc_load_dev(workspace.metadata.expert_recv_count_sum.get_ptr(buffer, local_expert_idx + 1)) &
                        layout::kCountValueMask);

        uint32_t expert_m_begin;
        return get_m_block_range(pool_block_idx, expert_m_begin);
    }
};

// Replay the same routed task order on AIC and AIVs
// With kNone, metadata_ready distinguishes a pending schedule from an exhausted one
template <const auto& workspace, bool kFullTopology>
__aicore__ inline uint32_t take_linear_task(uint32_t num_shared_tokens, uint32_t num_total_m_blocks,
                                            LinearTaskState& state, uint32_t core_idx,
                                            BlockPhase& block_phase, CoreTopology<kFullTopology> topology, bool metadata_ready = true) {
    constexpr auto& config = workspace.config;
    constexpr uint32_t kNumLinear1NBlocks = config.intermediate_hidden / (layout::BLOCK_N / 2);
    constexpr uint32_t kNumLinear2NBlocks = config.hidden / layout::BLOCK_N;
    constexpr uint32_t kNumSharedLinear1NBlocks = config.shared_intermediate_hidden / (layout::BLOCK_N / 2);
    constexpr bool kHasSharedExperts = config.num_shared_experts > 0;

    // Schedule shared Linear1 before shared Linear2
    if constexpr (kHasSharedExperts) {
        const uint32_t num_shared_m_blocks = ceil_div(num_shared_tokens, layout::BLOCK_M);
        #pragma unroll
        for (uint32_t layer_idx = 0; layer_idx < 2; ++ layer_idx) {
            auto& task_base = layer_idx == 0 ? state.shared_l1_task_base : state.shared_l2_task_base;
            const uint32_t num_tasks = num_shared_m_blocks * (layer_idx == 0 ? kNumSharedLinear1NBlocks : kNumLinear2NBlocks);
            if (task_base >= num_tasks)
                continue;

            // Advance past waves with no task for this core
            const uint32_t task_idx = task_base + core_idx;
            task_base += topology.cores();
            if (task_idx < num_tasks) {
                block_phase = layer_idx == 0 ? BlockPhase::kSharedLinear1 : BlockPhase::kSharedLinear2;
                return task_idx;
            }
        }
    }

    // Routed tasks require metadata
    if (not metadata_ready) {
        block_phase = BlockPhase::kNone;
        return ~uint32_t{0};
    }

    // Prepare the next Linear2 wave with one Linear1 wave of lookahead
    const uint32_t num_routed_tasks[2]{num_total_m_blocks * kNumLinear1NBlocks, num_total_m_blocks * kNumLinear2NBlocks};

    while (state.l1_task_base < num_routed_tasks[0] or state.l2_task_base < num_routed_tasks[1]) {
        bool is_linear1 = state.l2_task_base >= num_routed_tasks[1];
        if (not is_linear1 and state.l1_task_base < num_routed_tasks[0]) {
            const uint32_t required_linear1_task_end =
                ceil_div(min(state.l2_task_base + topology.cores(), num_routed_tasks[1]), kNumLinear2NBlocks) *
                kNumLinear1NBlocks;
            constexpr uint32_t lookahead = 1;
            is_linear1 = state.l1_task_base < required_linear1_task_end + lookahead * topology.cores();
        }

        const uint32_t layer_idx = is_linear1 ? 0 : 1;
        auto& task_base = is_linear1 ? state.l1_task_base : state.l2_task_base;
        const uint32_t routed_lane = MOE_TOKENS <= 80 and kFullTopology ? (core_idx + 16) % kNumAICores : core_idx;
        const uint32_t task_idx = task_base + routed_lane;
        task_base += topology.cores();
        if (task_idx < num_routed_tasks[layer_idx]) {
            block_phase = is_linear1 ? BlockPhase::kLinear1 : BlockPhase::kLinear2;
            return task_idx;
        }
    }

    block_phase = BlockPhase::kNone;
    return ~uint32_t{0};
}

} // namespace deep_gemm::mega_moe::sched
// END DeepSeek MIT port: include/deep_gemm/scheduler/mega_moe.hpp

// BEGIN DeepSeek MIT port: include/deep_gemm/simd/mega_moe.hpp


#include <c_api/asc_simd.h>

namespace deep_gemm::mega_moe {

// FP4 -> FP8 NZ, scaled by 2^-6; weight SFs compensate with +6
// Rows map to GEMM N; columns map to K
template <uint32_t kNumRows, uint32_t kNumCols>
__simd_vf__ inline void vf_fp4_to_fp8_nz(__ubuf__ uint8_t* src, __ubuf__ uint8_t* dst) {
    constexpr uint32_t kNumPackedCols = kNumCols / 2;
    constexpr uint32_t kNumPaddedRows = kNumRows + 1;
    constexpr uint32_t kNumPackedColsPerVector = sizeof(vector_int8_t) / 2;  // Two FP4 values per byte
    constexpr uint32_t kNumVectorsPerRow = kNumPackedCols / kNumPackedColsPerVector;
    static_assert(kNumPackedCols % kNumPackedColsPerVector == 0);

    const auto mask = asc_create_mask_b8(PAT_ALL);
    vector_int8_t shift_right, shift_left, fp8_mask;
    asc_duplicate_scalar(shift_right, static_cast<int8_t>(2));
    asc_duplicate_scalar(reinterpret_cast<vector_uint16_t&>(shift_left), uint16_t{4});  // Alternating byte shifts: 4, 0
    asc_duplicate_scalar(fp8_mask, static_cast<int8_t>(0x9c));  // Sign and shifted FP4 payload

    auto src_ptr = reinterpret_cast<__ubuf__ int8_t*>(src);
    constexpr uint32_t kNumRowsPerPass = 4;
    static_assert(kNumRows % kNumRowsPerPass == 0);
    constexpr uint32_t kNumVectors = kNumVectorsPerRow * kNumRowsPerPass;

    // Each vector writes one row fragment in NZ order; the extra row pads the stride
    __ubuf__ uint8_t* dst_ptrs[kNumVectors];
    #pragma unroll
    for (uint32_t vector_idx = 0; vector_idx < kNumVectors; ++ vector_idx)
        dst_ptrs[vector_idx] = dst + vector_idx % kNumVectorsPerRow * kNumPaddedRows * 2 * kNumPackedColsPerVector +
                               vector_idx / kNumVectorsPerRow * get_frac_k<uint8_t>();

    for (uint32_t row_idx = 0; row_idx < kNumRows; row_idx += kNumRowsPerPass) {
        // Expand packed nibbles into scaled FP8 values
        vector_int8_t values[kNumVectors];
        #pragma unroll
        for (uint32_t vector_idx = 0; vector_idx < kNumVectors; ++ vector_idx) {
            asc_loadalign_upsample_postupdate(values[vector_idx], src_ptr, kNumPackedColsPerVector);
            asc_shiftleft(values[vector_idx], values[vector_idx], shift_left, mask);
            asc_shiftright(values[vector_idx], values[vector_idx], shift_right, mask);
            asc_and(values[vector_idx], values[vector_idx], fp8_mask, mask);
        }

        // Scatter into padded NZ and advance by four rows
        #pragma unroll
        for (uint32_t vector_idx = 0; vector_idx < kNumVectors; ++ vector_idx)
            asc_storealign_postupdate(dst_ptrs[vector_idx], reinterpret_cast<vector_uint8_t&>(values[vector_idx]),
                                      kNumPaddedRows, kNumRowsPerPass, mask);
    }
}

#if MOE_PACKED_ROUTED_WEIGHTS
// The static FP4 payload already follows Cube's NZ order. Each VF expands
// adjacent packed nibbles into an adjacent pair of E4M3 bytes.
__simd_vf__ inline void expand_scaled_linear(__ubuf__ int8_t* input, __ubuf__ int8_t* output) {
    const auto mask = asc_create_mask_b8(PAT_ALL);
    vector_int8_t left, right, keep;
    asc_duplicate_scalar(reinterpret_cast<vector_uint16_t&>(left), uint16_t{4});
    asc_duplicate_scalar(right, int8_t{2});
    asc_duplicate_scalar(keep, int8_t(0x9c));
    for (uint32_t panel = 0; panel < 8; ++panel) {
        for (uint32_t i = 0; i < 8; ++i) {
            #pragma unroll
            for (uint32_t j = 0; j < 4; ++j) {
                vector_int8_t value;
                asc_loadalign_upsample_postupdate(value, input, 128);
                asc_shiftleft(value, value, left, mask);
                asc_shiftright(value, value, right, mask);
                asc_and(value, value, keep, mask);
                asc_storealign_postupdate(output, value, 256, mask);
            }
        }
        output += 32;
    }
}

// Quantization uses ceil(log2(amax / 448)) for finite inputs. Its largest
// possible E8M0 code is 247, so increasing the activation exponent by six is exact.
__simd_vf__ inline void compensate_routed_scale(__ubuf__ uint8_t* scales, uint32_t count) {
    for (uint32_t offset = 0; offset < count; offset += sizeof(vector_uint8_t)) {
        uint32_t remaining = count - offset;
        auto mask = plt_b8(remaining, POST_UPDATE);
        vector_uint8_t original, shifted;
        asc_loadalign(original, scales + offset);
        asc_add_scalar(shifted, original, uint8_t{6}, mask);
        vector_bool is_nan;
        vcmps_eq(is_nan, original, uint8_t{255}, mask);
        vsel(shifted, original, shifted, is_nan);
        asc_storealign(scales + offset, shifted, mask);
    }
}
#endif

// Linear1 epilogue: SwiGLU and FP8 quantization
template <bool kApplyRouteWeight, uint32_t STORE_BLOCK_M>
__simd_vf__ inline void vf_swiglu_quant(__ubuf__ bfloat16_t* input, __ubuf__ float8_e4m3_t* output,
                                        __ubuf__ int16_t* output_sf, __ubuf__ float* route_weights,
                                        uint32_t valid_m, float activation_clamp) {
    constexpr uint32_t kNumRowsPerIteration = 2;
    constexpr uint32_t kNumPairsPerRow = (layout::BLOCK_N / 2) / MX_SF_DIVISOR;
    constexpr uint32_t kNumPairs = kNumRowsPerIteration * kNumPairsPerRow;
    static_assert(layout::BLOCK_N == 256);

    const vector_bool group_mask_b32 = asc_create_mask_b32(PAT_VL32);
    const vector_bool pair_mask_b32 = asc_create_mask_b32(PAT_VL64);
    vector_bfloat16_t zero_bf16;
    asc_duplicate_scalar(zero_bf16, 0);
    vector_int32_t arange_b32;
    vector_uint32_t high_group_idx;
    asc_arange(arange_b32, 0);
    asc_add_scalar(high_group_idx, reinterpret_cast<vector_uint32_t&>(arange_b32), MX_SF_DIVISOR / 2, group_mask_b32);
    vector_uint32_t exponent_mask, inv_scale_bias;
    asc_duplicate_scalar(exponent_mask, 0xff800000u);
    asc_duplicate_scalar(inv_scale_bias, 0x7f000000u);  // (2 * 127) << 23

    auto gate_ptr = input;
    auto up_ptr = input + layout::BLOCK_N / 2;
    auto output_ptr = reinterpret_cast<__ubuf__ uint8_t*>(output);

    for (uint32_t m_idx = 0; m_idx < valid_m; m_idx += kNumRowsPerIteration) {
        // Load gate/up as FP32
        vector_float gate[kNumPairs], up[kNumPairs], value[kNumPairs];
        #pragma unroll
        for (uint32_t m_offset = 0; m_offset < kNumRowsPerIteration; ++ m_offset) {
            const uint32_t pair_idx = m_offset * kNumPairsPerRow;
            vector_bfloat16_t gate_bf16, up_bf16;
            if (m_idx + m_offset < valid_m) {
                asc_loadalign_postupdate(gate_bf16, gate_ptr, layout::BLOCK_N);
                asc_loadalign_postupdate(up_bf16, up_ptr, layout::BLOCK_N);
                asc_intlv(reinterpret_cast<vector_bfloat16_t&>(gate[pair_idx]),
                          reinterpret_cast<vector_bfloat16_t&>(gate[pair_idx + 1]), zero_bf16, gate_bf16);
                asc_intlv(reinterpret_cast<vector_bfloat16_t&>(up[pair_idx]),
                          reinterpret_cast<vector_bfloat16_t&>(up[pair_idx + 1]), zero_bf16, up_bf16);
            } else {
                asc_duplicate_scalar(gate[pair_idx], 0.0f);
                asc_duplicate_scalar(gate[pair_idx + 1], 0.0f);
                asc_duplicate_scalar(up[pair_idx], 0.0f);
                asc_duplicate_scalar(up[pair_idx + 1], 0.0f);
            }
        }

        // Load only route weights belonging to valid rows.
        vector_float route_weight[kNumRowsPerIteration];
        if constexpr (kApplyRouteWeight) {
            asc_loadalign_brc_postupdate(route_weight[0], route_weights, 1);
            if (m_idx + 1 < valid_m)
                asc_loadalign_brc_postupdate(route_weight[1], route_weights, 1);
            else
                asc_duplicate_scalar(route_weight[1], 0.0f);
        }

        // Clamp gate above and up on both sides
        #pragma unroll
        for (uint32_t pair_idx = 0; pair_idx < kNumPairs; ++ pair_idx) {
            asc_min_scalar(gate[pair_idx], gate[pair_idx], activation_clamp, pair_mask_b32);
            asc_min_scalar(up[pair_idx], up[pair_idx], activation_clamp, pair_mask_b32);
            asc_max_scalar(up[pair_idx], up[pair_idx], -activation_clamp, pair_mask_b32);
            asc_mul_scalar(value[pair_idx], gate[pair_idx], -1.0f, pair_mask_b32);
            asc_exp(value[pair_idx], value[pair_idx], pair_mask_b32);
        }

        // Preserve the original FP32 order: (gate / (1 + exp(-gate))) * up * route_weight.
        #pragma unroll
        for (uint32_t pair_idx = 0; pair_idx < kNumPairs; ++ pair_idx) {
            asc_add_scalar(value[pair_idx], value[pair_idx], 1.0f, pair_mask_b32);
            asc_div(value[pair_idx], gate[pair_idx], value[pair_idx], pair_mask_b32);
            asc_mul(value[pair_idx], value[pair_idx], up[pair_idx], pair_mask_b32);
            if constexpr (kApplyRouteWeight)
                asc_mul(value[pair_idx], value[pair_idx], route_weight[pair_idx / kNumPairsPerRow], pair_mask_b32);
        }

        // Reduce amax for both 32-channel groups in each SF pair
        #pragma unroll
        for (uint32_t pair_idx = 0; pair_idx < kNumPairs; ++ pair_idx) {
            const uint32_t m_offset = pair_idx / kNumPairsPerRow;
            const uint32_t sf_pair_idx = pair_idx % kNumPairsPerRow;
            if (m_idx + m_offset >= valid_m) continue;
            vector_float maximum[2];
            vector_uint32_t bits[2];
            asc_abs(maximum[0], value[pair_idx], pair_mask_b32);
            asc_gather(reinterpret_cast<vector_uint32_t&>(maximum[1]),
                       reinterpret_cast<vector_uint32_t&>(maximum[0]), high_group_idx);
            #pragma unroll
            for (uint32_t group_idx = 0; group_idx < 2; ++ group_idx) {
                asc_reduce_max(maximum[group_idx], maximum[group_idx], group_mask_b32);
                asc_duplicate(maximum[group_idx], maximum[group_idx], pair_mask_b32);
            }

            // Keep the original rounded FP32 reciprocal multiply before ceil encoding.
            asc_select(maximum[0], maximum[0], maximum[1], group_mask_b32);
            asc_max_scalar(maximum[0], maximum[0], 1e-4f, pair_mask_b32);
            asc_mul_scalar(maximum[0], maximum[0], 1.0f / 448.0f, pair_mask_b32);
            asc_add_scalar(bits[0], reinterpret_cast<vector_uint32_t&>(maximum[0]), 0x007fffffu, pair_mask_b32);
            asc_and(bits[0], bits[0], exponent_mask, pair_mask_b32);
            vector_uint32_t inv_scale_bits;
            asc_sub(inv_scale_bits, inv_scale_bias, bits[0], pair_mask_b32);
            asc_mul(value[pair_idx], value[pair_idx], reinterpret_cast<vector_float&>(inv_scale_bits), pair_mask_b32);
            if constexpr (not kApplyRouteWeight) {
                asc_max_scalar(value[pair_idx], value[pair_idx], -448.0f, pair_mask_b32);
                asc_min_scalar(value[pair_idx], value[pair_idx], 448.0f, pair_mask_b32);
            }

            // Store FP8 and UE8M0 scales
            vector_fp8_e4m3fn_t value_fp8;
            asc_float2e4m3_rn_sat(value_fp8, value[pair_idx], pair_mask_b32, ASC_DISPERSE_FIRST_QUARTER);
            asc_storealign_pack_quarter(reinterpret_cast<__ubuf__ uint32_t*>(output_ptr),
                                        reinterpret_cast<vector_uint32_t&>(value_fp8), pair_mask_b32);
            output_ptr += MX_SF_DIVISOR;

            asc_shiftright_scalar(bits[0], bits[0], 23, pair_mask_b32);
#if MOE_PACKED_ROUTED_WEIGHTS
            if constexpr (kApplyRouteWeight) {
                // B payloads are widened to FP8 / 64. Apply 64 to this activation
                // scale after computing its FP8 payload, only for the routed path.
                vector_bool is_nan;
                vcmps_eq(is_nan, bits[0], uint32_t{255}, pair_mask_b32);
                vector_uint32_t shifted;
                asc_add_scalar(shifted, bits[0], uint32_t{6}, pair_mask_b32);
                asc_select(bits[0], bits[0], shifted, is_nan);
            }
#endif
            asc_gather(bits[1], bits[0], high_group_idx);
            vector_uint8_t packed_sf_half, unused_sf_half;
            asc_intlv(packed_sf_half, unused_sf_half, reinterpret_cast<vector_uint8_t&>(bits[0]),
                      reinterpret_cast<vector_uint8_t&>(bits[1]));
            asc_storealign_1st(output_sf + sf_pair_idx * STORE_BLOCK_M,
                               reinterpret_cast<vector_int16_t&>(packed_sf_half), m_idx + m_offset);
        }
    }
}

// Combine: FP32 sum -> BF16; input stride is num_channels
__simd_vf__ inline void vf_combine_reduce(__ubuf__ bfloat16_t* inputs, uint32_t num_inputs, uint32_t num_channels) {
    constexpr uint32_t kNumChannelsPerVector = sizeof(vector_bfloat16_t) / sizeof(bfloat16_t);
    const auto mask_b16 = asc_create_mask_b16(PAT_ALL);
    const auto mask_b32 = asc_create_mask_b32(PAT_ALL);
    vector_bfloat16_t zero_bf16;
    asc_duplicate_scalar(zero_bf16, 0);
    auto output_ptr = inputs;

    for (uint32_t channel_idx = 0; channel_idx < num_channels; channel_idx += kNumChannelsPerVector) {
        vector_float sum0, sum1;
        asc_duplicate_scalar(sum0, 0.0f);
        asc_duplicate_scalar(sum1, 0.0f);
        auto input_ptr = inputs + channel_idx;
        if (num_inputs > 0) {
            vector_bfloat16_t first;
            asc_loadalign_postupdate(first, input_ptr, num_channels);
            asc_intlv(reinterpret_cast<vector_bfloat16_t&>(sum0),
                      reinterpret_cast<vector_bfloat16_t&>(sum1), zero_bf16, first);
        }
        for (uint32_t input_idx = 1; input_idx < num_inputs; ++ input_idx) {
            vector_bfloat16_t value;
            vector_float value0, value1;
            asc_loadalign_postupdate(value, input_ptr, num_channels);

            // Widen BF16 exactly with zero low bits
            asc_intlv(reinterpret_cast<vector_bfloat16_t&>(value0), reinterpret_cast<vector_bfloat16_t&>(value1), zero_bf16, value);
            asc_add(sum0, sum0, value0, mask_b32);
            asc_add(sum1, sum1, value1, mask_b32);
        }

        // Round to BF16 and pack both halves
        vector_bfloat16_t output0, output1, output, unused;
        asc_float2bfloat16_rn(output0, sum0, mask_b32, ASC_POSITION_EVEN);
        asc_float2bfloat16_rn(output1, sum1, mask_b32, ASC_POSITION_EVEN);
        asc_deintlv(output, unused, output0, output1);
        asc_storealign_postupdate(output_ptr, output, kNumChannelsPerVector, mask_b16);
    }
}

} // namespace deep_gemm::mega_moe
// END DeepSeek MIT port: include/deep_gemm/simd/mega_moe.hpp

// BEGIN DeepSeek MIT port: include/deep_gemm/utils/math.hpp


namespace deep_gemm {

// Smallest integer >= value that is coprime to modulus
// Requires 0 < value <= modulus < UINT32_MAX
__aicore__ __attribute__((always_inline)) constexpr uint32_t next_coprime(uint32_t value, uint32_t modulus) {
    while (true) {
        uint32_t a = value, b = modulus;
        while (b != 0) {
            const uint32_t remainder = a % b;
            a = b;
            b = remainder;
        }
        if (a == 1)
            break;
        ++ value;
    }
    return value;
}

} // namespace deep_gemm
// END DeepSeek MIT port: include/deep_gemm/utils/math.hpp

// BEGIN PyPTO adapter: original expert weight layout

/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include <cstdint>
#include <pto/pto-inst.hpp>

namespace deep_gemm::mega_moe::original_weights {

// Public FP4 bytes are [N256 panel, K256 panel, K256, N128 packed bytes].
// No whole-weight GM conversion or scale bias is introduced by these loads.
AICORE inline uint64_t fp4_offset(uint32_t expert, uint32_t input_k, uint32_t output_n,
                                 uint32_t column, uint32_t k_begin) {
    return static_cast<uint64_t>(expert) * input_k * output_n / 2 +
           static_cast<uint64_t>(column / 256) * (input_k / 256) * 32768 +
           static_cast<uint64_t>(k_begin / 256) * 32768 + (column % 256) / 2;
}

#if defined(__DAV_VEC__) || defined(__CPU_SIM)
template <uint32_t BlockK, uint32_t PackedColumns, uint32_t LoadPackedColumns>
AICORE inline void load_fp4_columns(uint32_t raw_address, __gm__ const uint8_t* weights,
                                    uint32_t expert, uint32_t input_k, uint32_t output_n,
                                    uint32_t column, uint32_t k_begin, uint32_t valid_k,
                                    uint32_t packed_column_offset = 0,
                                    asc_load_l2_cache_mode cache_hint = asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM) {
    using Raw = pto::Tile<pto::TileType::Vec, uint8_t, 256, PackedColumns,
                          pto::BLayout::RowMajor, pto::DYNAMIC, LoadPackedColumns>;
    using Global = pto::GlobalTensor<uint8_t, pto::Shape<1, 1, 1, pto::DYNAMIC, LoadPackedColumns>,
                                    pto::Stride<1, 1, 1, 128, 1>>;
    for (uint32_t offset = 0; offset < valid_k; offset += 256) {
        const uint32_t rows = valid_k - offset < 256 ? valid_k - offset : 256;
        Raw raw(rows);
        pto::TASSIGN(raw, raw_address + offset * PackedColumns + packed_column_offset);
        auto* source = const_cast<__gm__ uint8_t*>(weights) +
                       fp4_offset(expert, input_k, output_n, column, k_begin + offset);
        if constexpr (MOE_TOKENS <= 80) {
            asc_copy_gm2ub_align(reinterpret_cast<__ubuf__ uint8_t*>(raw_address + offset * PackedColumns + packed_column_offset),
                source, rows, LoadPackedColumns, 0, 0, false, cache_hint, 128, PackedColumns);
        } else {
            Global global(source, typename Global::Shape(1, 1, 1, rows, LoadPackedColumns));
            pto::TLOAD(raw, global);
        }
    }
}

template <uint32_t BlockK, uint32_t MadN>
AICORE inline void load_routed_panel(uint32_t raw_address, __gm__ const uint8_t* first,
                                     __gm__ const uint8_t* up, bool linear1,
                                     uint32_t expert, uint32_t input_k, uint32_t output_n,
                                     uint32_t n_block, uint32_t n_mad, uint32_t k_begin,
                                     uint32_t valid_k, asc_load_l2_cache_mode cache_hint) {
    static_assert(BlockK * MadN == 65536);
    constexpr uint32_t PackedColumns = MadN / 2;
    if constexpr (BlockK == 512) {
        if (valid_k < BlockK) {
            // Only the unused K tail is zeroed. The MTE2 head copy is disjoint,
            // and this V operation precedes the following decode in V order.
            using Tail = pto::Tile<pto::TileType::Vec, uint8_t, 256, PackedColumns>;
            Tail tail;
            pto::TASSIGN(tail, raw_address + 256 * PackedColumns);
            pto::TEXPANDS(tail, uint8_t{0});
        }
    }
#if MOE_PACKED_ROUTED_WEIGHTS
    static_assert(BlockK == 256 && MadN == 256);
    // Even and odd gate/up panels live in separate shards. Each 32 KiB
    // panel holds four gate N32 fragments followed by four up fragments.
    const uint8_t __gm__* shard = linear1 && (n_block & 1) ? up : first;
    const uint32_t column = (linear1 ? n_block / 2 : n_block) * 256;
    auto* source = const_cast<__gm__ uint8_t*>(shard) +
                   fp4_offset(expert, input_k, output_n, column, k_begin);
    asc_copy_gm2ub_align(reinterpret_cast<__ubuf__ uint8_t*>(raw_address), source,
                         1, valid_k * PackedColumns, 0, 0, false, cache_hint,
                         valid_k * PackedColumns, valid_k * PackedColumns);
    return;
#endif
    if (linear1) {
        const uint32_t column = n_block * 128;
        if constexpr (MadN == 128) {
            load_fp4_columns<BlockK, PackedColumns, 64>(raw_address, n_mad == 0 ? first : up,
                expert, input_k, output_n, column, k_begin, valid_k, 0, cache_hint);
        } else {
            load_fp4_columns<BlockK, PackedColumns, 64>(raw_address, first,
                expert, input_k, output_n, column, k_begin, valid_k, 0, cache_hint);
            load_fp4_columns<BlockK, PackedColumns, 64>(raw_address, up,
                expert, input_k, output_n, column, k_begin, valid_k, 64, cache_hint);
        }
    } else {
        load_fp4_columns<BlockK, PackedColumns, PackedColumns>(raw_address, first,
            expert, input_k, output_n, n_block * 256 + n_mad, k_begin, valid_k, 0, cache_hint);
    }
}

template <uint32_t BlockK, uint32_t MadN>
AICORE inline void decode_routed_panel(uint32_t raw_address, uint32_t nz_address, bool shifted_scale = false) {
#if MOE_PACKED_ROUTED_WEIGHTS
    static_assert(BlockK * MadN == 65536);
    deep_gemm::mega_moe::expand_scaled_linear(reinterpret_cast<__ubuf__ int8_t*>(raw_address),
                                              reinterpret_cast<__ubuf__ int8_t*>(nz_address));
    return;
#endif
    using Packed = pto::Tile<pto::TileType::Vec, float4_e2m1x2_t, BlockK, MadN>;
    using NZ = pto::Tile<pto::TileType::Vec, float8_e4m3_t, BlockK, MadN,
                         pto::BLayout::ColMajor, BlockK, MadN, pto::SLayout::RowMajor,
                         512, pto::PadValue::Null, pto::CompactMode::RowPlusOne>;
    static_assert(NZ::StorageBytes == (BlockK + 1) * MadN);
    Packed packed;
    NZ nz;
    pto::TASSIGN(packed, raw_address);
    pto::TASSIGN(nz, nz_address);
#if MOE_TOKENS <= 80
    // B / 64 pairs with the guarded +6 exponent shift in streamed W2 scales.
    if (shifted_scale)
        deep_gemm::mega_moe::vf_fp4_to_fp8_nz<BlockK, MadN>(
            reinterpret_cast<__ubuf__ uint8_t*>(raw_address), reinterpret_cast<__ubuf__ uint8_t*>(nz_address));
    else
#endif
        pto::TCVT(nz, packed, pto::RoundMode::CAST_RINT);
}
#endif

#if defined(__DAV_CUBE__) || defined(__CPU_SIM)
template <uint32_t Columns>
AICORE inline void load_scale_columns(uint32_t destination, __gm__ const uint8_t* scales,
                                      uint32_t experts, uint32_t input_k, uint32_t output_n,
                                      uint32_t expert, uint32_t column, uint32_t k_begin,
                                      uint32_t valid_pairs, asc_load_l2_cache_mode cache_hint) {
    constexpr uint32_t ScaleRows = 64;
    using Scale = pto::Tile<pto::TileType::Mat, float8_e8m0_t, ScaleRows, Columns,
                            pto::BLayout::ColMajor, pto::DYNAMIC, Columns,
                            pto::SLayout::ColMajor, 32>;
    using Global = pto::GlobalTensor<float8_e8m0_t,
        pto::TileShape2D<float8_e8m0_t, pto::DYNAMIC, Columns, pto::Layout::MX_B_NN>,
        pto::BaseShape2D<float8_e8m0_t, pto::DYNAMIC, pto::DYNAMIC, pto::Layout::MX_B_NN>,
        pto::Layout::MX_B_NN>;
    const uint64_t scale_rows = static_cast<uint64_t>(experts) * (input_k / 32);
    const uint64_t offset = column * scale_rows + (expert * (input_k / 32) + k_begin / 32) * 16;
    Scale target(valid_pairs * 2);
    pto::TASSIGN(target, destination);
    Global source(reinterpret_cast<__gm__ float8_e8m0_t*>(const_cast<__gm__ uint8_t*>(scales)) + offset,
                  typename Global::Shape(valid_pairs * 2, Columns),
                  typename Global::Stride(scale_rows, output_n));
    if constexpr (MOE_TOKENS <= 80) {
        if (cache_hint == asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM)
            pto::TLOAD<pto::TLoadL2Hint::NormalFirstVictim>(target, source);
        else
            pto::TLOAD<pto::TLoadL2Hint::NotAllocKeep>(target, source);
    } else {
        pto::TLOAD(target, source);
    }
}

AICORE inline void load_weight_scales(uint32_t destination, __gm__ const uint8_t* first,
                                      __gm__ const uint8_t* up, bool linear1,
                                      uint32_t experts, uint32_t input_k, uint32_t output_n,
                                      uint32_t expert, uint32_t n_block, uint32_t k_begin,
                                      uint32_t valid_pairs, asc_load_l2_cache_mode cache_hint) {
    if (linear1) {
        load_scale_columns<128>(destination, first, experts, input_k, output_n,
                                expert, n_block * 128, k_begin, valid_pairs, cache_hint);
        load_scale_columns<128>(destination + 64 * 128, up, experts, input_k, output_n,
                                expert, n_block * 128, k_begin, valid_pairs, cache_hint);
    } else {
        load_scale_columns<256>(destination, first, experts, input_k, output_n,
                                expert, n_block * 256, k_begin, valid_pairs, cache_hint);
    }
}

template <uint32_t BlockK, uint32_t Columns>
AICORE inline void load_shared_columns(uint32_t destination, __gm__ const uint8_t* weights,
                                       uint32_t input_k, uint32_t output_n, uint32_t column,
                                       uint32_t k_begin, uint32_t valid_k) {
    using Matrix = pto::Tile<pto::TileType::Mat, float8_e4m3_t, BlockK, Columns,
                             pto::BLayout::ColMajor, pto::DYNAMIC, Columns,
                             pto::SLayout::RowMajor, 512>;
    using Global = pto::GlobalTensor<float8_e4m3_t, pto::Shape<1, 1, 1, pto::DYNAMIC, Columns>,
                                    pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>>;
    Matrix target(valid_k);
    pto::TASSIGN(target, destination);
    Global source(reinterpret_cast<__gm__ float8_e4m3_t*>(const_cast<__gm__ uint8_t*>(weights)) +
                      static_cast<uint64_t>(k_begin) * output_n + column,
                  typename Global::Shape(1, 1, 1, valid_k, Columns),
                  typename Global::Stride(input_k * output_n, input_k * output_n,
                                         input_k * output_n, output_n, 1));
    pto::TLOAD(target, source);
}

template <uint32_t BlockK>
AICORE inline void load_shared_panel(uint32_t destination, __gm__ const uint8_t* first,
                                     __gm__ const uint8_t* up, bool linear1,
                                     uint32_t input_k, uint32_t output_n, uint32_t n_block,
                                     uint32_t k_begin, uint32_t valid_k) {
    if (linear1) {
        load_shared_columns<BlockK, 128>(destination, first, input_k, output_n,
                                         n_block * 128, k_begin, valid_k);
        load_shared_columns<BlockK, 128>(destination + BlockK * 128, up, input_k, output_n,
                                         n_block * 128, k_begin, valid_k);
    } else {
        load_shared_columns<BlockK, 256>(destination, first, input_k, output_n,
                                         n_block * 256, k_begin, valid_k);
    }
}
#endif

}  // namespace deep_gemm::mega_moe::original_weights
// END PyPTO adapter: original expert weight layout

/*
 * Operation-private DeepSeek V4.1 MoE pipeline port.
 *
 * The MIT notice below applies only to sections marked "DeepSeek MIT port"
 * in this file. Those sections are adapted from
 * https://github.com/deepseek-ai/DeepGEMM-Ascend at revision
 * 3a35326271b069b8b5bf1a754435acd6cc91575b.
 * Sections marked "PyPTO adapter" retain the CANN Open Software License
 * Agreement Version 2.0 and are outside this MIT notice's scope.
 *
 * MIT License
 *
 * Copyright (c) 2026 DeepSeek
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */


// BEGIN DeepSeek MIT port: include/deep_gemm/mega_moe.hpp

// Adapted from https://github.com/deepseek-ai/DeepGEMM-Ascend
// Revision: 3a35326271b069b8b5bf1a754435acd6cc91575b (MIT; see the local notice above).


#include <c_api/asc_simd.h>
#include <c_api/cache_ctrl/cache_ctrl.h>
#include <simt_api/device_functions.h>
#include <simt_api/common_functions.h>

namespace deep_gemm::mega_moe {

using layout::BLOCK_M;
using layout::BLOCK_N;

// Double-buffer L0A/B; L1 has two or three A stages and four routed B stages.
constexpr uint32_t kNumL0Stages = 2;
#if MOE_STREAM_W2_SCALES
constexpr uint32_t kNumL1AStages = 2;
#else
constexpr uint32_t kNumL1AStages = 3;
#endif
constexpr uint32_t kNumL1BStages = 4, kNumSFStages = 2;
constexpr uint32_t kL1BBase = 3 * L0ASizeBytes;

// Split each logical M-block between two FIX/epilogue buffers
constexpr uint32_t kNumEpilogueStages = 2;
constexpr uint32_t STORE_BLOCK_M = BLOCK_M / kNumEpilogueStages;

// FP8 compute shapes
// BLOCK_M  BLOCK_N  BLOCK_K  MAD_M  MAD_N  MAD_K
//     256      256      512    128    128    128
//     256      256      256    256    256    128
template <uint32_t kMadM>
struct MMAConfig {
    static constexpr uint32_t BLOCK_M = layout::BLOCK_M;
    static constexpr uint32_t BLOCK_N = layout::BLOCK_N;
    #if MOE_DECODE_K256
    // Keep the same L1B stage footprint when full M blocks and a partial
    // expert block alternate within the same dispatch.
    static constexpr uint32_t BLOCK_K = 256;
#else
    static constexpr uint32_t BLOCK_K = MOE_TOKENS <= 80 ? 256 : L0ASizeBytes / kMadM;
#endif
    static constexpr uint32_t MAD_M = kMadM;
    static constexpr uint32_t MAD_K = 128;
    static constexpr uint32_t MAD_N = L0BSizeBytes / BLOCK_K;
    static_assert(BLOCK_N % MAD_N == 0);
    static_assert(MAD_M * BLOCK_N * sizeof(float) <= L0CSizeBytes);

    // Shared B reuses routed B's region; events 5-7 permit at most three stages
    static constexpr __aicore__ uint32_t get_num_shared_l1b_stages() {
        constexpr uint32_t num_slots = kNumL1BStages * L0BSizeBytes / (BLOCK_N * BLOCK_K);
        return num_slots < 3 ? num_slots : 3;
    }
};

// AIC layout (KiB); shared and routed B occupy the same L1 region
// Memory  Region       Offset  Stages x size
// L1      A                 0      2 x 64 (streamed W2 SF), otherwise 3 x 64
//         W2 SF           128      2 x 16 (streamed W2 SF only)
//         routed B        192      4 x 64
//         shared B        192      2 x 128 (M=128), 3 x 64 (M=256)
//         A SF            448      2 x 16
//         B SF            480      2 x 16
// L0A     A                 0      2 x 32
// L0B     B                 0      2 x 32
// L0C     FP32 accum        0      2 x 64 (M=128), 1 x 256 (M=256)

// AIC: L1A 0–1 or 0–2, SF 3–4, shared L1B 5–6 or 5–7
constexpr PipeEvent<PIPE_MTE2, PIPE_MTE1> kL1AFull{0}, kSFFull{3}, kSharedL1BFull{5};
constexpr PipeEvent<PIPE_MTE1, PIPE_MTE2> kL1AEmpty{0}, kSFEmpty{3}, kSharedL1BEmpty{5};

// AIC: A uses 0–1; split-N B uses 2–3, otherwise readiness shares A's events
constexpr PipeEvent<PIPE_MTE1, PIPE_M> kL0AFull{0}, kL0BFull{2};
constexpr PipeEvent<PIPE_M, PIPE_MTE1> kL0AEmpty{0}, kL0BEmpty{2};

// AIV0: FP4 has 3 stages at 0; NZ has 2 stages at 0
constexpr PipeEvent<PIPE_MTE2, PIPE_V> kFP4Full{0};
constexpr PipeEvent<PIPE_V, PIPE_MTE2> kFP4Empty{0};
constexpr PipeEvent<PIPE_V, PIPE_MTE3> kNZFull{0};
constexpr PipeEvent<PIPE_MTE3, PIPE_V> kNZEmpty{0};
constexpr PipeEvent<PIPE_V, PIPE_S> kMetadataDone{0};

// AIV1: pull 0–1 and push 2–5, full forward and empty backward
constexpr uint32_t kNumPullStages = 2, kNumPushStages = 4;
constexpr PipeEvent<PIPE_MTE2, PIPE_MTE3> kPullFull{0}, kPushFull{2};
constexpr PipeEvent<PIPE_MTE3, PIPE_MTE2> kPullEmpty{0}, kPushEmpty{2};

// AIV1: top-k weights, epilogue output, and store completion
constexpr PipeEvent<PIPE_MTE2, PIPE_V> kTopkFull{1};
constexpr PipeEvent<PIPE_V, PIPE_MTE2> kTopkEmpty{0};
constexpr PipeEvent<PIPE_V, PIPE_MTE3> kQuantFull{0};
constexpr PipeEvent<PIPE_MTE3, PIPE_S> kPullDone{0}, kStoresDone{2};

// AIV0/AIV1: reduce stages 6–7 and the final communication drain
constexpr uint32_t kNumReduceStages = 2;
constexpr PipeEvent<PIPE_MTE2, PIPE_V> kReduceFull{6};
constexpr PipeEvent<PIPE_V, PIPE_MTE3> kReduceOutputFull{6};
constexpr PipeEvent<PIPE_MTE3, PIPE_MTE2> kReduceEmpty{6};
constexpr PipeEvent<PIPE_MTE3, PIPE_S> kCommDone{3};

// AIC/AIV0: MTE3 -> MTE1 full, MTE1 -> MTE3 empty
// Metadata uses flag 2 before routed L1B stages 0–3
constexpr IntraBlockEvent<0> kMetadataFull{2};
constexpr IntraBlockEvent<0> kL1BFull{0}, kL1BEmpty{0};
// AIV0/AIC streamed W2 scales use separate intra-block flags 4–5.
constexpr IntraBlockEvent<0> kStreamSFFull{4}, kStreamSFEmpty{4};
constexpr PipeEvent<PIPE_MTE2, PIPE_V> kStreamSFLoaded{5};
constexpr PipeEvent<PIPE_V, PIPE_S> kStreamSFChecked{5};
constexpr PipeEvent<PIPE_V, PIPE_MTE3> kStreamSFConverted{5};
constexpr PipeEvent<PIPE_MTE3, PIPE_MTE2> kStreamSFUBEmpty{5};

// AIC/AIV1: FIX -> V/S full, MTE3/S -> FIX empty
// Linear1 and Linear2 share the same epilogue credits
constexpr IntraBlockEvent<1> kEpilogueFull{0}, kEpilogueEmpty{0};

// Block sync: both AIVs' MTE3 -> AIC Scalar after final output stores
constexpr uint32_t kCombineDoneID = 4;

// CANN 9.2 forwards mode 12 but does not name write-through-share in its C API enum
constexpr auto kWriteThroughShare = static_cast<asc_store_l2_cache_mode>(12);

template <const auto& workspace, bool kFullTopology>
struct CombineReduceState {
    static constexpr uint32_t kNumBitsPerWord = sizeof(uint32_t) * 8;
    static constexpr uint32_t kUninitializedStage = ~0u;

    // Resume at a channel boundary
    uint32_t token_idx, channel_idx = 0, stage_idx = kUninitializedStage;
    uint32_t num_remaining_tokens;
    __gm__ uint32_t* local_route_ids;

    uint32_t valid_topk_mask[ceil_div(workspace.config.num_topk, kNumBitsPerWord)]{};
    bool shared_output_ready = false;
    uint32_t blocking_rank_idx = 0;
    uint64_t completed_blocks[workspace.config.num_ranks]{};

    // One bit per owned token, strided by the AIV count
    uint32_t reduced_token_mask[ceil_div(workspace.config.num_max_tokens_per_rank,
        2 * CoreTopology<kFullTopology>::kMinCores * kNumBitsPerWord)]{};
};

// Describe staged UB scratch without allocating memory
template <const auto& workspace>
__aicore__ constexpr auto make_reduce_buffer(uint32_t addr = 0) {
    constexpr auto& config = workspace.config;
    constexpr uint32_t kNumInputs = config.num_topk + (config.num_shared_experts > 0);
    constexpr uint32_t kNumChannelsPerVector = sizeof(vector_bfloat16_t) / sizeof(bfloat16_t);
    const uint32_t num_channels = (UBSizeBytes - addr) / (kNumReduceStages * kNumInputs * sizeof(bfloat16_t)) /
        kNumChannelsPerVector * kNumChannelsPerVector;
    return ub_ptr<bfloat16_t>(kNumInputs, config.hidden < num_channels ? config.hidden : num_channels, addr);
}

// True when all tokens are processed; each step issues one output chunk
template <const auto& workspace, bool kOverlapReduce, bool kFullTopology>
__aicore__ inline bool advance_combine_reduce(__gm__ uint8_t* buffer, __gm__ bfloat16_t* output,
                                      uint32_t num_tokens, CombineReduceState<workspace, kFullTopology>& state,
                                      const ub_ptr<bfloat16_t>& ub_reduce_inputs, CoreTopology<kFullTopology> topology,
                                      uint32_t num_steps = ~0u) {
    constexpr auto& config = workspace.config;
    constexpr uint32_t kNumLinear2NBlocks = config.hidden / BLOCK_N;
    constexpr uint32_t kNumExpertsPerRank = config.num_experts / config.num_ranks;
    constexpr bool kHasSharedExperts = config.num_shared_experts > 0;

    const uint32_t num_chunk_channels = ub_reduce_inputs.shape_n;

    const auto combine_input = workspace.combine_input.get_ptr(buffer);
    const auto topk_idx = state.local_route_ids;
    auto& stage_idx = state.stage_idx;

    // Wait for all active shared Linear2 cores
    if (kOverlapReduce and kHasSharedExperts and state.num_remaining_tokens > 0 and not state.shared_output_ready) {
        state.shared_output_ready = comm::is_block_ready<workspace>(buffer, layout::kLinear2Ready, 0,
            min(ceil_div(num_tokens, BLOCK_M) * kNumLinear2NBlocks, topology.cores()));
        if (not state.shared_output_ready)
            return false;
    }

    while (state.num_remaining_tokens > 0) {
        const uint32_t token_idx = state.token_idx;
        const uint32_t next_token_idx = not kOverlapReduce ? token_idx + 1 :
            token_idx + topology.vectors() < num_tokens ? token_idx + topology.vectors() : token_idx % topology.vectors();
        const uint32_t word_idx = token_idx / (topology.vectors() * state.kNumBitsPerWord);
        const uint32_t token_bit = 1u << (token_idx / topology.vectors() % state.kNumBitsPerWord);

        // Skip completed tokens
        if (kOverlapReduce and (state.reduced_token_mask[word_idx] & token_bit)) {
            state.token_idx = next_token_idx;
            continue;
        }

        // Check the actual producer block for each route before the first chunk
        if (kOverlapReduce and state.channel_idx == 0) {
            #pragma unroll
            for (uint32_t word = 0; word < ceil_div(config.num_topk, state.kNumBitsPerWord); ++word)
                state.valid_topk_mask[word] = 0;
            for (uint32_t topk_slot_idx = 0; topk_slot_idx < config.num_topk; ++ topk_slot_idx) {
                const uint32_t token_topk_idx = token_idx * config.num_topk + topk_slot_idx;
                const uint64_t expert_idx = asc_load_dev(topk_idx + token_topk_idx);
                if (expert_idx >= config.num_experts)
                    continue;
                const uint32_t peer_idx = expert_idx / kNumExpertsPerRank;
                const uint32_t pool_block_idx = asc_load_dev(workspace.route_m_blocks.get_ptr(buffer, token_topk_idx));
                // Completed prefixes only advance; refresh a peer when its cached value is insufficient
                auto& completed_blocks = state.completed_blocks[peer_idx];
                if (pool_block_idx >= completed_blocks) {
                    completed_blocks = asc_load_dev(workspace.metadata.combine_ready_prefix.get_ptr(buffer, peer_idx));
                    if (pool_block_idx >= completed_blocks) {
                        state.blocking_rank_idx = peer_idx;
                        state.token_idx = next_token_idx;
                        return false;
                    }
                }
                state.valid_topk_mask[topk_slot_idx / state.kNumBitsPerWord] |=
                    1u << (topk_slot_idx % state.kNumBitsPerWord);
            }
        }

        // Initialize UB credits on first use
        if (stage_idx == state.kUninitializedStage) {
            notify<PIPE_MTE3, PIPE_MTE2, kNumReduceStages>(kReduceEmpty);
            stage_idx = 0;
        }

        for (uint32_t channel_base = state.channel_idx; channel_base < config.hidden; channel_base += num_chunk_channels) {
            const uint32_t num_channels = min(config.hidden - channel_base, num_chunk_channels);
            wait<PIPE_MTE3, PIPE_MTE2>(kReduceEmpty, stage_idx);

            // Preserve the public shared-first, then TopK0..TopK-1 FP32 order.
            uint32_t num_inputs = 0;
            for (uint32_t input_order = 0; input_order < config.num_topk + kHasSharedExperts; ++ input_order) {
                const uint32_t topk_slot_idx = kHasSharedExperts ?
                    (input_order == 0 ? config.num_topk : input_order - 1) : input_order;
                if (topk_slot_idx < config.num_topk and (kOverlapReduce ?
                        (state.valid_topk_mask[topk_slot_idx / state.kNumBitsPerWord] &
                         (1u << (topk_slot_idx % state.kNumBitsPerWord))) == 0 :
                        asc_load_dev(topk_idx + static_cast<uint64_t>(token_idx) * config.num_topk + topk_slot_idx) >= config.num_experts))
                    continue;
                const auto src = combine_input + workspace.get_combine_m_idx(topk_slot_idx, token_idx) * config.hidden + channel_base;
                copy_gm_to_ub(ub_reduce_inputs[stage_idx].ptr() + num_inputs++ * num_channels,
                              src, 1, num_channels, num_channels * sizeof(bfloat16_t), asc_load_l2_cache_mode::NOTALLOC_KEEP);
            }
            notify<PIPE_MTE2, PIPE_V>(kReduceFull, stage_idx);

            // Accumulate in FP32, then round once to BF16
            wait<PIPE_MTE2, PIPE_V>(kReduceFull, stage_idx);
            vf_combine_reduce(ub_reduce_inputs[stage_idx].ptr(), num_inputs, num_channels);
            notify<PIPE_V, PIPE_MTE3>(kReduceOutputFull, stage_idx);

            // Store BF16 output
            wait<PIPE_V, PIPE_MTE3>(kReduceOutputFull, stage_idx);
            copy_ub_to_gm(output + static_cast<uint64_t>(token_idx) * config.hidden + channel_base,
                          ub_reduce_inputs[stage_idx].ptr(), 1, num_channels,
                          num_channels * sizeof(bfloat16_t), kWriteThroughShare);
            notify<PIPE_MTE3, PIPE_MTE2>(kReduceEmpty, stage_idx);
            stage_idx = (stage_idx + 1) % kNumReduceStages;

            // Save the resume position
            state.channel_idx = channel_base + num_channels;
            if (state.channel_idx == config.hidden) {
                state.channel_idx = 0;
                state.token_idx = next_token_idx;
                if constexpr (kOverlapReduce)
                    state.reduced_token_mask[word_idx] |= token_bit;
                -- state.num_remaining_tokens;
            }

            if (-- num_steps == 0)
                return state.num_remaining_tokens == 0;
        }
    }
    return true;
}

template <const auto& workspace, bool kOverlapReduce, bool kFullTopology>
__aicore__ __forceinline__ void combine_reduce(__gm__ uint8_t* buffer, __gm__ bfloat16_t* output,
                                              uint32_t num_tokens, uint32_t rank_idx, uint32_t aiv_idx,
                                              CombineReduceState<workspace, kFullTopology>& reduce, const layout::SymBuffer<>& sym_buffer,
                                              CoreTopology<kFullTopology> topology) {
    constexpr auto& config = workspace.config;

    // Drain communication before reusing UB
    if constexpr (kOverlapReduce) {
        notify<PIPE_MTE3, PIPE_S>(kCommDone);
        wait<PIPE_MTE3, PIPE_S>(kCommDone);
    } else {
        comm::rank_barrier<workspace, 1>(buffer, sym_buffer, aiv_idx);
    }

    // Drain reduce stages before changing chunk width
    if (reduce.stage_idx != reduce.kUninitializedStage)
        wait<PIPE_MTE3, PIPE_MTE2, kNumReduceStages>(kReduceEmpty);
    reduce.stage_idx = reduce.kUninitializedStage;

    // Reduce remaining tokens
    // After draining, both AIVs reuse UB from offset 0 for two reduce stages
    constexpr auto ub_reduce_inputs = make_reduce_buffer<workspace>();
    comm::wait_until([&]() __aicore__ {
        return advance_combine_reduce<workspace, kOverlapReduce>(buffer, output, num_tokens, reduce, ub_reduce_inputs, topology);
    }, [&]() __aicore__ {
        if (config.num_shared_experts > 0 and not reduce.shared_output_ready) {
            comm::print_local_timeout(rank_idx);
            return;
        }
        // This is the last observed blocker; token_idx already points to the next probe
        const uint32_t peer_idx = reduce.blocking_rank_idx;
        comm::print_peer_timeout(rank_idx, aiv_idx, peer_idx,
            asc_load_dev(workspace.metadata.combine_ready_prefix.get_ptr(buffer, peer_idx)), false);
    });

    // Empty ranks also wait for peer writes before the next invocation reuses scratch
    if constexpr (kOverlapReduce) {
        for (uint32_t peer_idx = aiv_idx; peer_idx < config.num_ranks; peer_idx += topology.vectors()) {
            const auto prefix_ptr = workspace.metadata.combine_ready_prefix.get_ptr(buffer, peer_idx);
            comm::wait_until([&]() __aicore__ { return asc_load_dev(prefix_ptr) == ~uint64_t{0}; }, [&]() __aicore__ {
                comm::print_peer_timeout(rank_idx, aiv_idx, peer_idx, asc_load_dev(prefix_ptr), true);
            });
        }
    }

    // Consume final reduce credits only on AIVs that initialized these stages.
    if (reduce.stage_idx != reduce.kUninitializedStage) {
        wait<PIPE_MTE3, PIPE_MTE2, kNumReduceStages>(kReduceEmpty);
        reduce.stage_idx = reduce.kUninitializedStage;
    }
}

#if MOE_STREAM_W2_SCALES
// Shift sixteen N16 SF panels by +6 and flag any original byte >= 249.
// Each panel has a 1024-byte physical pitch; valid bytes are a multiple of 256.
__simd_vf__ inline void stream_shift_sf(__ubuf__ uint8_t* data, __ubuf__ uint32_t* status, uint32_t bytes_per_panel) {
    vector_uint32_t unsafe, lowmask, highmask, part;
    asc_duplicate_scalar(unsafe, 0u);
    asc_duplicate_scalar(lowmask, 0x7f7f7f7fu);
    asc_duplicate_scalar(highmask, 0x80808080u);
    const auto m32 = asc_create_mask_b32(PAT_ALL);
    const auto m8 = asc_create_mask_b8(PAT_ALL);
    for (uint32_t panel = 0; panel < 16; ++panel) {
        auto* input = data + panel * 1024;
        auto* output = input;
        for (uint32_t i = 0; i < bytes_per_panel; i += 256) {
            vector_uint8_t value;
            asc_loadalign_postupdate(value, input, 256);
            auto words = reinterpret_cast<vector_uint32_t&>(value);
            asc_and(part, words, lowmask, m32);
            asc_add_scalar(part, part, 0x07070707u, m32);
            asc_and(part, part, words, m32);
            asc_and(part, part, highmask, m32);
            asc_or(unsafe, unsafe, part, m32);
            asc_add_scalar(value, value, uint8_t{6}, m8);
            asc_storealign_postupdate(output, value, 256, m8);
        }
    }
    asc_reduce_max(unsafe, unsafe, m32);
    asc_storealign(status, unsafe, asc_create_mask_b32(PAT_VL1));
}

// Restore original SF bytes modulo 256 for the exact-conversion fallback.
__simd_vf__ inline void stream_restore_sf(__ubuf__ uint8_t* data, uint32_t bytes_per_panel) {
    const auto mask = asc_create_mask_b8(PAT_ALL);
    for (uint32_t panel = 0; panel < 16; ++panel) {
        auto* input = data + panel * 1024;
        auto* output = input;
        for (uint32_t i = 0; i < bytes_per_panel; i += 256) {
            vector_uint8_t value;
            asc_loadalign_postupdate(value, input, 256);
            asc_add_scalar(value, value, uint8_t{250}, mask);
            asc_storealign_postupdate(output, value, 256, mask);
        }
    }
}

template <uint32_t Columns>
__aicore__ inline void load_stream_sf_columns(__ubuf__ uint8_t* destination, __gm__ const uint8_t* source,
    uint32_t experts, uint32_t num_k, uint32_t expert, uint32_t column, uint32_t k_begin, uint32_t valid_pairs) {
    const uint64_t scale_rows = static_cast<uint64_t>(experts) * (num_k / 32);
    const uint64_t offset = column * scale_rows + (expert * (num_k / 32) + k_begin / 32) * 16;
    asc_copy_gm2ub_align(destination, const_cast<__gm__ uint8_t*>(source) + offset,
        Columns / 16, valid_pairs * 32, 0, 0, false, asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
        scale_rows * 16, 1024);
}
#endif

// AIV0: FP4 weights -> FP8 NZ -> L1B
template <const auto& workspace, bool kOverlapReduce, bool kFullTopology>
__aicore__ inline void aiv0_weight_pipeline(__gm__ uint8_t* buffer, __gm__ bfloat16_t* output,
                                            uint32_t num_tokens, CombineReduceState<workspace, kFullTopology>& reduce, uint32_t num_total_m_blocks,
                                            const layout::Weights& weights, uint32_t core_idx, CoreTopology<kFullTopology> topology) {
    constexpr auto& config = workspace.config;
    constexpr uint32_t kNumLinear1NBlocks = config.intermediate_hidden / (BLOCK_N / 2);
    constexpr uint32_t kNumLinear2NBlocks = config.hidden / BLOCK_N;
    constexpr uint32_t kNumRingBlocks = workspace.get_num_ring_blocks();

    // AIV0 UB after metadata (KiB)
    // Region   Offset  Stages x size
    // FP4           0      3 x 32
    // FP8 NZ       96      2 x 64.5
    // Reduce      225      2 stages within the remaining 31 KiB (overlap reduce)
    // W2 SF       225      16 KiB (streamed W2 SF, no overlap reduce)
    // SF status   241      32 bytes
    constexpr uint32_t kNumFP4Stages = 3, kNumNZStages = 2;
    constexpr ub_ptr<uint8_t> ub_fp4(1, L0BSizeBytes / 2);
    constexpr ub_ptr<uint8_t> ub_nz(1, L0BSizeBytes + 512, ub_fp4.offset(kNumFP4Stages));
    constexpr auto ub_reduce_inputs = make_reduce_buffer<workspace>(ub_nz.offset(kNumNZStages));
    constexpr uint32_t stream_sf_ub = ub_nz.offset(kNumNZStages);
    constexpr uint32_t stream_sf_status_ub = stream_sf_ub + 16384;
    constexpr l1_ptr<int16_t> stream_sfb(BLOCK_N, 32, 2 * L0ASizeBytes);
    static_assert(MOE_TOKENS > 80 || stream_sf_status_ub + 32 <= UBSizeBytes);

    static_assert(ub_nz.offset(kNumNZStages) <= UBSizeBytes and ub_reduce_inputs.shape_n > 0, "AIV0 UB overflow");

    // Initialize UB reuse credits; AIC releases the initial L1B credits
    notify<PIPE_V, PIPE_MTE2, kNumFP4Stages>(kFP4Empty);
    notify<PIPE_MTE3, PIPE_V, kNumNZStages>(kNZEmpty);
#if MOE_STREAM_W2_SCALES
    notify<PIPE_MTE3, PIPE_MTE2>(kStreamSFUBEmpty);
#endif

    // Size the reduce window from local output tokens
    const uint32_t num_reduce_window_m_blocks = ceil_div(num_tokens, BLOCK_M);
    sched::LinearTaskState schedule;
    sched::ExpertCursor<workspace> expert_cursors[2]{};
    uint32_t fp4_stage_idx = 0, nz_stage_idx = 0;
    uint32_t l1b_stage_idx = 0;
    bool had_routed_task = false;
    uint32_t stream_sf_stage = 0;
    bool stream_shifted = false;

    while (true) {
        sched::BlockPhase block_phase;
        // AIC handles shared weights; zero shared tokens skips those tasks
        const auto task_idx = sched::take_linear_task<workspace>(0, num_total_m_blocks, schedule, core_idx, block_phase, topology);
        if (block_phase == sched::BlockPhase::kNone)
            break;
        had_routed_task = true;

        // Select task weights
        const bool is_linear1 = block_phase == sched::BlockPhase::kLinear1;
        const uint32_t num_n_blocks = is_linear1 ? kNumLinear1NBlocks : kNumLinear2NBlocks;
        const uint32_t num_k = is_linear1 ? config.hidden : config.intermediate_hidden;
        const uint32_t pool_block_idx = task_idx / num_n_blocks;
        const uint32_t n_block_idx = task_idx % num_n_blocks;
        const uint32_t layer_idx = is_linear1 ? 0 : 1;
        stream_shifted = false;
        const uint32_t valid_m = expert_cursors[layer_idx].advance_to(buffer, pool_block_idx);
        const auto weight_l2_hint = expert_cursors[layer_idx].num_expert_tokens > BLOCK_M ?
            asc_load_l2_cache_mode::NORMAL_LAST_VICTIM : asc_load_l2_cache_mode::NOTALLOC_KEEP;
        const uint32_t weight_expert = expert_cursors[layer_idx].local_expert_idx;
        const uint32_t weight_output_n = is_linear1 ? config.intermediate_hidden : config.hidden;
        const auto first_weight = is_linear1 ? weights.l1.data : weights.l2.data;
        const auto up_weight = weights.l1_up.data;

#if MOE_TOKENS <= 80
        const auto load_weights = [&]<uint32_t MAD_M, bool Linear1>() __aicore__ {
            const bool is_linear1 = Linear1;
#else
        const auto load_weights = [&]<uint32_t MAD_M>() __aicore__ {
#endif
            constexpr uint32_t BLOCK_K = MMAConfig<MAD_M>::BLOCK_K;
            constexpr uint32_t MAD_N = MMAConfig<MAD_M>::MAD_N;
            constexpr uint32_t kNumColGroups = MAD_N / get_frac_k<uint8_t>();
            constexpr l1_ptr<float8_e4m3_t, Major::MN> l1b(MAD_N, BLOCK_K, kL1BBase);

            for (uint32_t k_idx = 0; k_idx < num_k; k_idx += BLOCK_K) {
                // valid_k remains a multiple of MAD_K, including the final K block
                const uint32_t valid_k = min(BLOCK_K, num_k - k_idx);
                bool first_panel_prefetched = false;
#if MOE_STREAM_W2_SCALES
                if (not is_linear1 and k_idx % 2048 == 0) {
                    const uint32_t valid_pairs = min(32u, (num_k - k_idx) / 64);
                    const uint32_t bytes_per_panel = valid_pairs * 32;
                    wait<PIPE_MTE3, PIPE_MTE2>(kStreamSFUBEmpty);
                    auto* scale_ub = reinterpret_cast<__ubuf__ uint8_t*>(stream_sf_ub);
                    auto* first_sf = reinterpret_cast<__gm__ const uint8_t*>(is_linear1 ? weights.l1.sf : weights.l2.sf);
                    if (is_linear1) {
                        load_stream_sf_columns<128>(scale_ub, first_sf, config.num_experts / config.num_ranks,
                            num_k, weight_expert, n_block_idx * 128, k_idx, valid_pairs);
                        load_stream_sf_columns<128>(scale_ub + 8192, reinterpret_cast<__gm__ const uint8_t*>(weights.l1_up.sf),
                            config.num_experts / config.num_ranks, num_k, weight_expert, n_block_idx * 128, k_idx, valid_pairs);
                    } else {
                        load_stream_sf_columns<256>(scale_ub, first_sf, config.num_experts / config.num_ranks,
                            num_k, weight_expert, n_block_idx * 256, k_idx, valid_pairs);
                    }
                    notify<PIPE_MTE2, PIPE_V>(kStreamSFLoaded);
                    // Prefetch raw weights into a protected slot before checking the SF chunk.
                    wait<PIPE_V, PIPE_MTE2>(kFP4Empty, fp4_stage_idx);
#ifdef __DAV_VEC__
                    original_weights::load_routed_panel<BLOCK_K, MAD_N>(
                        ub_fp4[fp4_stage_idx].addr, first_weight, up_weight, is_linear1,
                        weight_expert, num_k, weight_output_n, n_block_idx, 0, k_idx, valid_k, weight_l2_hint);
#endif
                    notify<PIPE_MTE2, PIPE_V>(kFP4Full, fp4_stage_idx);
                    first_panel_prefetched = true;
                    wait<PIPE_MTE2, PIPE_V>(kStreamSFLoaded);
                    // Partial 256-byte SF slices retain the original scale and exact conversion.
                    stream_shifted = false;
                    if (bytes_per_panel % 256 == 0) {
                        auto* status = reinterpret_cast<__ubuf__ uint32_t*>(stream_sf_status_ub);
                        stream_shift_sf(scale_ub, status, bytes_per_panel);
                        notify<PIPE_V, PIPE_S>(kStreamSFChecked);
                        wait<PIPE_V, PIPE_S>(kStreamSFChecked);
                        stream_shifted = *status == 0;
                        if (not stream_shifted)
                            stream_restore_sf(scale_ub, bytes_per_panel);
                    }
                    notify<PIPE_V, PIPE_MTE3>(kStreamSFConverted);
                    wait<PIPE_V, PIPE_MTE3>(kStreamSFConverted);
                    wait_intra_block<PIPE_MTE3, 0>(kStreamSFEmpty, stream_sf_stage);
                    asc_copy_ub2l1(stream_sfb[stream_sf_stage].template ptr<uint8_t>(), scale_ub,
                        16, valid_pairs, 32 - valid_pairs, 32 - valid_pairs);
                    notify_intra_block<PIPE_MTE3, 0>(kStreamSFFull, stream_sf_stage);
                    notify<PIPE_MTE3, PIPE_MTE2>(kStreamSFUBEmpty);
                    stream_sf_stage = (stream_sf_stage + 1) % kNumSFStages;
                }
#endif
                for (uint32_t n_mad_idx = 0; n_mad_idx < BLOCK_N; n_mad_idx += MAD_N) {
                    // The SF-boundary first panel was already loaded into this protected slot.
                    if (not first_panel_prefetched or n_mad_idx != 0) {
                        wait<PIPE_V, PIPE_MTE2>(kFP4Empty, fp4_stage_idx);
#ifdef __DAV_VEC__
                        original_weights::load_routed_panel<BLOCK_K, MAD_N>(
                            ub_fp4[fp4_stage_idx].addr, first_weight, up_weight, is_linear1,
                            weight_expert, num_k, weight_output_n, n_block_idx, n_mad_idx, k_idx, valid_k, weight_l2_hint);
#endif
                        notify<PIPE_MTE2, PIPE_V>(kFP4Full, fp4_stage_idx);
                    }

                    // Dequantize to FP8 NZ
                    wait<PIPE_MTE2, PIPE_V>(kFP4Full, fp4_stage_idx);
                    wait<PIPE_MTE3, PIPE_V>(kNZEmpty, nz_stage_idx);
#ifdef __DAV_VEC__
                    original_weights::decode_routed_panel<BLOCK_K, MAD_N>(
                        ub_fp4[fp4_stage_idx].addr, ub_nz[nz_stage_idx].addr, stream_shifted);
#endif
                    notify<PIPE_V, PIPE_MTE2>(kFP4Empty, fp4_stage_idx);
                    notify<PIPE_V, PIPE_MTE3>(kNZFull, nz_stage_idx);

                    // Publish L1B to AIC
                    wait<PIPE_V, PIPE_MTE3>(kNZFull, nz_stage_idx);
                    wait_intra_block<PIPE_MTE3, 0>(kL1BEmpty, l1b_stage_idx);
                    asc_copy_ub2l1(l1b[l1b_stage_idx].template ptr<uint8_t>(), ub_nz[nz_stage_idx].ptr(), kNumColGroups, BLOCK_K, 1, 0);
                    notify_intra_block<PIPE_MTE3, 0>(kL1BFull, l1b_stage_idx);
                    notify<PIPE_MTE3, PIPE_V>(kNZEmpty, nz_stage_idx);

                    fp4_stage_idx = (fp4_stage_idx + 1) % kNumFP4Stages;
                    nz_stage_idx = (nz_stage_idx + 1) % kNumNZStages;
                    l1b_stage_idx = (l1b_stage_idx + 1) % kNumL1BStages;
                }

                // Reduce one chunk in unused UB
                if (kOverlapReduce and num_total_m_blocks > kNumRingBlocks and pool_block_idx + num_reduce_window_m_blocks >= num_total_m_blocks)
                    advance_combine_reduce<workspace, kOverlapReduce>(buffer, output, num_tokens, reduce, ub_reduce_inputs, topology, 1);
            }
        };
#if MOE_TOKENS <= 80
        if (is_linear1) {
            if (valid_m <= 128)
                load_weights.template operator()<128, true>();
            else
                load_weights.template operator()<256, true>();
        } else {
            if (valid_m <= 128)
                load_weights.template operator()<128, false>();
            else
                load_weights.template operator()<256, false>();
        }
#else
        if (valid_m <= 128)
            load_weights.template operator()<128>();
        else
            load_weights.template operator()<256>();
#endif
    }

    // Return with neutral events after all routed tasks on this core.
    wait<PIPE_V, PIPE_MTE2, kNumFP4Stages>(kFP4Empty);
    wait<PIPE_MTE3, PIPE_V, kNumNZStages>(kNZEmpty);
    if (had_routed_task) {
        wait_intra_block<PIPE_MTE3, 0, kNumL1BStages>(kL1BEmpty);
    }
#if MOE_STREAM_W2_SCALES
    wait_intra_block<PIPE_MTE3, 0, kNumSFStages>(kStreamSFEmpty);
    wait<PIPE_MTE3, PIPE_MTE2>(kStreamSFUBEmpty);
#endif
}

// AIC: shared and routed GEMMs
template <const auto& workspace, bool kFullTopology>
__aicore__ inline void aic_gemm_pipeline(__gm__ uint8_t* buffer, uint32_t num_tokens, uint32_t rank_idx,
                                         const layout::Weights& weights, uint32_t core_idx, CoreTopology<kFullTopology> topology) {
    constexpr auto& config = workspace.config;
    constexpr uint32_t kNumRingBlocks = workspace.get_num_ring_blocks();
    constexpr bool kHasSharedExperts = config.num_shared_experts > 0;
    constexpr uint32_t kNumSFPairsPerChunk = 32;
    constexpr l1_ptr<int16_t> l1_sfa(BLOCK_M, kNumSFPairsPerChunk,
                                   kL1BBase + kNumL1BStages * L0BSizeBytes);
    constexpr l1_ptr<int16_t> l1_sfb(BLOCK_N, kNumSFPairsPerChunk, l1_sfa.offset(kNumSFStages));
    constexpr l1_ptr<int16_t> l1_w2_sfb(BLOCK_N, kNumSFPairsPerChunk, 2 * L0ASizeBytes);
    static_assert(MOE_TOKENS > 80 || l1_w2_sfb.offset(kNumSFStages) <= kL1BBase);
    constexpr ub_ptr<bfloat16_t> ub_accum(STORE_BLOCK_M, BLOCK_N);
    static_assert(l1_sfb.offset(kNumSFStages) <= L1SizeBytes, "L1 overflow");

    const auto num_issued_gemms_ptr = workspace.progress.num_issued_gemms.get_ptr(buffer, core_idx);
    const uint32_t num_shared_m_blocks = kHasSharedExperts ? ceil_div(num_tokens, BLOCK_M) : 0;
    const bool reuse_shared_weights = num_shared_m_blocks > 1;
    const auto routed_output = workspace.routed.output.get_ptr(buffer);
    const auto combine_input = workspace.combine_input.get_ptr(buffer);

    // Initialize reuse credits
    asc_set_mmad_direction_n();
    notify<PIPE_M, PIPE_MTE1, kNumL0Stages>(kL0AEmpty);
    notify<PIPE_M, PIPE_MTE1, kNumL0Stages>(kL0BEmpty);
    notify<PIPE_MTE1, PIPE_MTE2, kNumL1AStages>(kL1AEmpty);
    notify<PIPE_MTE1, PIPE_MTE2, kNumSFStages>(kSFEmpty);
#if MOE_STREAM_W2_SCALES
    notify_intra_block<PIPE_MTE1, 0, kNumSFStages>(kStreamSFEmpty);
#endif
    if constexpr (kHasSharedExperts)
        notify<PIPE_MTE1, PIPE_MTE2, 3>(kSharedL1BEmpty);

    sched::LinearTaskState schedule;
    sched::ExpertCursor<workspace> expert_cursors[2]{};
    uint32_t l1a_stage_idx = 0;
    uint32_t l1b_stage_idx = 0;
    uint32_t sf_stage_idx = 0;
    uint32_t w2_sf_stage_idx = 0;
    uint8_t num_issued_gemms = 0;
    bool routed_l1b_initialized = false;
    uint32_t num_total_m_blocks = 0;  // Routed pool only; shared blocks are separate
    bool metadata_ready = false;
    const auto num_m_blocks_certificate = workspace.metadata.num_m_blocks_certificate.get_ptr(buffer);

    while (true) {
        sched::BlockPhase block_phase;
        const auto task_idx = sched::take_linear_task<workspace>(
            num_tokens, num_total_m_blocks, schedule, core_idx, block_phase, topology, metadata_ready);
        if (block_phase == sched::BlockPhase::kNone) {
            if (metadata_ready)
                break;

            // Consume metadata before routed L1B reuses the same intra-block event
            wait_intra_block<PIPE_S, 0>(kMetadataFull);
            num_total_m_blocks = asc_load_dev(num_m_blocks_certificate) & layout::kCountValueMask;
            metadata_ready = true;
            continue;
        }

        // Specialize short-capacity projection phases at the task boundary.
#if MOE_TOKENS <= 80
        const auto run_task = [&]<bool is_shared, bool is_linear1>() __aicore__ {
        constexpr auto block_phase = is_shared ?
            (is_linear1 ? sched::BlockPhase::kSharedLinear1 : sched::BlockPhase::kSharedLinear2) :
            (is_linear1 ? sched::BlockPhase::kLinear1 : sched::BlockPhase::kLinear2);
#else
        // Decode the GEMM task
        const bool is_shared = block_phase == sched::BlockPhase::kSharedLinear1 or block_phase == sched::BlockPhase::kSharedLinear2;
        const bool is_linear1 = block_phase == sched::BlockPhase::kLinear1 or block_phase == sched::BlockPhase::kSharedLinear1;
#endif

        // Shared experts widen Linear1 N and Linear2 K
        const uint32_t intermediate_hidden = is_shared ? config.shared_intermediate_hidden : config.intermediate_hidden;
        const uint32_t num_linear1_n_blocks = intermediate_hidden / (BLOCK_N / 2);
        const uint32_t num_n_blocks = is_linear1 ? num_linear1_n_blocks : config.hidden / BLOCK_N;
        const uint32_t num_k = is_linear1 ? config.hidden : intermediate_hidden;
        const uint32_t num_sf_pairs = num_k / MX_SF_DIVISOR;
        const uint32_t pool_block_idx = task_idx / num_n_blocks;
        const uint32_t n_block_idx = task_idx % num_n_blocks;
        const uint32_t layer_idx = is_linear1 ? 0 : 1;
        const uint32_t valid_m = is_shared ? min(num_tokens - pool_block_idx * BLOCK_M, BLOCK_M) :
            expert_cursors[layer_idx].advance_to(buffer, pool_block_idx);
        const auto weight_sf_l2_hint = (is_shared ? reuse_shared_weights : expert_cursors[layer_idx].num_expert_tokens > BLOCK_M) ?
            asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM : asc_load_l2_cache_mode::NOTALLOC_KEEP;
        const uint32_t aligned_valid_m = aligned(valid_m, FRAC_MN);
        const uint32_t weight_expert = is_shared ? 0 : expert_cursors[layer_idx].local_expert_idx;
        const uint32_t weight_experts = is_shared ? 1 : config.num_experts / config.num_ranks;
        const uint32_t weight_output_n = is_linear1 ? intermediate_hidden : config.hidden;
        const auto task_weights = is_linear1 ? (is_shared ? weights.shared_l1 : weights.l1) : (is_shared ? weights.shared_l2 : weights.l2);
        const auto up_weights = is_shared ? weights.shared_l1_up : weights.l1_up;

        // Hand L1B over to AIV0 after shared tasks
        if (not is_shared and not routed_l1b_initialized) {
            l1b_stage_idx = 0;
            notify_intra_block<PIPE_MTE1, 0, kNumL1BStages>(kL1BEmpty);
            routed_l1b_initialized = true;
        }

        // Select input data and scales
        const uint32_t input_block_idx = is_shared ? pool_block_idx : layout::get_ring_block_idx<workspace>(pool_block_idx);
        const uint64_t input_m_idx = static_cast<uint64_t>(input_block_idx) * BLOCK_M;
        const uint64_t input_offset = input_m_idx * num_k;
        const uint64_t input_sf_offset = input_m_idx * num_sf_pairs;
        const auto input = is_linear1 ?
            (is_shared ? workspace.input.acts.data.get_ptr(buffer, input_offset) : workspace.routed.l1_acts.data.get_ptr(buffer, input_offset)) :
            (is_shared ? workspace.shared_l2_acts.data.get_ptr(buffer, input_offset) : workspace.routed.l2_acts.data.get_ptr(buffer, input_offset));
        const auto input_sf = is_linear1 ?
            (is_shared ? workspace.input.acts.sf.get_ptr(buffer, input_sf_offset) : workspace.routed.l1_acts.sf.get_ptr(buffer, input_sf_offset)) :
            (is_shared ? workspace.shared_l2_acts.sf.get_ptr(buffer, input_sf_offset) : workspace.routed.l2_acts.sf.get_ptr(buffer, input_sf_offset));

        const auto compute = [&]<uint32_t MAD_M>() __aicore__ {
            constexpr uint32_t BLOCK_K = MMAConfig<MAD_M>::BLOCK_K;
            constexpr uint32_t MAD_K = MMAConfig<MAD_M>::MAD_K;
            constexpr uint32_t MAD_N = MMAConfig<MAD_M>::MAD_N;
            constexpr auto l0_full = BLOCK_N == MAD_N ? kL0AFull : kL0BFull;
            constexpr uint32_t kNumSharedL1BStages = MMAConfig<MAD_M>::get_num_shared_l1b_stages();
            constexpr l1_ptr<float8_e4m3_t, Major::K> l1a(L0ASizeBytes / BLOCK_K, BLOCK_K, 0);
            constexpr l1_ptr<float8_e4m3_t, Major::MN> l1b(MAD_N, BLOCK_K, kL1BBase);
            constexpr l1_ptr<float8_e4m3_t, Major::MN> shared_l1b(BLOCK_N, BLOCK_K, l1b.addr);
            constexpr l0a_ptr<float8_e4m3_t> l0a(MAD_M, MAD_K);
            constexpr l0b_ptr<float8_e4m3_t> l0b(MAD_N, MAD_K);
            constexpr l0c_ptr<float> l0c(MAD_M, MAD_N);

            for (uint32_t k_idx = 0; k_idx < num_k; k_idx += BLOCK_K) {
                // valid_k remains a multiple of MAD_K, including the final K block
                const uint32_t valid_k = min(BLOCK_K, num_k - k_idx);
                const uint32_t sf_pair_idx = k_idx / MX_SF_DIVISOR;
                const uint32_t num_chunk_pairs = min(kNumSFPairsPerChunk, num_sf_pairs - sf_pair_idx);

                // Load shared FP8 weights
                if (is_shared) {
                    wait<PIPE_MTE1, PIPE_MTE2>(kSharedL1BEmpty, l1b_stage_idx);
#ifdef __DAV_CUBE__
                    original_weights::load_shared_panel<BLOCK_K>(shared_l1b[l1b_stage_idx].addr,
                        task_weights.data, up_weights.data, is_linear1, num_k, weight_output_n,
                        n_block_idx, k_idx, valid_k);
#endif
                    notify<PIPE_MTE2, PIPE_MTE1>(kSharedL1BFull, l1b_stage_idx);
                }

                // Load weight scales once per SF chunk
                if (sf_pair_idx % kNumSFPairsPerChunk == 0) {
                    wait<PIPE_MTE1, PIPE_MTE2>(kSFEmpty, sf_stage_idx);
#ifdef __DAV_CUBE__
#if MOE_STREAM_W2_SCALES
                    if (is_shared or is_linear1)
#endif
                    original_weights::load_weight_scales(l1_sfb[sf_stage_idx].addr,
                        reinterpret_cast<__gm__ const uint8_t*>(task_weights.sf),
                        reinterpret_cast<__gm__ const uint8_t*>(up_weights.sf), is_linear1,
                        weight_experts, num_k, weight_output_n, weight_expert, n_block_idx,
                        k_idx, num_chunk_pairs, weight_sf_l2_hint);
#endif
                }

                // Prefetch independent weights before waiting for activations
                if (k_idx == 0 and (not is_shared or not is_linear1))
                    comm::wait_for_block<workspace>(buffer, rank_idx,
                        is_linear1 ? layout::kDispatched : layout::kLinear1Ready,
                        (is_shared ? 0 : num_shared_m_blocks) + pool_block_idx,
                        is_linear1 ? topology.cores() : num_linear1_n_blocks);

                // Load activations and scales
                wait<PIPE_MTE1, PIPE_MTE2>(kL1AEmpty, l1a_stage_idx);
                copy_gm_to_l1(l1a[l1a_stage_idx],
                              gm_ptr<float8_e4m3_t, Major::K>(num_k, reinterpret_cast<uintptr_t>(input + k_idx)),
                              valid_m, valid_k, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                notify<PIPE_MTE2, PIPE_MTE1>(kL1AFull, l1a_stage_idx);

                if (sf_pair_idx % kNumSFPairsPerChunk == 0) {
                    if (is_linear1) {
                        copy_gm_to_l1_mx(l1_sfa[sf_stage_idx],
                                         gm_ptr<int16_t, Major::K>(num_sf_pairs, reinterpret_cast<uintptr_t>(input_sf + sf_pair_idx)),
                                         valid_m, num_chunk_pairs, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                    } else {
                        copy_gm_to_l1_mx(l1_sfa[sf_stage_idx],
                                         gm_ptr<int16_t, Major::MN>(BLOCK_M, reinterpret_cast<uintptr_t>(input_sf + sf_pair_idx * BLOCK_M)),
                                         valid_m, num_chunk_pairs, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                    }

                    // One completion covers both scale loads on the same MTE2 queue
                    notify<PIPE_MTE2, PIPE_MTE1>(kSFFull, sf_stage_idx);
                }

                // Wait for L1 inputs
                wait<PIPE_MTE2, PIPE_MTE1>(kL1AFull, l1a_stage_idx);
                if (is_shared) {
                    wait<PIPE_MTE2, PIPE_MTE1>(kSharedL1BFull, l1b_stage_idx);
                } else {
                    for (uint32_t n_mad_idx = 0; n_mad_idx < BLOCK_N; n_mad_idx += MAD_N)
                        wait_intra_block<PIPE_MTE1, 0>(kL1BFull, (l1b_stage_idx + n_mad_idx / MAD_N) % kNumL1BStages);
                }

                // Reuse L0A across the BLOCK_N / MAD_N multiplies
                for (uint32_t k_mad_idx = 0; k_mad_idx < valid_k; k_mad_idx += MAD_K) {
                    const uint32_t l0a_stage_idx = (k_mad_idx / MAD_K) % kNumL0Stages;
                    const uint32_t sf_pair_idx_in_chunk = (sf_pair_idx + k_mad_idx / MX_SF_DIVISOR) % kNumSFPairsPerChunk;
                    const auto l0a_mad = l0a[l0a_stage_idx].as_mad_aligned(aligned_valid_m, MAD_K);
                    wait<PIPE_M, PIPE_MTE1>(kL0AEmpty, l0a_stage_idx);
                    copy_l1_to_l0a(l0a_mad, l1a[l1a_stage_idx], 0, k_mad_idx);
                    if (sf_pair_idx % kNumSFPairsPerChunk == 0 and k_mad_idx == 0)
                        wait<PIPE_MTE2, PIPE_MTE1>(kSFFull, sf_stage_idx);
                    copy_l1_to_l0a_mx(l0a_mad, l1_sfa[sf_stage_idx], 0, sf_pair_idx_in_chunk, MAD_K);
                    const bool is_last_k = k_idx + k_mad_idx + MAD_K == num_k;

                    // Keep B reuse credits independent across MAD shapes
                    for (uint32_t n_mad_idx = 0; n_mad_idx < BLOCK_N; n_mad_idx += MAD_N) {
                        const uint32_t l0c_idx = n_mad_idx / MAD_N;
                        const uint32_t l0b_stage_idx = (l0a_stage_idx * (BLOCK_N / MAD_N) + l0c_idx) % kNumL0Stages;
                        wait<PIPE_M, PIPE_MTE1>(kL0BEmpty, l0b_stage_idx);
                        const auto l1b_src = is_shared ? shared_l1b[l1b_stage_idx] : l1b[(l1b_stage_idx + n_mad_idx / MAD_N) % kNumL1BStages];
                        copy_l1_to_l0b(l0b[l0b_stage_idx], l1b_src, is_shared ? n_mad_idx : 0, k_mad_idx);
#if MOE_STREAM_W2_SCALES
                        if (not is_shared and not is_linear1 and sf_pair_idx % kNumSFPairsPerChunk == 0 and k_mad_idx == 0 and n_mad_idx == 0)
                            wait_intra_block<PIPE_MTE1, 0>(kStreamSFFull, w2_sf_stage_idx);
                        const auto b_scale = not is_shared and not is_linear1 ? l1_w2_sfb[w2_sf_stage_idx] : l1_sfb[sf_stage_idx];
#else
                        const auto b_scale = l1_sfb[sf_stage_idx];
#endif
                        copy_l1_to_l0b_mx(l0b[l0b_stage_idx], b_scale, n_mad_idx, sf_pair_idx_in_chunk, MAD_K);
                        notify<PIPE_MTE1, PIPE_M>(l0_full, l0b_stage_idx);

                        // B readiness also covers the earlier A/scales loads on the same MTE1 queue
                        const auto l0c_mad = l0c[l0c_idx].as_mad_aligned(aligned_valid_m, MAD_N);
                        wait<PIPE_MTE1, PIPE_M>(l0_full, l0b_stage_idx);
                        mad(l0c_mad, l0a_mad, l0b[l0b_stage_idx], aligned_valid_m, MAD_N, MAD_K,
                            is_last_k ? asc_unit_flag_mode::ENABLE_UPDATE : asc_unit_flag_mode::ENABLE_KEEP,
                            k_idx + k_mad_idx == 0);
                        if (n_mad_idx + MAD_N == BLOCK_N)
                            notify<PIPE_M, PIPE_MTE1>(kL0AEmpty, l0a_stage_idx);
                        notify<PIPE_M, PIPE_MTE1>(kL0BEmpty, l0b_stage_idx);
                        if (not is_last_k)
                            continue;

                        // Reuse output rows after all sends complete
                        if (block_phase == sched::BlockPhase::kLinear2 and n_mad_idx == 0 and pool_block_idx >= kNumRingBlocks)
                            comm::wait_for_block<workspace>(buffer, rank_idx, layout::kCombineSent,
                                                            num_shared_m_blocks + pool_block_idx - kNumRingBlocks, topology.cores());

                        // One epilogue credit covers BLOCK_N columns
                        const uint32_t fix_step_m = is_linear1 ? STORE_BLOCK_M : valid_m;
                        for (uint32_t m_idx_in_block = 0; m_idx_in_block < valid_m; m_idx_in_block += fix_step_m) {
                            const uint32_t ub_stage_idx = m_idx_in_block / STORE_BLOCK_M;
                            if (n_mad_idx == 0)
                                wait_intra_block<PIPE_FIX, 1>(kEpilogueEmpty, ub_stage_idx);
                            if (is_linear1) {
                                // Store MAD_N gate/up columns with a BLOCK_N destination stride
                                const ub_ptr<bfloat16_t> ub_dst(STORE_BLOCK_M, BLOCK_N,
                                                                ub_accum[ub_stage_idx].addr + n_mad_idx * sizeof(bfloat16_t));
                                copy_l0c_to_ub(ub_dst, l0c_mad.offset_nz_m(m_idx_in_block),
                                               min(STORE_BLOCK_M, valid_m - m_idx_in_block),
                                               MAD_N, asc_dual_dst_mode::DUAL_DST_DISABLE, 1);
                            } else {
                                // Store shared output to combine; routed output to the ring
                                const auto output_base = is_shared ? combine_input : routed_output;
                                const uint64_t output_m_idx = is_shared ?
                                    workspace.get_combine_m_idx(config.num_topk, static_cast<uint64_t>(pool_block_idx) * BLOCK_M) :
                                    layout::get_ring_m_idx<workspace>(pool_block_idx);
                                const auto l2_output = output_base + output_m_idx * config.hidden + n_block_idx * BLOCK_N + n_mad_idx;
                                copy_l0c_to_gm(l2_output, l0c_mad, valid_m, MAD_N, config.hidden, kWriteThroughShare);
                            }
                            if (n_mad_idx + MAD_N == BLOCK_N)
                                notify_intra_block<PIPE_FIX, 1>(kEpilogueFull, ub_stage_idx);
                        }
                    }
                }

                // Release L1 after the last L0 load
                if (is_shared) {
                    notify<PIPE_MTE1, PIPE_MTE2>(kSharedL1BEmpty, l1b_stage_idx);
                    l1b_stage_idx = (l1b_stage_idx + 1) % kNumSharedL1BStages;
                } else {
                    for (uint32_t n_mad_idx = 0; n_mad_idx < BLOCK_N; n_mad_idx += MAD_N)
                        notify_intra_block<PIPE_MTE1, 0>(kL1BEmpty, (l1b_stage_idx + n_mad_idx / MAD_N) % kNumL1BStages);
                    l1b_stage_idx = (l1b_stage_idx + (BLOCK_N / MAD_N)) % kNumL1BStages;
                }
                notify<PIPE_MTE1, PIPE_MTE2>(kL1AEmpty, l1a_stage_idx);
                l1a_stage_idx = (l1a_stage_idx + 1) % kNumL1AStages;

                if ((sf_pair_idx + BLOCK_K / MX_SF_DIVISOR) % kNumSFPairsPerChunk == 0 or k_idx + valid_k == num_k) {
                    notify<PIPE_MTE1, PIPE_MTE2>(kSFEmpty, sf_stage_idx);
#if MOE_STREAM_W2_SCALES
                    if (not is_shared and not is_linear1) {
                        notify_intra_block<PIPE_MTE1, 0>(kStreamSFEmpty, w2_sf_stage_idx);
                        w2_sf_stage_idx = (w2_sf_stage_idx + 1) % kNumSFStages;
                    }
#endif
                    sf_stage_idx = (sf_stage_idx + 1) % kNumSFStages;
                }
            }

        };
        // Shared tasks keep one shape so their L1B stages retain the same physical stride
        if ((is_shared ? num_tokens : valid_m) <= 128)
            compute.template operator()<128>();
        else
            compute.template operator()<256>();

#if MOE_TOKENS <= 80
        };
        if (block_phase == sched::BlockPhase::kSharedLinear1)
            run_task.template operator()<true, true>();
        else if (block_phase == sched::BlockPhase::kSharedLinear2)
            run_task.template operator()<true, false>();
        else if (block_phase == sched::BlockPhase::kLinear1)
            run_task.template operator()<false, true>();
        else
            run_task.template operator()<false, false>();
#endif

        asc_store_dev(num_issued_gemms_ptr, ++ num_issued_gemms);
    }

    // Consume initialized credits, including stages unused by short or empty inputs.
    wait<PIPE_M, PIPE_MTE1, kNumL0Stages>(kL0AEmpty);
    wait<PIPE_M, PIPE_MTE1, kNumL0Stages>(kL0BEmpty);
    wait<PIPE_MTE1, PIPE_MTE2, kNumL1AStages>(kL1AEmpty);
    wait<PIPE_MTE1, PIPE_MTE2, kNumSFStages>(kSFEmpty);
    if constexpr (kHasSharedExperts)
        wait<PIPE_MTE1, PIPE_MTE2, 3>(kSharedL1BEmpty);
    wait_intra_block<PIPE_FIX, 1, kNumEpilogueStages>(kEpilogueEmpty);
    if constexpr (kHasSharedExperts)
        asc_sync_data_barrier(DSB_DDR);
}

// AIV1: epilogue, dispatch, and combine
template <const auto& workspace, bool kOverlapReduce, bool kFullTopology>
__aicore__ inline void aiv1_epilogue_comm(__gm__ uint8_t* buffer, __gm__ bfloat16_t* output,
                                          uint32_t num_tokens, uint32_t rank_idx,
                                          CombineReduceState<workspace, kFullTopology>& reduce,
                                          float activation_clamp, uint32_t core_idx, const layout::SymBuffer<>& sym_buffer,
                                          CoreTopology<kFullTopology> topology) {
    constexpr auto& config = workspace.config;
    constexpr uint32_t kNumLinear1NBlocks = config.intermediate_hidden / (BLOCK_N / 2);
    constexpr uint32_t kNumLinear2NBlocks = config.hidden / BLOCK_N;
    constexpr uint32_t kNumSharedLinear1NBlocks = config.shared_intermediate_hidden / (BLOCK_N / 2);
    constexpr uint32_t kNumRingBlocks = workspace.get_num_ring_blocks();
    constexpr bool kHasSharedExperts = config.num_shared_experts > 0;

    constexpr uint32_t kNumOutputSFPairs = 2;
    constexpr uint32_t kNumUBAlignmentBytes = 32;

    // AIV1 UB in address order (bytes); H=hidden, R=num_ranks
    // Region        Buffers  Bytes/buffer
    // FIX accum           2  STORE_BLOCK_M * BLOCK_N * 2
    // Route weights       1  STORE_BLOCK_M * 4
    // Output SF           2  STORE_BLOCK_M * 4
    // Pull tokens         2  H
    // Pull SF             2  align(H/32, 32)
    // Source begin/end    2  R * 4
    // Push tokens         4  H * 2
    // Reduce              2  Fits remaining UB, rounded down to whole vectors
    // Source begin and push tokens start at 32-byte boundaries
    constexpr ub_ptr<bfloat16_t> ub_accum(STORE_BLOCK_M, BLOCK_N);
    constexpr ub_ptr<float> ub_route_weights(1, STORE_BLOCK_M, ub_accum.offset(kNumEpilogueStages));
    constexpr ub_ptr<int16_t> ub_output_sf(kNumOutputSFPairs, STORE_BLOCK_M, ub_route_weights.offset(1));

    // Dispatch inputs and source ranges
    constexpr ub_ptr<float8_e4m3_t> ub_pull_token(1, config.hidden, ub_output_sf.offset(kNumEpilogueStages));
    constexpr ub_ptr<int16_t> ub_pull_sf(1, aligned(config.num_input_sf_pairs * sizeof(int16_t), kNumUBAlignmentBytes) / sizeof(int16_t),
                                         ub_pull_token.offset(kNumPullStages));
    constexpr ub_ptr<uint32_t> ub_src_m_begin(1, config.num_ranks, aligned(ub_pull_sf.offset(kNumPullStages), kNumUBAlignmentBytes));
    constexpr ub_ptr<uint32_t> ub_src_m_end(1, config.num_ranks, ub_src_m_begin.offset(1));

    // Combine sends and reduction
    constexpr ub_ptr<bfloat16_t> ub_push_token(1, config.hidden, aligned(ub_src_m_end.offset(1), kNumUBAlignmentBytes));
    constexpr auto ub_reduce_inputs = make_reduce_buffer<workspace>(ub_push_token.offset(kNumPushStages));

    // Size the reduce window from local output tokens
    const uint32_t num_reduce_window_m_blocks = max(ceil_div(num_tokens, BLOCK_M), 1u);
    const uint32_t num_shared_m_blocks = kHasSharedExperts ? ceil_div(num_tokens, BLOCK_M) : 0;

    static_assert(ub_push_token.offset(kNumPushStages) <= UBSizeBytes and ub_reduce_inputs.shape_n > 0, "AIV1 UB overflow");

    notify_intra_block<PIPE_MTE3, 1, kNumEpilogueStages>(kEpilogueEmpty);
    notify<PIPE_MTE3, PIPE_MTE2, kNumPushStages>(kPushEmpty);
    notify<PIPE_V, PIPE_MTE2>(kTopkEmpty);
    notify<PIPE_MTE3, PIPE_MTE2, kNumPullStages>(kPullEmpty);

    const auto routed_weights = workspace.routed.topk_weights.get_ptr(buffer);
    const auto src_metadata = workspace.routed.src_metadata.get_ptr(buffer);
    const auto src_token_metadata = workspace.metadata.src_token_metadata.get_ptr(buffer);

    uint32_t num_total_m_blocks = 0;  // Routed pool only; shared blocks are separate
    bool metadata_ready = false;
    const auto num_m_blocks_certificate = workspace.metadata.num_m_blocks_certificate.get_ptr(buffer);

    const auto publish_block_progress = [&](layout::ProgressStage stage,
                                            uint32_t slot_begin, uint32_t num_slots) __aicore__ {
        // Wait for stores; dispatch and FIX callers already waited
        if (stage == layout::kLinear1Ready or stage == layout::kCombineSent) {
            notify<PIPE_MTE3, PIPE_S>(kStoresDone);
            wait<PIPE_MTE3, PIPE_S>(kStoresDone);
        }

        // Publish this core's contribution after its Scalar metadata is visible
        asc_sync_data_barrier(DSB_DDR);
        for (uint32_t slot_offset = 0; slot_offset < num_slots; ++ slot_offset)
            asc_atomic_add(workspace.progress.blocks.get_ptr(buffer, workspace.get_progress_idx(stage, slot_begin) + slot_offset), 1u);
    };

    const auto num_issued_gemms_ptr = workspace.progress.num_issued_gemms.get_ptr(buffer, core_idx);

    const auto l1_acts = workspace.routed.l1_acts.data.get_ptr(buffer);
    const auto l1_acts_sf = workspace.routed.l1_acts.sf.get_ptr(buffer);
    struct DispatchPullState {
        uint32_t pool_block_idx = 0;
        uint32_t m_idx_in_block;  // BLOCK_M marks pending stores
        uint32_t valid_m;
        sched::ExpertCursor<workspace> expert;

        // Permute the concatenated source rows without changing their counts
        uint32_t src_m_idx = 0, src_m_stride = 0, src_m_step = 0;
        uint32_t src_rank_idx = 0;
    };
    DispatchPullState pull{.m_idx_in_block = core_idx};

    // Pull one batch of input rows
    const auto advance_dispatch_pull = [&]() __aicore__ {
        if (pull.pool_block_idx == num_total_m_blocks)
            return;

        if (pull.m_idx_in_block >= BLOCK_M) {
            // Wait for the previous block's stores before publishing readiness
            wait<PIPE_MTE3, PIPE_S>(kPullDone);

            // Empty row owners must also contribute
            publish_block_progress(layout::kDispatched, num_shared_m_blocks + pull.pool_block_idx, 1);
            ++ pull.pool_block_idx;
            pull.m_idx_in_block = core_idx;
            return;
        }

        if (pull.m_idx_in_block == core_idx) {
            // Reuse input rows after all Linear1 N-blocks finish
            if (pull.pool_block_idx >= kNumRingBlocks and
                    not comm::is_block_ready<workspace>(buffer, layout::kLinear1Ready,
                                                        num_shared_m_blocks + pull.pool_block_idx - kNumRingBlocks,
                                                        kNumLinear1NBlocks))
                return;

            // Load source counts and advance the expert
            while (pull.pool_block_idx >= pull.expert.pool_block_end) {
                const uint32_t next_expert_idx = pull.expert.local_expert_idx + 1;
                uint32_t num_expert_tokens = 0, num_sources = 0;
                for (uint32_t src_rank_idx = 0; src_rank_idx < config.num_ranks; ++ src_rank_idx) {
                    const uint32_t count = asc_load_dev(workspace.metadata.expert_recv_count.get_ptr(
                        buffer, workspace.get_expert_recv_idx(src_rank_idx, next_expert_idx)));
                    ub_src_m_begin.ptr()[src_rank_idx] = num_expert_tokens;
                    num_expert_tokens += count;
                    ub_src_m_end.ptr()[src_rank_idx] = num_expert_tokens;
                    num_sources += count != 0;
                }
                pull.expert.next_expert(num_expert_tokens);
                pull.src_rank_idx = 0;
                if (num_expert_tokens != 0) {
                    pull.src_m_stride = next_coprime(ceil_div(num_expert_tokens, num_sources), num_expert_tokens) % num_expert_tokens;
                    pull.src_m_step = topology.cores() * pull.src_m_stride % num_expert_tokens;
                }
            }
            uint32_t expert_m_begin;
            pull.valid_m = pull.expert.get_m_block_range(pull.pool_block_idx, expert_m_begin);
            pull.src_m_idx = static_cast<uint64_t>(expert_m_begin + core_idx) * pull.src_m_stride % pull.expert.num_expert_tokens;
            if (pull.src_m_idx < ub_src_m_begin.ptr()[pull.src_rank_idx])
                pull.src_rank_idx = 0;
        }

        for (uint32_t step_idx = 0; step_idx < kNumPullStages and pull.m_idx_in_block < pull.valid_m; ++ step_idx) {
            const uint32_t m_idx_in_block = pull.m_idx_in_block;
            pull.m_idx_in_block += topology.cores();
            const uint32_t stage_idx = m_idx_in_block / topology.cores() % kNumPullStages;
            const uint64_t pool_m_idx = static_cast<uint64_t>(pull.pool_block_idx) * BLOCK_M + m_idx_in_block;
            const uint64_t ring_m_idx = layout::get_ring_m_idx<workspace>(pull.pool_block_idx, m_idx_in_block);
            while (pull.src_m_idx >= ub_src_m_end.ptr()[pull.src_rank_idx])
                ++ pull.src_rank_idx;
            const uint32_t src_rank_idx = pull.src_rank_idx;
            const uint32_t src_slot_idx = pull.src_m_idx - ub_src_m_begin.ptr()[src_rank_idx];

            pull.src_m_idx += pull.src_m_step;
            if (pull.src_m_idx >= pull.expert.num_expert_tokens) {
                pull.src_m_idx -= pull.expert.num_expert_tokens;
                pull.src_rank_idx = 0;
            }
            const auto remote_base = sym_buffer.map(buffer, src_rank_idx);

            // Resolve the source token
            wait<PIPE_MTE3, PIPE_MTE2>(kPullEmpty, stage_idx);
            const uint64_t token_metadata = asc_load_dev(src_token_metadata +
                workspace.get_src_metadata_idx(pull.expert.local_expert_idx, src_rank_idx, src_slot_idx));
            const uint32_t token_topk_idx = static_cast<uint32_t>(token_metadata);
            const uint32_t src_token_idx = token_topk_idx / config.num_topk;
            const uint32_t src_topk_idx = token_topk_idx % config.num_topk;
            const auto remote_token = workspace.input.acts.data.get_ptr(remote_base, static_cast<uint64_t>(src_token_idx) * config.hidden);
            const auto remote_sf = workspace.input.acts.sf.get_ptr(remote_base, static_cast<uint64_t>(src_token_idx) * config.num_input_sf_pairs);

            // Pull token data and scales
            copy_gm_to_ub(ub_pull_token[stage_idx].ptr(), remote_token,
                          1, config.hidden, config.hidden, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
            copy_gm_to_ub(ub_pull_sf[stage_idx].ptr(), remote_sf, 1, config.num_input_sf_pairs,
                          config.num_input_sf_pairs * sizeof(int16_t), asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM);
            notify<PIPE_MTE2, PIPE_MTE3>(kPullFull, stage_idx);

#if MOE_PACKED_ROUTED_WEIGHTS
            notify<PIPE_MTE2, PIPE_V>(PipeEvent<PIPE_MTE2, PIPE_V>{2}, stage_idx);
            wait<PIPE_MTE2, PIPE_V>(PipeEvent<PIPE_MTE2, PIPE_V>{2}, stage_idx);
            compensate_routed_scale(reinterpret_cast<__ubuf__ uint8_t*>(ub_pull_sf[stage_idx].ptr()),
                                    config.num_input_sf_pairs * sizeof(int16_t));
            notify<PIPE_V, PIPE_MTE3>(PipeEvent<PIPE_V, PIPE_MTE3>{2}, stage_idx);
            wait<PIPE_V, PIPE_MTE3>(PipeEvent<PIPE_V, PIPE_MTE3>{2}, stage_idx);
#endif
            // Stage routed inputs
            wait<PIPE_MTE2, PIPE_MTE3>(kPullFull, stage_idx);
            copy_ub_to_gm(l1_acts + ring_m_idx * config.hidden,
                          ub_pull_token[stage_idx].ptr(), 1, config.hidden,
                          config.hidden, kWriteThroughShare);
            copy_ub_to_gm(l1_acts_sf + ring_m_idx * config.num_input_sf_pairs,
                          ub_pull_sf[stage_idx].ptr(), 1, config.num_input_sf_pairs,
                          config.num_input_sf_pairs * sizeof(int16_t), kWriteThroughShare);
            notify<PIPE_MTE3, PIPE_MTE2>(kPullEmpty, stage_idx);

            // Copy route weights as FP32 bits while DMA runs
            const uint32_t route_weight_bits = static_cast<uint32_t>(token_metadata >> 32);
            asc_store_dev(reinterpret_cast<__gm__ uint32_t*>(routed_weights + pool_m_idx), route_weight_bits);
            // Record the actual row assignment; combine readiness never infers source/token order
            if constexpr (kOverlapReduce)
                asc_store_dev(workspace.route_m_blocks.get_ptr(remote_base, token_topk_idx), pull.pool_block_idx);
            // Source metadata: rank [31:24], combine M [23:0]
            asc_store_dev(src_metadata + pool_m_idx,
                static_cast<uint32_t>(workspace.get_combine_m_idx(src_topk_idx, src_token_idx)) | (src_rank_idx << 24));
        }

        if (pull.m_idx_in_block >= pull.valid_m) {
            // BLOCK_M marks pending stores; publish on the next step
            pull.m_idx_in_block = BLOCK_M;
            notify<PIPE_MTE3, PIPE_S>(kPullDone);
        }
    };

    struct EpilogueState {
        sched::LinearTaskState schedule;
        sched::ExpertCursor<workspace> expert;
        uint32_t task_idx;
        uint32_t valid_m;
        sched::BlockPhase phase = sched::BlockPhase::kNone;
        uint32_t m_idx_in_block = 0;
        uint8_t num_completed_gemms = 0;  // Advanced by epilogue, paired with num_issued_gemms
    };
    EpilogueState epi{};

    // Process Linear1 rows or wait for Linear2 stores; return true on progress
    const auto advance_epilogue = [&]() __aicore__ -> bool {
        if (epi.phase == sched::BlockPhase::kNone)
            return false;

        // Issued tasks still need their FIX completion event
        if (epi.m_idx_in_block == 0 and asc_load_dev(num_issued_gemms_ptr) == epi.num_completed_gemms)
            return false;

        const bool is_shared = epi.phase == sched::BlockPhase::kSharedLinear1 or epi.phase == sched::BlockPhase::kSharedLinear2;
        const bool is_linear1 = epi.phase == sched::BlockPhase::kLinear1 or epi.phase == sched::BlockPhase::kSharedLinear1;
        const uint32_t num_n_blocks = is_linear1 ? (is_shared ? kNumSharedLinear1NBlocks : kNumLinear1NBlocks) : kNumLinear2NBlocks;
        const uint32_t pool_block_idx = epi.task_idx / num_n_blocks;

        if (not is_linear1) {
            // Wait for Linear2 stores and return FIX credit
            wait_intra_block<PIPE_S, 1>(kEpilogueFull);
            notify_intra_block<PIPE_S, 1>(kEpilogueEmpty);
        } else {
            // Reuse Linear2 inputs after all output N-blocks finish
            if (not is_shared and epi.m_idx_in_block == 0 and pool_block_idx >= kNumRingBlocks and
                    not comm::is_block_ready<workspace>(buffer, layout::kLinear2Ready,
                                                        num_shared_m_blocks + pool_block_idx - kNumRingBlocks,
                                                        kNumLinear2NBlocks))
                return false;

            if (epi.m_idx_in_block == 0)
                epi.valid_m = is_shared ? min(num_tokens - pool_block_idx * BLOCK_M, BLOCK_M) :
                    epi.expert.advance_to(buffer, pool_block_idx);
            const uint32_t stage_idx = epi.m_idx_in_block / STORE_BLOCK_M;
            const uint32_t m_idx_in_stage = epi.m_idx_in_block % STORE_BLOCK_M;
            const uint32_t stage_m_begin = stage_idx * STORE_BLOCK_M;
            const uint32_t stage_valid_m = min(STORE_BLOCK_M, epi.valid_m - stage_m_begin);
            // Yield after at most 32 rows
            const uint32_t step_m = min(32u, epi.valid_m - epi.m_idx_in_block);

            // Load route weights and wait for FIX
            if (m_idx_in_stage == 0) {
                if (not is_shared) {
                    const auto route_weights = workspace.routed.topk_weights.get_ptr(
                        buffer, static_cast<uint64_t>(pool_block_idx) * BLOCK_M + stage_m_begin);
                    wait<PIPE_V, PIPE_MTE2>(kTopkEmpty);
                    copy_gm_to_ub(ub_route_weights.ptr(), route_weights, 1, stage_valid_m,
                                  stage_valid_m * sizeof(float), asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM);
                    notify<PIPE_MTE2, PIPE_V>(kTopkFull);
                }
                wait_intra_block<PIPE_V, 1>(kEpilogueFull, stage_idx);
                if (not is_shared)
                    wait<PIPE_MTE2, PIPE_V>(kTopkFull);
            }

            // Quantize one row group
            const auto input = ub_accum[stage_idx].ptr() + m_idx_in_stage * BLOCK_N;
            const auto output = reinterpret_cast<__ubuf__ float8_e4m3_t*>(ub_accum[stage_idx].ptr()) +
                                m_idx_in_stage * (BLOCK_N / 2);
            const auto output_sf = ub_output_sf[stage_idx].ptr() + m_idx_in_stage;
            if (is_shared) {
                vf_swiglu_quant<false, STORE_BLOCK_M>(input, output, output_sf, nullptr, step_m, activation_clamp);
            } else {
                vf_swiglu_quant<true, STORE_BLOCK_M>(input, output, output_sf, ub_route_weights.ptr() + m_idx_in_stage, step_m, activation_clamp);
            }
            epi.m_idx_in_block += step_m;

            // Finish GM stores before FIX reuses this UB stage
            if (m_idx_in_stage + step_m == stage_valid_m) {
                if (not is_shared)
                    notify<PIPE_V, PIPE_MTE2>(kTopkEmpty);
                notify<PIPE_V, PIPE_MTE3>(kQuantFull);
                wait<PIPE_V, PIPE_MTE3>(kQuantFull);

                const uint32_t n_block_idx = epi.task_idx % num_n_blocks;
                const uint64_t output_m_idx = static_cast<uint64_t>(
                    is_shared ? pool_block_idx : layout::get_ring_block_idx<workspace>(pool_block_idx)) * BLOCK_M;
                const auto dst_layout = is_shared ? workspace.shared_l2_acts : workspace.routed.l2_acts;
                const uint32_t num_cols = num_n_blocks * (BLOCK_N / 2);
                const uint32_t num_sf_pairs = num_cols / MX_SF_DIVISOR;
                const auto dst = dst_layout.data.get_ptr(buffer,
                    (output_m_idx + stage_m_begin) * num_cols + n_block_idx * (BLOCK_N / 2));
                const auto dst_sf = dst_layout.sf.get_ptr(buffer,
                    output_m_idx * num_sf_pairs + n_block_idx * kNumOutputSFPairs * BLOCK_M + stage_m_begin);

                copy_ub_to_gm(dst, ub_ptr<float8_e4m3_t>(STORE_BLOCK_M, BLOCK_N / 2, ub_accum[stage_idx].addr),
                              stage_valid_m, BLOCK_N / 2, num_cols, kWriteThroughShare);
                copy_ub_to_gm(dst_sf, ub_output_sf[stage_idx],
                              kNumOutputSFPairs, stage_valid_m, BLOCK_M * sizeof(int16_t), kWriteThroughShare);
                notify_intra_block<PIPE_MTE3, 1>(kEpilogueEmpty, stage_idx);
            }
            if (epi.m_idx_in_block != epi.valid_m)
                return true;
            epi.m_idx_in_block = 0;
        }

        ++ epi.num_completed_gemms;

        // Shared early reduce counts each active core once in slot 0
        if (is_linear1)
            publish_block_progress(layout::kLinear1Ready, (is_shared ? 0 : num_shared_m_blocks) + pool_block_idx, 1);
        else if (not is_shared or (kOverlapReduce and epi.task_idx + topology.cores() >= num_shared_m_blocks * kNumLinear2NBlocks))
            publish_block_progress(layout::kLinear2Ready, is_shared ? 0 : num_shared_m_blocks + pool_block_idx, 1);
        epi.phase = sched::BlockPhase::kNone;
        return true;
    };

    const auto routed_output = workspace.routed.output.get_ptr(buffer);
    struct CombinePushState {
        sched::ExpertCursor<workspace> expert;
        uint32_t pool_block_idx = 0;
        uint32_t m_idx_in_block;
        uint32_t valid_m;
        uint32_t num_completed_m_blocks = 0;
    };
    CombinePushState push{.m_idx_in_block = core_idx};

    // Send BF16 rows; return true when this step advances
    const auto advance_combine_push = [&]() __aicore__ -> bool {
        // Batch store waits to overlap remote writes with other work
        const uint32_t num_m_blocks_per_wait = max(1u, 32 * topology.cores() / BLOCK_M);
        if (push.pool_block_idx == num_total_m_blocks or (push.m_idx_in_block == core_idx and
                not comm::is_block_ready<workspace>(buffer, layout::kLinear2Ready,
                                                    num_shared_m_blocks + push.pool_block_idx,
                                                    kNumLinear2NBlocks)))
            return false;

        const uint32_t pool_block_idx = push.pool_block_idx;
        if (push.m_idx_in_block == core_idx)
            push.valid_m = push.expert.advance_to(buffer, pool_block_idx);
        const auto src_metadata_base = src_metadata + static_cast<uint64_t>(pool_block_idx) * BLOCK_M;
        const auto src_block = routed_output + layout::get_ring_m_idx<workspace>(pool_block_idx) * config.hidden;

        // Send at most two rows before yielding
        for (uint32_t step_idx = 0; step_idx < 2 and push.m_idx_in_block < push.valid_m; ++ step_idx) {
            const uint32_t m_idx_in_block = push.m_idx_in_block;
            const uint32_t stage_idx = (m_idx_in_block / topology.cores()) % kNumPushStages;
            const uint32_t combine_metadata = asc_load_dev(src_metadata_base + m_idx_in_block);
            const uint32_t combine_m_idx = combine_metadata & 0x00ffffffu;
            const auto src = src_block + m_idx_in_block * config.hidden;

            // Load a BF16 output row
            wait<PIPE_MTE3, PIPE_MTE2>(kPushEmpty, stage_idx);
            copy_gm_to_ub(ub_push_token[stage_idx].ptr(), src, 1, config.hidden,
                          config.hidden * sizeof(bfloat16_t), asc_load_l2_cache_mode::NOTALLOC_KEEP);
            notify<PIPE_MTE2, PIPE_MTE3>(kPushFull, stage_idx);

            const uint32_t dst_rank_idx = combine_metadata >> 24;
            const auto dst = sym_buffer.map(workspace.combine_input.get_ptr(buffer, static_cast<uint64_t>(combine_m_idx) * config.hidden), dst_rank_idx);

            // Send the BF16 row to its source token
            wait<PIPE_MTE2, PIPE_MTE3>(kPushFull, stage_idx);
            copy_ub_to_gm(dst, ub_push_token[stage_idx].ptr(), 1, config.hidden,
                          config.hidden * sizeof(bfloat16_t), asc_store_l2_cache_mode::NOTALLOC_CLEAN);
            notify<PIPE_MTE3, PIPE_MTE2>(kPushEmpty, stage_idx);
            push.m_idx_in_block += topology.cores();
        }

        if (push.m_idx_in_block >= push.valid_m) {
            push.m_idx_in_block = core_idx;
            ++ push.pool_block_idx;

            // Wait for batched stores; publish each last-expert block for early reduce
            if (push.pool_block_idx % num_m_blocks_per_wait == 0 or
                    (kOverlapReduce ? push.expert.pool_block_end == num_total_m_blocks : push.pool_block_idx == num_total_m_blocks)) {
                publish_block_progress(layout::kCombineSent,
                                       num_shared_m_blocks + push.num_completed_m_blocks, push.pool_block_idx - push.num_completed_m_blocks);
                push.num_completed_m_blocks = push.pool_block_idx;
            }
        }
        return true;
    };

    // Publish the exact completed-block prefix shared by all source ranks
    struct CombinePrefixState {
        uint32_t num_published_m_blocks = 0;
        bool finished;
    };
    CombinePrefixState ready_prefix{.finished = not kOverlapReduce or core_idx >= config.num_ranks};

    const auto publish_combine_prefix = [&]() __aicore__ {
        const uint32_t pool_block_end = push.num_completed_m_blocks;
        if (ready_prefix.finished or (num_total_m_blocks != 0 and pool_block_end <= ready_prefix.num_published_m_blocks))
            return;
        if (pool_block_end != 0 and not comm::is_block_ready<workspace>(
                buffer, layout::kCombineSent, num_shared_m_blocks + pool_block_end - 1, topology.cores()))
            return;

        ready_prefix.num_published_m_blocks = pool_block_end;
        ready_prefix.finished = pool_block_end == num_total_m_blocks;
        const uint64_t prefix = ready_prefix.finished ? ~uint64_t{0} : pool_block_end;
        for (uint32_t src_rank_idx = core_idx; src_rank_idx < config.num_ranks; src_rank_idx += topology.cores()) {
            asc_sync_data_barrier(DSB_DDR);
            asc_store_dev(sym_buffer.map(workspace.metadata.combine_ready_prefix.get_ptr(buffer, rank_idx), src_rank_idx), prefix);
        }
    };

    bool prefer_push = false;
    comm::wait_until([&]() __aicore__ {
        // Service the reduce window; check token readiness inside
        const bool reduce_window_open = kOverlapReduce and metadata_ready and push.pool_block_idx + num_reduce_window_m_blocks >= num_total_m_blocks and
            asc_load_dev(num_issued_gemms_ptr) == epi.num_completed_gemms;

        if (reduce_window_open) {
            publish_combine_prefix();
            if (num_total_m_blocks > kNumRingBlocks)
                advance_combine_reduce<workspace, kOverlapReduce>(buffer, output, num_tokens, reduce, ub_reduce_inputs, topology, 2);
        }

        // Fetch the next epilogue task
        if (epi.phase == sched::BlockPhase::kNone)
            epi.task_idx = sched::take_linear_task<workspace>(
                num_tokens, num_total_m_blocks, epi.schedule, core_idx, epi.phase, topology, metadata_ready);

        // Poll metadata arrival
        if (not metadata_ready and epi.phase != sched::BlockPhase::kSharedLinear1 and
            asc_load_dev(workspace.progress.metadata_ready.get_ptr(buffer, core_idx)) != 0) {
            asc_sync_data_barrier(DSB_DDR);
            num_total_m_blocks = asc_load_dev(num_m_blocks_certificate) & layout::kCountValueMask;
            metadata_ready = true;
        }

        const bool pull_finished = metadata_ready and pull.pool_block_idx == num_total_m_blocks;
        if (pull_finished and epi.phase == sched::BlockPhase::kNone and push.pool_block_idx == num_total_m_blocks and ready_prefix.finished)
            return true;

        // Advance epilogue before choosing push or pull
        const bool epi_progress = advance_epilogue();
        if (metadata_ready and not (pull_finished and epi_progress)) {
            if ((prefer_push or pull_finished) and advance_combine_push()) {
                prefer_push = false;
            } else if (not pull_finished) {
                // Fall back to pull; prefer push again on the next step
                advance_dispatch_pull();
                prefer_push = true;
            }
        }
        return false;
    }, [&]() __aicore__ { comm::print_local_timeout(rank_idx); });

    // Consume communication credits after all epilogue, pull, and push work.
    wait<PIPE_MTE3, PIPE_MTE2, kNumPullStages>(kPullEmpty);
    wait<PIPE_MTE3, PIPE_MTE2, kNumPushStages>(kPushEmpty);
    wait<PIPE_V, PIPE_MTE2>(kTopkEmpty);
    if constexpr (kHasSharedExperts)
        asc_sync_data_barrier(DSB_DDR);
}

template <uint32_t kNumRanks, uint32_t kNumExperts, uint32_t kNumSharedExperts,
          uint32_t kNumMaxTokensPerRank, uint32_t kNumTopk, uint32_t kHidden, uint32_t kIntermediateHidden,
          bool kFullTopology>
__aicore__ inline void mega_moe_device(__gm__ bfloat16_t* output, __gm__ int32_t* cumulative_local_expert_recv_stats,
                                            const layout::Weights& weights,
                                            uint32_t num_tokens, float activation_clamp,
                                            layout::SymBuffer<> sym_buffer, uint32_t logical_core_idx, uint32_t logical_subblock_idx,
                                            CoreTopology<kFullTopology> topology
                                            , __gm__ uint32_t* local_route_ids, __gm__ uint32_t* local_route_weights
                                            ) {
    constexpr auto& workspace = layout::kWorkspace<kNumRanks, kNumExperts, kNumSharedExperts,
                                                   kNumMaxTokensPerRank, kNumTopk, kHidden, kIntermediateHidden>;
    constexpr auto& config = workspace.config;

    // Measured 32 MiB combine-work threshold for tail reduction
    // For the balanced 4096-row specialization, waiting once after the last
    // combine avoids repeatedly polling route readiness across all experts.
    constexpr bool kOverlapReduce = static_cast<uint64_t>(config.num_max_tokens_per_rank) *
        (config.num_topk + (config.num_shared_experts > 0)) * config.hidden * sizeof(bfloat16_t) > 32ull * 1024 * 1024;

    constexpr uint32_t kNumExpertsPerRank = config.num_experts / config.num_ranks;

    // Block readiness reads the low byte of each completion counter
    static_assert(config.hidden / BLOCK_N <= UINT8_MAX and
                  config.intermediate_hidden / (BLOCK_N / 2) <= UINT8_MAX and
                  config.shared_intermediate_hidden / (BLOCK_N / 2) <= UINT8_MAX);

    asc_init();
    const uint32_t core_idx = logical_core_idx;

    const auto buffer = sym_buffer.get_base_ptr<__gm__ uint8_t*>();
    const uint32_t rank_idx = sym_buffer.rank_idx;
    const auto topk_idx = local_route_ids;
    const auto num_m_blocks_certificate = workspace.metadata.num_m_blocks_certificate.get_ptr(buffer);

    // AIC: GEMMs, then wait for both AIVs
    if ASCEND_IS_AIC {
        aic_gemm_pipeline<workspace>(buffer, num_tokens, rank_idx, weights, core_idx, topology);
        asc_sync_block_wait(PIPE_S, kCombineDoneID);
        asc_sync_pipe(PIPE_ALL);
        return;
    }

    // Partition tokens across AIVs
    const uint32_t subblock_idx = logical_subblock_idx;
    const uint32_t aiv_idx = 2 * core_idx + subblock_idx;

    // Overlap mode revisits round-robin tokens; the final-only path uses contiguous ranges
    const uint32_t first_token_idx = kOverlapReduce ? aiv_idx : min(aiv_idx * ceil_div(num_tokens, topology.vectors()), num_tokens);
    const uint32_t num_reduce_tokens = kOverlapReduce ?
        (num_tokens > aiv_idx ? ceil_div(num_tokens - aiv_idx, topology.vectors()) : 0) :
        min(ceil_div(num_tokens, topology.vectors()), num_tokens - first_token_idx);
    CombineReduceState<workspace, kFullTopology> reduce{.token_idx = first_token_idx, .num_remaining_tokens = num_reduce_tokens};
    reduce.local_route_ids = local_route_ids;

    // Clear counts before metadata publication
    for (uint32_t expert_idx = aiv_idx; expert_idx < kNumExpertsPerRank; expert_idx += topology.vectors())
        asc_store_dev(workspace.metadata.expert_recv_count_sum.get_ptr(buffer, expert_idx), uint32_t{0});
    if (aiv_idx == 0)
        asc_store_dev(num_m_blocks_certificate, uint32_t{0});
    if (aiv_idx < kNumExpertsPerRank)
        asc_sync_data_barrier(DSB_DDR);
    comm::rank_barrier<workspace>(buffer, sym_buffer, aiv_idx);

    // AIV0: metadata and routed weights
    if (subblock_idx == 0) {
        constexpr bool direct = comm::kDirectMetadata<workspace> && !kOverlapReduce;
        const uint32_t num_metadata_workers = direct ? 1 :
            min(max(ceil_div(num_tokens * config.num_topk, layout::kNumDispatchThreads), 1u), topology.cores());
        if (core_idx < num_metadata_workers) {
            // SIMT reads UB offsets; metadata owns this scratch until the VF completes
            auto* offsets = reinterpret_cast<__ubuf__ int64_t*>(0);
            for (uint32_t i = 0; i < kNumRanks; ++ i)
                offsets[i] = sym_buffer.offsets[i];
            asc_sync_data_barrier(DSB_UB);
            if constexpr (direct) {
                auto* weights = local_route_weights;
                asc_vf_call<comm::build_direct_metadata<workspace>>(
                    dim3(layout::kNumDispatchThreads, 1, 1), buffer, topk_idx, weights, num_tokens, rank_idx, offsets);
            } else {
                asc_vf_call<comm::build_dispatch_metadata<workspace, kOverlapReduce, kFullTopology>>(
                    dim3(layout::kNumDispatchThreads, 1, 1),
                    buffer, topk_idx,
                    local_route_weights,
                    num_tokens, rank_idx, core_idx, num_metadata_workers, offsets);
            }
            notify<PIPE_V, PIPE_S>(kMetadataDone);
            wait<PIPE_V, PIPE_S>(kMetadataDone);
        }

        const uint32_t num_total_m_blocks = comm::wait_dispatch_metadata<workspace>(buffer, rank_idx, core_idx, cumulative_local_expert_recv_stats, topology);

        // Notify AIV1 via GM and AIC via event
        asc_store_dev(workspace.progress.metadata_ready.get_ptr(buffer, core_idx), uint8_t{1});
        notify_intra_block<PIPE_S, 0>(kMetadataFull);

        aiv0_weight_pipeline<workspace, kOverlapReduce>(buffer, output, num_tokens, reduce, num_total_m_blocks, weights, core_idx, topology);
    } else {
        // AIV1: epilogue and communication
        aiv1_epilogue_comm<workspace, kOverlapReduce>(buffer, output, num_tokens, rank_idx, reduce, activation_clamp, core_idx, sym_buffer, topology);
    }

    combine_reduce<workspace, kOverlapReduce>(buffer, output, num_tokens, rank_idx, aiv_idx, reduce, sym_buffer, topology);

    // Notify AIC after final output stores
    asc_sync_block_arrive(PIPE_MTE3, kCombineDoneID);
    asc_sync_pipe(PIPE_ALL);
}

} // namespace deep_gemm::mega_moe
// END DeepSeek MIT port: include/deep_gemm/mega_moe.hpp

// BEGIN PyPTO adapter: symmetric buffer binding

/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <tensor.h>
#include <intrinsic.h>
#include <platform_comm/comm_context.h>

// The window is an internal byte allocation owned by this MoE communication
// domain. The wrapper passes runtime context indices from its generated ABI.
template <uint32_t NumRanks, uint64_t RequiredBytes>
__aicore__ inline deep_gemm::layout::SymBuffer<> bind_pypto_sym_buffer(
    __gm__ int64_t* args, int tensor_index, int context_index)
{
    auto* tensor = reinterpret_cast<__gm__ Tensor*>(args[tensor_index]);
    auto* context = reinterpret_cast<__gm__ CommContext*>(args[context_index]);
    dcci(tensor, SINGLE_CACHE_LINE);
    dcci(context, SINGLE_CACHE_LINE);
    if (context->rankNum != NumRanks || context->rankId >= NumRanks) trap();
    const uint32_t rank = context->rankId;
    dcci(context->windowsIn + rank, SINGLE_CACHE_LINE);
    const uint64_t local_window = context->windowsIn[rank];
    const uint64_t local_tensor = tensor->buffer.addr + tensor->start_offset;
    if (local_tensor < local_window || RequiredBytes > context->winSize ||
        local_tensor - local_window > context->winSize - RequiredBytes) trap();
    deep_gemm::layout::SymBuffer<> result{};
    result.base = static_cast<int64_t>(local_tensor);
    result.rank_idx = rank;
    for (uint32_t peer = 0; peer < NumRanks; ++peer) {
        dcci(context->windowsIn + peer, SINGLE_CACHE_LINE);
        result.offsets[peer] = static_cast<int64_t>(context->windowsIn[peer] - local_window);
    }
    return result;
}
// END PyPTO adapter: symmetric buffer binding

// BEGIN PyPTO adapter: BF16 input publication

/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include <pto/pto-inst.hpp>

#include "mx_quant.hpp"
namespace v41_official_input {

using namespace pto;

template <typename T, int R, int C>
using Vec = Tile<TileType::Vec, T, R, C, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
template <typename T>
using Global = GlobalTensor<T, Shape<1, 1, 1, DYNAMIC, DYNAMIC>,
                            pto::Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, 1>>;

template <typename T>
__aicore__ inline Global<T> global(__gm__ T* pointer, uint32_t rows, uint32_t cols, uint32_t stride)
{
    return Global<T>(pointer, typename Global<T>::Shape(1, 1, 1, rows, cols),
                     typename Global<T>::Stride(1, 1, 1, stride, 1));
}

// All scratch is dead before the official persistent pipeline initializes its UB.
template <const auto& workspace, bool kFullTopology>
__aicore__ inline void publish(__gm__ uint8_t* buffer, __gm__ bfloat16_t* x,
                               __gm__ bfloat16_t* output,
                               uint32_t active, uint32_t aiv_idx, deep_gemm::CoreTopology<kFullTopology> topology)
{
#ifdef __DAV_VEC__
    constexpr auto& config = workspace.config;
    constexpr uint32_t rows = 16;
    constexpr uint32_t width = config.hidden;
    constexpr uint32_t pairs = width / 64;
    static_assert(config.num_max_tokens_per_rank % rows == 0);
    static_assert(rows * width * sizeof(bfloat16_t) <= PTO_UBUF_SIZE_BYTES);

    // Only progress is reset. Rank-barrier epochs and grid-sync phase bits persist.
    Vec<uint32_t, 1, 128> zero_words(1, 128);
    TASSIGN(zero_words, 0);
    TEXPANDS(zero_words, uint32_t{0});
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    for (uint32_t offset = aiv_idx * 512; offset < workspace.num_progress_bytes;
         offset += topology.vectors() * 512) {
        auto target = global(reinterpret_cast<__gm__ uint32_t*>(
            buffer + workspace.progress.blocks.offset + offset), 1, 128, 128);
        TSTORE(target, zero_words);
    }
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);

    constexpr uint32_t chunks = width / 1024;
    static_assert(width % 1024 == 0);
    for (uint32_t task = aiv_idx; task < (config.num_max_tokens_per_rank / rows) * chunks; task += topology.vectors()) {
        const uint32_t first = (task / chunks) * rows;
        const uint32_t col = (task % chunks) * 1024;
        // Pack four adjacent 256-column groups so each scale row is a full DMA block.
        constexpr uint32_t tile_bytes = rows * 256;
        constexpr uint32_t output_base = 4 * tile_bytes * sizeof(bfloat16_t);
        constexpr uint32_t scale_base = output_base + 4 * tile_bytes;
        static_assert(scale_base + rows * 32 <= PTO_UBUF_SIZE_BYTES);
        moe_input_quant::DirectScales scratch;
        TASSIGN(scratch, scale_base);
        #pragma unroll
        for (uint32_t part = 0; part < 4; ++part) {
            moe_input_quant::Input input;
            TASSIGN(input, part * tile_bytes * sizeof(bfloat16_t));
            auto src = global(x + first * width + col + part * 256,
                              rows, 256, width);
            TLOAD(input, src);
            set_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(part));
        }
        #pragma unroll
        for (uint32_t part = 0; part < 4; ++part) {
            moe_input_quant::Input input;
            moe_input_quant::Payload payload;
            TASSIGN(input, part * tile_bytes * sizeof(bfloat16_t));
            TASSIGN(payload, output_base + part * tile_bytes);
            wait_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(part));
            moe_input_quant::quantize<true, moe_input_quant::DirectScales>(
                payload.data(), scratch.data(), input.data(), part);
            set_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(part));
            wait_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(part));
            auto dst = global(workspace.input.acts.data.template get_ptr<uint8_t>(
                buffer, first * width + col + part * 256), rows, 256, width);
            TSTORE(dst, payload);
        }
        Vec<uint16_t, rows, 16> scales(rows, 16);
        TASSIGN(scales, scale_base);
        auto sf=global(workspace.input.acts.sf.template get_ptr<uint16_t>(buffer,first*pairs+col/64),rows,16,pairs);
        TSTORE(sf,scales);
        pipe_barrier(PIPE_ALL);
    }
    for (uint32_t first = aiv_idx * rows; first < config.num_max_tokens_per_rank;
         first += topology.vectors() * rows) {
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
        const uint32_t tail_begin = first > active ? first : active;
        if (tail_begin < first + rows) {
            Vec<bfloat16_t, rows, width> zeros(first + rows - tail_begin, width);
            TASSIGN(zeros, 0);
            TEXPANDS(zeros, bfloat16_t{0});
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
            auto tail = global(output + tail_begin * width, first + rows - tail_begin, width, width);
            TSTORE(tail, zeros);
            set_flag(PIPE_MTE3, PIPE_S, EVENT_ID2);
            wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID2);
        }
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
#endif
}

// AIC can start shared Linear1 immediately inside the official kernel. Release
// each AIC only after every launched AIV has finished local input/progress IO.
__aicore__ inline void publication_fence(uint32_t subblock)
{
#ifdef __DAV_VEC__
    asc_sync_inter_arrive(PIPE_MTE3, 15);
    asc_sync_inter_wait(PIPE_S, 15);
    if (subblock == 0)
        deep_gemm::notify_intra_block<PIPE_S, 0>(deep_gemm::IntraBlockEvent<0>{15});
#else
    deep_gemm::wait_intra_block<PIPE_S, 0>(deep_gemm::IntraBlockEvent<0>{15});
#endif
}

} // namespace v41_official_input
// END PyPTO adapter: BF16 input publication

#if !defined(MOE_EP) || !defined(MOE_EXPERTS) || !defined(MOE_TOKENS) || \
    !defined(MOE_TOPK) || !defined(MOE_HIDDEN) || !defined(MOE_INTER)
#error "The immutable native specialization must define all MEGA shape macros."
#endif

namespace v41_official_entry {

template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, uint32_t index)
{
    auto* tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    if constexpr (MOE_TOKENS > 80)
        dcci(tensor, SINGLE_CACHE_LINE);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

__aicore__ inline int64_t dimension(__gm__ int64_t* args, uint32_t index, uint32_t axis)
{
    auto* tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    if constexpr (MOE_TOKENS > 80) {
        dcci(tensor, SINGLE_CACHE_LINE);
        dcci(reinterpret_cast<__gm__ uint8_t*>(tensor) + 64, SINGLE_CACHE_LINE);
    }
    return tensor->shapes[axis];
}

__aicore__ inline void acquire_metadata(__gm__ int64_t* args)
{
    static_assert(alignof(Tensor) >= 64 && sizeof(Tensor) == 128);
    static_assert(offsetof(Tensor, start_offset) + sizeof(uint64_t) <= 64);
    static_assert(offsetof(Tensor, shapes) + sizeof(Tensor::shapes) <= 128);
    #pragma unroll
    for (uint32_t index = 0; index < 16; ++index)
        dcci(reinterpret_cast<__gm__ Tensor*>(args[index]), SINGLE_CACHE_LINE);
    // The symmetric binding acquires argument 16's first metadata line.
    dcci(reinterpret_cast<__gm__ uint8_t*>(args[0]) + 64, SINGLE_CACHE_LINE);
    dcci(reinterpret_cast<__gm__ uint8_t*>(args[1]) + 64, SINGLE_CACHE_LINE);
    dcci(reinterpret_cast<__gm__ uint8_t*>(args[3]) + 64, SINGLE_CACHE_LINE);
    dcci(reinterpret_cast<__gm__ uint8_t*>(args[9]) + 64, SINGLE_CACHE_LINE);
    dcci(reinterpret_cast<__gm__ uint8_t*>(args[15]) + 64, SINGLE_CACHE_LINE);
    dcci(reinterpret_cast<__gm__ uint8_t*>(args[16]) + 64, SINGLE_CACHE_LINE);
}

template <uint64_t workspace_bytes>
__aicore__ inline void validate_shapes(__gm__ int64_t* args)
{
    if (dimension(args, 0, 0) != MOE_TOKENS || dimension(args, 0, 1) != MOE_TOPK ||
        dimension(args, 1, 0) != MOE_TOKENS || dimension(args, 1, 1) != MOE_HIDDEN ||
        dimension(args, 3, 0) != MOE_EXPERTS ||
        dimension(args, 9, 0) != MOE_HIDDEN || dimension(args, 9, 1) != MOE_INTER ||
        dimension(args, 15, 0) != MOE_TOKENS || dimension(args, 15, 1) != MOE_HIDDEN ||
        dimension(args, 16, 0) < static_cast<int64_t>(workspace_bytes)) trap();
}

template <bool kFullTopology>
__aicore__ inline void run(__gm__ int64_t* args, deep_gemm::CoreTopology<kFullTopology> topology)
{
    constexpr auto& workspace = deep_gemm::mega_moe::layout::kWorkspace<
        MOE_EP, MOE_EP * MOE_EXPERTS, 1, MOE_TOKENS,
        MOE_TOPK, MOE_HIDDEN, MOE_INTER>;
    constexpr uint64_t workspace_bytes = workspace.get_num_bytes();
    const auto sym = [&]() __aicore__ {
        if constexpr (MOE_TOKENS <= 80) {
            acquire_metadata(args);
            const auto acquired = bind_pypto_sym_buffer<MOE_EP, workspace_bytes>(args, 16, 18);
            validate_shapes<workspace_bytes>(args);
            return acquired;
        } else {
            validate_shapes<workspace_bytes>(args);
            return bind_pypto_sym_buffer<MOE_EP, workspace_bytes>(args, 16, 18);
        }
    }();
    const uint32_t core = get_block_idx(args);
#ifdef __DAV_VEC__
    const uint32_t lane = get_sub_block_id(args);
#else
    const uint32_t lane = 0;
#endif
    const int32_t requested = static_cast<int32_t>(args[17]);
    const uint32_t active = requested < 0 ? 0 : (requested > MOE_TOKENS ? MOE_TOKENS : requested);
    auto* output = data<bfloat16_t>(args, 15);
    // Both asc_init and the Cube MAD direction update CTRL within this call.
    const uint64_t saved_ctrl = get_ctrl();
    asc_init();
    v41_official_input::publish<workspace>(
        sym.template get_base_ptr<__gm__ uint8_t*>(), data<bfloat16_t>(args, 1),
        output, active, 2 * core + lane, topology);
    v41_official_input::publication_fence(lane);
    const auto weights = deep_gemm::mega_moe::layout::make_original_weights(
        data<uint8_t>(args, 3), data<uint8_t>(args, 4), data<uint8_t>(args, 5), data<uint8_t>(args, 6),
        data<uint8_t>(args, 7), data<uint8_t>(args, 8), data<uint8_t>(args, 9), data<uint8_t>(args, 10),
        data<uint8_t>(args, 11), data<uint8_t>(args, 12), data<uint8_t>(args, 13), data<uint8_t>(args, 14));
    deep_gemm::mega_moe::mega_moe_device<MOE_EP, MOE_EP * MOE_EXPERTS, 1,
        MOE_TOKENS, MOE_TOPK, MOE_HIDDEN, MOE_INTER>(
        output, nullptr, weights, active, 10.0f, sym, core, lane, topology, data<uint32_t>(args, 0), data<uint32_t>(args, 2));
    // The core drains its event credits and all pipes before returning. Keep
    // the PTO unit-repeat DMA convention and restore the caller's CTRL.
#ifdef __DAV_VEC__
    constexpr uint64_t unit_loop = (uint64_t{1} << 21) | 1;
    set_loop_size_ubtoout(unit_loop);
    set_loop_size_outtoub(unit_loop);
#endif
    set_ctrl(saved_ctrl);
}

} // namespace v41_official_entry

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args)
{
    const uint32_t cores = get_block_num(args);
    if (cores == 0 || cores > deep_gemm::kNumAICores) trap();
    // Retain compile-time scheduling on a full cohort; smaller cohorts share
    // the same pipeline with runtime work strides and completion counts.
    if (cores == deep_gemm::kNumAICores)
        v41_official_entry::run(args, deep_gemm::CoreTopology<true>{cores});
    else
        v41_official_entry::run(args, deep_gemm::CoreTopology<false>{cores});
}
