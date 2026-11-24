// Copyright (c) PyPTO Contributors.
// SPDX-License-Identifier: CANN-Open-Software-License-2.0 AND MIT
// DeepGEMM-Ascend: 3a35326271b069b8b5bf1a754435acd6cc91575b
// DeepSelect: bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18
// Third-party portions are distributed under the following license:
/*
MIT License

Copyright (c) 2026 DeepSeek (DeepGEMM-Ascend)
Copyright (c) 2025 DeepSeek (DeepSelect)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifdef __cpp_concepts
#undef __cpp_concepts
#endif
#include <utils/debug/asc_assert.h>
#if defined(__DAV_VEC__)
#include <adv_api/sort/sort.h>
#endif
#include "tensor.h"
#include "intrinsic.h"

namespace {
enum Stage : int64_t { kMetadata = 0, kAttention = 1, kScore = 2, kPool = 3, kScoreCandidates = 4 };

template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int slot) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[slot]);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}
}

#include <c_api/asc_simd.h>
#if defined(__DAV_VEC__)
// Parse CANN hardware type aliases before PTO imports its A5 types into global lookup.
#include <adv_api/sort/sort.h>
#include <kernel_operator.h>
#endif
// DeepGEMM-Ascend: common.hpp
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

// Ascend 950DT core topology
inline constexpr uint32_t kNumAICores = 32;
inline constexpr uint32_t kNumAIVs = 2 * kNumAICores;

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

// DeepGEMM-Ascend: ascend/tile_ptr.hpp
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

// DeepGEMM-Ascend: ascend/copy.hpp
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

// DeepGEMM-Ascend: ascend/sync.hpp
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

#include <pto/pto-inst.hpp>
// DeepGEMM-Ascend: scheduler/mqa_logits.hpp
namespace deep_gemm {

template <uint32_t kNumQTokensPerBlock, uint32_t kSplitKV, uint32_t kNumCores>
struct MQALogitsScheduler {
    uint32_t current_q_block_idx;
    uint32_t num_q_blocks;
    uint32_t num_q_tokens;
    uint32_t num_kv_tokens;
    uint32_t block_kv_end;
    __gm__ uint32_t* cu_seq_len_k_start;
    __gm__ uint32_t* cu_seq_len_k_end;
    uint32_t num_heads;

    __aicore__ MQALogitsScheduler(
        uint32_t core_id, uint32_t num_q_tokens, uint32_t num_kv_tokens,
        __gm__ uint32_t* cu_seq_len_k_start, __gm__ uint32_t* cu_seq_len_k_end,
        uint32_t num_heads
    ):
        current_q_block_idx(core_id),
        num_q_blocks(ceil_div(num_q_tokens, kNumQTokensPerBlock)),
        num_q_tokens(num_q_tokens),
        num_kv_tokens(num_kv_tokens),
        cu_seq_len_k_start(cu_seq_len_k_start),
        cu_seq_len_k_end(cu_seq_len_k_end),
        num_heads(num_heads) {}

    __aicore__ bool next_q_block(uint32_t& q_block_idx, uint32_t& kv_token_base, uint32_t& num_kv_splits) {
        if (current_q_block_idx >= num_q_blocks)
            return false;

        q_block_idx = current_q_block_idx;
        current_q_block_idx += kNumCores;

        uint32_t start = 0xffffffffu;
        uint32_t end = 0;
        const uint32_t q_base = q_block_idx * kNumQTokensPerBlock;
        for (uint32_t q_idx = 0; q_idx < kNumQTokensPerBlock; ++q_idx) {
            const uint32_t row_idx = min(q_base + q_idx, num_q_tokens - 1);
            const uint32_t row_start = min(cu_seq_len_k_start[row_idx], num_kv_tokens);
            const uint32_t row_end = min(cu_seq_len_k_end[row_idx], num_kv_tokens);
            start = min(start, row_start);
            end = max(end, row_end);
        }
        block_kv_end = end;
        kv_token_base = start / 8 * 8;
        num_kv_splits = ceil_div(end - kv_token_base, kSplitKV);
        return true;
    }

    __aicore__ uint32_t get_q_base(uint32_t q_block_idx) const {
        return q_block_idx * kNumQTokensPerBlock;
    }
    __aicore__ uint32_t get_num_q_tokens_in_block(uint32_t q_block_idx) const {
        return min(kNumQTokensPerBlock, num_q_tokens - get_q_base(q_block_idx));
    }
    // NOTES: SF strides are in int16_t elements, not bytes.
    __aicore__ uint32_t get_q_sf_stride() const {
        return num_q_tokens * num_heads;
    }
    __aicore__ uint32_t get_kv_token_offset(uint32_t kv_token_base, uint32_t split_idx) const {
        return kv_token_base + split_idx * kSplitKV;
    }
    __aicore__ uint32_t get_num_kv_block_tokens(uint32_t kv_token_base, uint32_t split_idx) const {
        const uint32_t offset = get_kv_token_offset(kv_token_base, split_idx);
        return offset < block_kv_end ? min(kSplitKV, block_kv_end - offset) : 0;
    }
    __aicore__ uint32_t get_kv_sf_stride() const {
        return num_kv_tokens;
    }
};

} // namespace deep_gemm

// DeepGEMM-Ascend: scheduler/paged_mqa_logits.hpp
namespace deep_gemm {

struct PagedMQARequestInfo {
    uint32_t q_token_start;
    uint32_t num_q_tokens;
    uint32_t context_len;
    uint32_t num_q_blocks;
};

__host_aicore__ constexpr uint32_t mqa_request_context_len(uint32_t q, uint32_t count, __gm__ uint32_t* context_lens) {
    uint32_t context_len = 0;
    for (uint32_t row = q; row < q + count; ++row)
        context_len = context_len > context_lens[row] ? context_len : context_lens[row];
    return context_len;
}

template <uint32_t kNumQTokensPerBlock>
__aicore__ inline PagedMQARequestInfo make_paged_mqa_request(uint32_t q, uint32_t num_q_tokens,
                                                             __gm__ uint32_t* context_lens,
                                                             __gm__ uint32_t* indices) {
    uint32_t count = 1;
    while (q + count < num_q_tokens && indices[q + count] == indices[q])
        ++count;
    return {
        .q_token_start=q,
        .num_q_tokens=count,
        .context_len=mqa_request_context_len(q, count, context_lens),
        .num_q_blocks=ceil_div(count, kNumQTokensPerBlock),
    };
}

template <uint32_t kNumQTokensPerBlock, uint32_t kSplitKV, uint32_t kPageSizeValue, uint32_t kNumHeads>
struct PagedMQALogitsScheduler {
    static constexpr uint32_t kPageSize = kPageSizeValue;

    uint32_t num_q_tokens;
    __gm__ uint32_t* context_lens;
    __gm__ uint32_t* indices;
    __gm__ uint32_t* block_table;
    uint32_t block_table_row_stride;
    uint64_t kv_page_stride_bytes;
    uint64_t sf_page_stride_bytes;
    uint32_t q_token_idx;
    uint32_t kv_split_idx;
    uint32_t end_q_token_idx;
    uint32_t end_kv_split_idx;
    PagedMQARequestInfo request;
    uint32_t current_q_block_idx = 0;

    __aicore__ PagedMQALogitsScheduler(
        uint32_t core_id, uint32_t num_q_tokens,
        __gm__ uint32_t* context_lens, __gm__ uint32_t* indices,
        __gm__ uint32_t* metadata, __gm__ uint32_t* block_table,
        uint32_t block_table_row_stride, uint64_t kv_page_stride_bytes, uint64_t sf_page_stride_bytes
    ):
        num_q_tokens(num_q_tokens),
        context_lens(context_lens),
        indices(indices),
        block_table(block_table),
        block_table_row_stride(block_table_row_stride),
        kv_page_stride_bytes(kv_page_stride_bytes),
        sf_page_stride_bytes(sf_page_stride_bytes) {
        const uint32_t metadata_idx = core_id * 2;
        q_token_idx = metadata[metadata_idx];
        kv_split_idx = metadata[metadata_idx + 1];
        end_q_token_idx = metadata[metadata_idx + 2];
        end_kv_split_idx = metadata[metadata_idx + 3];
        if (!done())
            request = make_paged_mqa_request<kNumQTokensPerBlock>(q_token_idx, num_q_tokens, context_lens, indices);
    }

    __aicore__ bool done() const {
        return q_token_idx >= num_q_tokens ||
               (q_token_idx == end_q_token_idx && kv_split_idx >= end_kv_split_idx);
    }
    __aicore__ bool next_q_block(uint32_t& q_block_idx, uint32_t& kv_token_base, uint32_t& num_kv_splits) {
        while (!done()) {
            const uint32_t split_end = request.q_token_start == end_q_token_idx ? end_kv_split_idx : ceil_div(request.context_len, kSplitKV);
            if (current_q_block_idx == request.num_q_blocks) {
                current_q_block_idx = 0;
                kv_split_idx = split_end;
            }
            if (kv_split_idx < split_end) {
                q_block_idx = current_q_block_idx++;
                // Scheduler state stays in split units; the shared kernel consumes token units.
                kv_token_base = kv_split_idx * kSplitKV;
                num_kv_splits = split_end - kv_split_idx;
                return true;
            }
            if (request.q_token_start == end_q_token_idx)
                break;
            q_token_idx = request.q_token_start + request.num_q_tokens;
            kv_split_idx = 0;
            if (!done())
                request = make_paged_mqa_request<kNumQTokensPerBlock>(q_token_idx, num_q_tokens, context_lens, indices);
        }
        return false;
    }

    __aicore__ uint32_t get_q_base(uint32_t q_block_idx) const {
        return request.q_token_start + q_block_idx * kNumQTokensPerBlock;
    }
    __aicore__ uint32_t get_num_q_tokens_in_block(uint32_t q_block_idx) const {
        return min(kNumQTokensPerBlock, request.num_q_tokens - q_block_idx * kNumQTokensPerBlock);
    }
    __aicore__ bool has_multiple_q_blocks() const {
        return request.num_q_tokens > kNumQTokensPerBlock;
    }
    __aicore__ uint32_t get_kv_token_offset(uint32_t kv_token_base, uint32_t split_idx) const {
        return kv_token_base + split_idx * kSplitKV;
    }
    __aicore__ uint32_t get_num_kv_block_tokens(uint32_t kv_token_base, uint32_t split_idx) const {
        const uint32_t offset = get_kv_token_offset(kv_token_base, split_idx);
        return offset < request.context_len ? min(kSplitKV, request.context_len - offset) : 0;
    }
    __aicore__ uint32_t get_physical_page(uint32_t kv_token_base, uint32_t split_idx, uint32_t compact_page) const {
        const uint32_t logical_page_idx = get_kv_token_offset(kv_token_base, split_idx) / kPageSize + compact_page;
        return block_table[static_cast<uint64_t>(indices[request.q_token_start]) * block_table_row_stride +
                           logical_page_idx];
    }
    // Adjacent logical pages start at the smaller physical page; AIV restores reversed pairs.
    __aicore__ uint32_t get_physical_page_batch(uint32_t kv_token_base, uint32_t split_idx, uint32_t compact_page,
                                                uint32_t num_pages, uint32_t& first_page,
                                                uint32_t& page_gap) const {
        const uint32_t num_batch = min(2u, num_pages - compact_page);
        const uint32_t page0 = get_physical_page(kv_token_base, split_idx, compact_page);
        const uint32_t page1 = num_batch == 2 ? get_physical_page(kv_token_base, split_idx, compact_page + 1) : page0;
        first_page = min(page0, page1);
        page_gap = page0 > page1 ? page0 - page1 : page1 - page0;
        return num_batch;
    }
    __aicore__ uint64_t get_sf_page_offset_bytes(uint32_t physical_page) const {
        return physical_page * sf_page_stride_bytes;
    }
    __aicore__ uint64_t get_kv_page_offset_bytes(uint32_t physical_page) const {
        return physical_page * kv_page_stride_bytes;
    }
};

} // namespace deep_gemm

// DeepGEMM-Ascend: mqa_logits.hpp
#include <pto/pto-inst.hpp>

namespace deep_gemm {
using namespace __cce_simd;

// Normalize raw BF16 model weights with the original FP32 multiply and BF16 rounding.
template <uint32_t kMaxElems>
__aicore__ inline void mqa_logits_normalize_weights(
    __ubuf__ bfloat16_t* weights, __ubuf__ float* scratch, uint32_t num_elems
) {
#if defined(__DAV_VEC__)
    pto::Tile<pto::TileType::Vec, bfloat16_t, 1, kMaxElems, pto::BLayout::RowMajor, 1, pto::DYNAMIC> raw(num_elems);
    pto::Tile<pto::TileType::Vec, float, 1, kMaxElems, pto::BLayout::RowMajor, 1, pto::DYNAMIC> scaled(num_elems);
    pto::TASSIGN(raw, reinterpret_cast<uintptr_t>(weights));
    pto::TASSIGN(scaled, reinterpret_cast<uintptr_t>(scratch));
    pto::TCVT(scaled, raw, pto::RoundMode::CAST_RINT);
    pto::TMULS(scaled, scaled, 0.015625f);
    pto::TCVT(raw, scaled, pto::RoundMode::CAST_RINT);
#endif
}

// Pack adjacent scale pairs into the 16-token fragments consumed by L0B_MX.
__simd_vf__ inline void vf_mqa_logits_pack_scales(__ubuf__ int16_t* scales, uint32_t num_elems) {
    vector_bool mask = asc_create_mask_b16(PAT_ALL);
    vector_int16_t lane, low, high, pair, bits;
    asc_arange(lane, static_cast<int16_t>(0));
    asc_duplicate_scalar(bits, static_cast<int16_t>(FRAC_MN - 1));
    asc_and(low, lane, bits, mask);
    asc_shiftleft_scalar(low, low, 1, mask);
    asc_duplicate_scalar(bits, static_cast<int16_t>(FRAC_MN));
    asc_and(pair, lane, bits, mask);
    asc_shiftright_scalar(pair, pair, __builtin_ctz(FRAC_MN), mask);
    asc_duplicate_scalar(bits, static_cast<int16_t>(-2 * FRAC_MN));
    asc_and(high, lane, bits, mask);
    asc_add(low, low, pair, mask);
    asc_add(low, low, high, mask);
    for (uint32_t offset = 0; offset < num_elems; offset += asc_get_vf_len() / sizeof(int16_t)) {
        vector_int16_t src, dst;
        asc_loadalign(src, scales + offset);
        asc_gather(dst, src, reinterpret_cast<vector_uint16_t&>(low));
        asc_storealign(scales + offset, dst, mask);
    }
}

// Ascend 950 MTE3 requires a 32B-aligned UB source; GM address and length are unrestricted.
constexpr uint32_t kLogitsAlignElems = 32 / sizeof(bfloat16_t);

// Stage an unaligned row head through aligned UB scratch before MTE3.
__simd_vf__ inline void vf_mqa_logits_stage_head(__ubuf__ bfloat16_t* dst, __ubuf__ bfloat16_t* src) {
    vector_load_unalign desc;
    asc_loadunalign_pre(desc, src);
    vector_bfloat16_t v;
    asc_loadunalign(v, desc, src);
    asc_storealign(dst, v, asc_create_mask_b16(PAT_ALL));
}

template <uint32_t kScratchEventIdx>
__aicore__ inline void mqa_logits_store(
    __gm__ bfloat16_t* dst, __ubuf__ bfloat16_t* src,
    __ubuf__ bfloat16_t* scratch,
    uint32_t num_elems
) {
    const uint32_t src_misalignment = (reinterpret_cast<uintptr_t>(src) / sizeof(bfloat16_t)) & (kLogitsAlignElems - 1);
    const uint32_t num_prefix_elems = min(num_elems, (kLogitsAlignElems - src_misalignment) & (kLogitsAlignElems - 1));
    if (num_prefix_elems) {
        asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(kScratchEventIdx));
        vf_mqa_logits_stage_head(scratch, src);
        asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(kScratchEventIdx));
        asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(kScratchEventIdx));
        const uint32_t num_prefix_bytes = num_prefix_elems * sizeof(bfloat16_t);
        asc_copy_ub2gm_align(dst, scratch, 1, num_prefix_bytes, asc_store_l2_cache_mode::NOTALLOC_CLEAN,
                             num_prefix_bytes, num_prefix_bytes);
        asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(kScratchEventIdx));
    }
    if (num_prefix_elems < num_elems) {
        const uint32_t num_tail_bytes = (num_elems - num_prefix_elems) * sizeof(bfloat16_t);
        asc_copy_ub2gm_align(dst + num_prefix_elems, src + num_prefix_elems, 1, num_tail_bytes,
                             asc_store_l2_cache_mode::NOTALLOC_CLEAN, num_tail_bytes, num_tail_bytes);
    }
}

// Restore logical page order after gathering each pair in physical-address order.
// kPageSize=0 is continuous, 64 swaps two halves in one vector, and 128 swaps
// the destinations of two adjacent vectors.
template <uint32_t kPageSize>
__simd_callee__ inline void vf_mqa_logits_store_bf16(
    __ubuf__ bfloat16_t* dst, vector_bfloat16_t src,
    uint32_t reversed_pairs, uint32_t vec_idx
) {
    constexpr uint32_t kNumElemsPerVecReg = asc_get_vf_len() / sizeof(bfloat16_t);
    if constexpr (kPageSize == 128) {
        const uint32_t reversed = (reversed_pairs >> (vec_idx >> 1)) & 1u;
        dst += (vec_idx ^ reversed) * kNumElemsPerVecReg;
    } else {
        dst += vec_idx * kNumElemsPerVecReg;
    }
    if constexpr (kPageSize == 64) {
        const bool reversed = (reversed_pairs >> vec_idx) & 1u;
        if (reversed) {
            constexpr uint32_t kPageElems = kNumElemsPerVecReg / 2;
            vector_bool low_half = asc_create_mask_b16(PAT_VL64);
            vector_bool high_half;
            asc_not(high_half, low_half, asc_create_mask_b16(PAT_ALL));
            asc_storealign(dst + kPageElems, src, low_half);
            asc_storealign(dst - kPageElems, src, high_half);
            return;
        }
    }
    static_assert(kPageSize == 0 || kPageSize == 64 || kPageSize == 128);
    asc_storealign(dst, src, asc_create_mask_b16(PAT_ALL));
}

template <uint32_t kPageSize, uint32_t kNumElems, uint32_t kNumHeads, uint32_t kSplitKV, uint32_t kNumQueries>
__simd_vf__ inline void vf_mqa_logits_reduce(
    __ubuf__ bfloat16_t* score, __ubuf__ bfloat16_t* weights,
    __ubuf__ bfloat16_t* line,
    uint32_t q_local, uint32_t q_idx,
    uint32_t reversed_pairs
) {
    constexpr uint32_t kNumElemsPerVecReg = asc_get_vf_len() / sizeof(bfloat16_t);
    constexpr uint32_t kNumVecRegs = kNumElems / kNumElemsPerVecReg;
    static_assert(kNumElems % kNumElemsPerVecReg == 0 && kNumVecRegs >= 1 && kNumVecRegs <= 4);
    static_assert((kNumQueries == 1 || kNumQueries == 2) && kSplitKV == 512);
#if defined(PREFILL_FP32_HEAD_SUM)
    if constexpr (kPageSize == 128) {
        // Match the prefill expression: each product rounds to BF16, while
        // the head reduction accumulates in FP32 and rounds once at the end.
        constexpr uint32_t lanes = asc_get_vf_len() / sizeof(float);
        constexpr uint32_t pieces = kNumElems / lanes;
        const auto full = pset_b32(PAT_ALL);
        const auto full16 = pset_b16(PAT_ALL);
        vector_float sums[kNumQueries][pieces];
        #pragma unroll
        for (uint32_t q = 0; q < kNumQueries; ++q) {
            #pragma unroll
            for (uint32_t p = 0; p < pieces; ++p) {
                vdup(sums[q][p], 0.0f, full, MODE_ZEROING);
            }
        }
        #pragma unroll 1
        for (uint16_t h = 0; h < kNumHeads; ++h) {
            #pragma unroll
            for (uint32_t q = 0; q < kNumQueries; ++q) {
                vector_bfloat16_t weight;
                vlds(weight, weights, (q_idx + q) * kNumHeads + h, BRC_B16);
                const uint32_t base = ((q_local + q) * kNumHeads + h) * kNumElems;
                #pragma unroll
                for (uint32_t p = 0; p < pieces; ++p) {
                    vector_bfloat16_t value, product;
                    vector_float wide;
                    vlds(value, score, base + p * lanes, UNPK_B16);
                    vmul(product, value, weight, full16, MODE_ZEROING);
                    vcvt(wide, product, full, PART_EVEN);
                    vadd(sums[q][p], sums[q][p], wide, full, MODE_ZEROING);
                }
            }
        }
        #pragma unroll
        for (uint32_t q = 0; q < kNumQueries; ++q) {
            #pragma unroll
            for (uint32_t p = 0; p < pieces; ++p) {
                const uint32_t page = p / 2;
                const uint32_t reverse = (reversed_pairs >> (page / 2)) & 1u;
                const uint32_t offset = (page ^ reverse) * kNumElemsPerVecReg + (p % 2) * lanes;
                vector_bfloat16_t rounded;
                vcvt(rounded, sums[q][p], full, ROUND_R, RS_DISABLE, PART_EVEN);
                vsts(rounded, line, q * kSplitKV + offset, PK_B32, full);
            }
        }
        return;
    }
#endif
    vector_bool mask = asc_create_mask_b16(PAT_ALL);
    vector_bfloat16_t sum[kNumVecRegs][kNumQueries];
    #pragma unroll
    for (uint32_t vec_idx = 0; vec_idx < kNumVecRegs; ++vec_idx) {
        #pragma unroll
        for (uint32_t query_idx = 0; query_idx < kNumQueries; ++query_idx)
            asc_duplicate_scalar(sum[vec_idx][query_idx], static_cast<bfloat16_t>(0), mask);
    }

    __ubuf__ bfloat16_t* weight_ptr[kNumQueries];
    __ubuf__ bfloat16_t* score_ptr[kNumQueries];
    #pragma unroll
    for (uint32_t query_idx = 0; query_idx < kNumQueries; ++query_idx) {
        weight_ptr[query_idx] = weights + (q_idx + query_idx) * kNumHeads;
        score_ptr[query_idx] = score + (q_local + query_idx) * kNumHeads * kNumElems;
    }
    #pragma unroll 1
    for (uint32_t h = 0; h < kNumHeads; ++h) {
        vector_bfloat16_t weight[kNumQueries];
        #pragma unroll
        for (uint32_t query_idx = 0; query_idx < kNumQueries; ++query_idx)
            asc_loadalign_brc_postupdate(weight[query_idx], weight_ptr[query_idx], 1);
        #pragma unroll
        for (uint32_t vec_idx = 0; vec_idx < kNumVecRegs; ++vec_idx) {
            #pragma unroll
            for (uint32_t query_idx = 0; query_idx < kNumQueries; ++query_idx) {
                vector_bfloat16_t value;
                asc_loadalign_postupdate(value, score_ptr[query_idx], kNumElemsPerVecReg);
                asc_mula(sum[vec_idx][query_idx], value, weight[query_idx], mask);
            }
        }
    }

    #pragma unroll
    for (uint32_t vec_idx = 0; vec_idx < kNumVecRegs; ++vec_idx) {
        #pragma unroll
        for (uint32_t query_idx = 0; query_idx < kNumQueries; ++query_idx)
            vf_mqa_logits_store_bf16<kPageSize>(line + query_idx * kSplitKV, sum[vec_idx][query_idx], reversed_pairs, vec_idx);
    }
}

template <uint32_t kPageSize, uint32_t kNumQTokensPerTile, uint32_t kNumHeads, uint32_t kSplitKV>
__aicore__ inline uint32_t mqa_logits_reduce_queries(
    __ubuf__ bfloat16_t* score, __ubuf__ bfloat16_t* weight,
    __ubuf__ bfloat16_t* line,
    uint32_t q_local, uint32_t q_idx,
    uint32_t num_q_tokens_in_block, uint32_t issued_n,
    uint32_t reversed_pairs
) {
    uint32_t num_reduced_queries = 1;
    const auto reduce = [&](auto num_elems) __aicore__ {
        constexpr uint32_t kNumElems = decltype(num_elems)::value;
        if constexpr (kNumQTokensPerTile >= 2) {
            if (q_local + 1 < kNumQTokensPerTile && q_idx + 1 < num_q_tokens_in_block) {
                vf_mqa_logits_reduce<kPageSize, kNumElems, kNumHeads, kSplitKV, 2>(score, weight, line, q_local, q_idx,
                                                                                   reversed_pairs);
                num_reduced_queries = 2;
                return;
            }
        }
        vf_mqa_logits_reduce<kPageSize, kNumElems, kNumHeads, kSplitKV, 1>(score, weight, line, q_local, q_idx, reversed_pairs);
    };
    constexpr uint32_t kNumElemsPerVecReg = asc_get_vf_len() / sizeof(bfloat16_t);
    switch (issued_n / kNumElemsPerVecReg) {
        case 1: reduce(std::integral_constant<uint32_t, kNumElemsPerVecReg>{}); break;
        case 2: reduce(std::integral_constant<uint32_t, 2 * kNumElemsPerVecReg>{}); break;
        case 3: reduce(std::integral_constant<uint32_t, 3 * kNumElemsPerVecReg>{}); break;
        case 4: reduce(std::integral_constant<uint32_t, 4 * kNumElemsPerVecReg>{}); break;
        default: ascendc_assert(false, "Unsupported issued_n for MQA logits reduction");
    }
    return num_reduced_queries;
}

// Reduce consecutive logical BF16 tokens into eight-token candidate blocks.
__simd_vf__ inline void vf_mqa_logits_block_max(
    __ubuf__ bfloat16_t* dst, __ubuf__ bfloat16_t* src, __ubuf__ float* scratch,
    uint32_t valid_tokens, uint32_t block_begin, uint32_t last_block
) {
    mem_bar(VST_VLD);
    const auto full = pset_b32(PAT_ALL);
    uint32_t reduced_count = 8;
    const auto reduced = plt_b32(reduced_count, POST_UPDATE);
    vector_uint32_t lane, columns, blocks;
    vector_float value, maximum, negative_inf, positive_inf, quiet_nan;
    vector_bfloat16_t rounded, packed;
    vector_bool valid, newest, not_nan, block_has_nan;
    vci(reinterpret_cast<vector_int32_t&>(lane), 0, INC_ORDER);
    vdup(negative_inf, -__builtin_inff(), full, MODE_ZEROING);
    vdup(positive_inf, __builtin_inff(), full, MODE_ZEROING);
    vdup(quiet_nan, __builtin_nanf(""), reduced, MODE_ZEROING);
    for (uint16_t chunk = 0; chunk < (valid_tokens + 63) / 64; ++chunk) {
        vlds(rounded, src, chunk * 64, UNPK_B16);
        vcvt(value, rounded, full, PART_EVEN);
        vadds(columns, lane, uint32_t(chunk * 64), full, MODE_ZEROING);
        vcmps_lt(valid, columns, valid_tokens, full);
        vsel(value, value, negative_inf, valid);
        // Reserve +inf for a visible NaN during the eight-token max.
        vcmp_eq(not_nan, value, value, full);
        vmins(value, value, 3.3895313892515355e38f, full, MODE_ZEROING);
        vsel(value, value, positive_inf, not_nan);
        vcgmax(maximum, value, full);
        vcmp_eq(block_has_nan, maximum, positive_inf, reduced);
        vadds(blocks, lane, block_begin + chunk * 8, reduced, MODE_ZEROING);
        vcmps_eq(newest, blocks, last_block, reduced);
        vsel(maximum, positive_inf, maximum, newest);
        vsel(maximum, quiet_nan, maximum, block_has_nan);
        vsts(maximum, scratch, chunk * 8, NORM_B32, reduced);
    }
    mem_bar(VST_VLD);
    vlds(maximum, scratch, 0, NORM);
    vcvt(packed, maximum, full, ROUND_R, RS_DISABLE, PART_EVEN);
    vsts(packed, dst, 0, PK_B32, full);
}

template <
    bool kIsFP4,
    uint32_t kNumHeads, uint32_t kHeadDim,
    uint32_t kSplitKV, uint32_t kMadM, uint32_t kNumCDStages,
    uint32_t kNumQStages, uint32_t kNumKVStages, uint32_t kNumAccumStages,
    uint32_t kNumLogitsStages,
    uint32_t kPageSize,
    bool kUseAscendKVLayout,
    bool kEmitBlockMax,
    typename MakeScheduler
>
__aicore__ inline void mqa_logits_core_impl(
    gm_ptr<uint8_t, Major::K> q,
    gm_ptr<uint8_t, Major::K> kv,
    __gm__ int16_t* sf_q,
    __gm__ int16_t* sf_kv,
    gm_ptr<bfloat16_t, Major::K> weights,
    gm_ptr<bfloat16_t, Major::K> logits,
    gm_ptr<bfloat16_t, Major::K> block_scores,
    const MakeScheduler& make_scheduler, uint32_t core_id, uint32_t subblock_id
) {
    using Scheduler = decltype(make_scheduler(core_id));
    constexpr bool kIsPaged = kPageSize != 0;
    static_assert(kNumKVStages % 2 == 0, "SF stages alternate between two AIVs");
    static_assert(kSplitKV == 512, "MQA logits kernel expects bf16 SPLIT_KV=512");
    static_assert(kNumHeads >= 4 && kNumHeads <= 64 && kNumHeads % 4 == 0,
                  "Number of heads must be a multiple of 4 in [4, 64]");
    static_assert(kHeadDim == 64 || kHeadDim == 128, "Unsupported head dimension");
    using qk_dtype_t = std::conditional_t<kIsFP4, float4_e2m1x2_t, float8_e4m3_t>;

    constexpr uint32_t kNumVecElems = asc_get_vf_len() / sizeof(bfloat16_t);
    constexpr uint32_t MAD_M = kMadM;
    constexpr uint32_t MAD_M_PADDED = aligned(MAD_M, FRAC_MN);
    constexpr uint32_t MAD_N = kSplitKV;
    constexpr uint32_t MAD_K = kHeadDim;
    // Fixed 8-tile specialization: L0A capacity cap (8 * MAD_M_PADDED * D * 1B <= 64KB); runtime trims usage
    constexpr uint32_t kNumQHTilesPerBlock = 8;
    constexpr uint32_t kNumQHRowsPerBlock = kNumQHTilesPerBlock * MAD_M;
    constexpr uint32_t kNumPaddedQHRowsPerBlock = kNumQHTilesPerBlock * MAD_M_PADDED;
    constexpr uint32_t kNumQTokensPerBlock = kNumQHRowsPerBlock / kNumHeads;
    constexpr uint32_t kNumQTokensPerTile = MAD_M / kNumHeads;
    constexpr uint32_t kSFPairs = kHeadDim / MX_SF_DIVISOR;
    static_assert(kNumQHRowsPerBlock % kNumHeads == 0, "QH block must contain full query/head groups");
    static_assert(kNumCDStages == 2 || kNumCDStages == 4, "MQA logits currently supports 2 or 4 CD stages");
    static_assert(MAD_M % kNumHeads == 0, "MAD_M must contain full query/head groups");
    static_assert(kNumQTokensPerTile > 0, "AIV score slice must contain at least one query");
    static_assert(kSplitKV != 512 || kNumCDStages == 2, "SPLIT_KV=512 expects 2 CD stages to fit L0C");

    constexpr l1_ptr<qk_dtype_t, Major::K> l1q(kNumPaddedQHRowsPerBlock, kHeadDim, 0);
    constexpr l1_ptr<qk_dtype_t, Major::K> l1kv(kSplitKV, kHeadDim, l1q.offset(kNumQStages));
    constexpr l1_ptr<int16_t> l1_sf_q(kNumPaddedQHRowsPerBlock, kSFPairs, l1kv.offset(kNumKVStages));
    constexpr l1_ptr<int16_t> l1_sf_kv(kSplitKV, kSFPairs, l1_sf_q.offset(kNumQStages));
    constexpr l1_ptr<int16_t> l1_sf_q_tiles(MAD_M_PADDED, kSFPairs, l1_sf_q.addr);
    constexpr l0a_ptr<qk_dtype_t> l0a(MAD_M_PADDED, MAD_K);
    constexpr l0b_ptr<qk_dtype_t> l0b(MAD_N, MAD_K);
    constexpr l0c_ptr<float> l0c(MAD_M_PADDED, MAD_N);

    constexpr uint32_t kNumWeightStages = 2;
    constexpr ub_ptr<bfloat16_t> ub_accum(MAD_M_PADDED, MAD_N, 0);
    constexpr ub_ptr<bfloat16_t> ub_weights(kNumQTokensPerBlock, kNumHeads, ub_accum.offset(kNumAccumStages));
    constexpr ub_ptr<bfloat16_t> ub_logits(kNumQTokensPerTile, MAD_N, ub_weights.offset(kNumWeightStages));

    constexpr ub_ptr<bfloat16_t> ub_logits_head(1, asc_get_vf_len() / sizeof(bfloat16_t),
                                                    ub_logits.offset(kNumLogitsStages));
    // One row per Q block; the 8-element pad keeps each buffer 32B-aligned
    // (paged H=64 blocks hold a single token, so the raw count may not).
    constexpr ub_ptr<uint32_t> ub_seq_start(1, aligned(kNumQTokensPerBlock, 8), ub_logits_head.offset(1));
    constexpr ub_ptr<uint32_t> ub_seq_end(1, aligned(kNumQTokensPerBlock, 8), ub_seq_start.offset(1));
    constexpr uint32_t kNumSFUBStages = kNumKVStages / 2;
    constexpr ub_ptr<int16_t> ub_sf_kv(kIsPaged ? kSplitKV : 0, kSFPairs, ub_seq_end.offset(1));
    constexpr ub_ptr<float> ub_weight_fp32(1, kNumQTokensPerBlock * kNumHeads, ub_sf_kv.offset(kNumSFUBStages));
    constexpr ub_ptr<bfloat16_t> ub_block_scores(kEmitBlockMax ? kNumQTokensPerTile : 0, kSplitKV / 8,
                                                ub_weight_fp32.offset(1));
    constexpr ub_ptr<float> ub_block_scratch(kEmitBlockMax ? 1 : 0, kSplitKV / 8,
                                            ub_block_scores.offset(kNumLogitsStages));

    static_assert(l1_sf_kv.offset(kNumKVStages) <= L1SizeBytes, "L1 overflow");
    static_assert(l0a.offset(kNumQHTilesPerBlock) <= L0ASizeBytes, "L0A overflow");
    static_assert(l0b.offset(1) <= L0BSizeBytes, "L0B overflow");
    static_assert(l0c.offset(kNumCDStages) <= L0CSizeBytes, "L0C overflow");
    static_assert(ub_logits.offset(kNumLogitsStages) <= UBSizeBytes, "UB overflow");
    static_assert(ub_logits_head.offset(1) <= UBSizeBytes, "UB overflow");
    static_assert(ub_seq_end.offset(1) <= UBSizeBytes, "UB overflow");
    static_assert(ub_sf_kv.offset(kNumSFUBStages) <= UBSizeBytes, "UB overflow");
    static_assert(ub_weight_fp32.offset(1) <= UBSizeBytes, "UB overflow");
    static_assert(ub_block_scores.offset(kNumLogitsStages) <= UBSizeBytes, "UB overflow");
    static_assert(ub_block_scratch.offset(1) <= UBSizeBytes, "UB overflow");
    static_assert(!kEmitBlockMax || kIsPaged, "Candidate blocks require paged logits");
    static_assert(!kIsPaged || kNumAccumStages + kNumKVStages <= 16, "Scale flag overflow");

    auto scheduler = make_scheduler(core_id);
    const gm_ptr<int16_t, Major::K> sf_q_t(kSFPairs, reinterpret_cast<uintptr_t>(sf_q));
    const gm_ptr<qk_dtype_t, Major::K> q_t(q.stride_outer, q.addr, q.stride_batch);
    const gm_ptr<qk_dtype_t, Major::K> kv_t(kv.stride_outer, kv.addr, kv.stride_batch);
    uint32_t q_block_idx, kv_token_base, num_kv_splits;
    if ASCEND_IS_AIC {
        asc_set_mmad_direction_n();

        constexpr uint32_t kQEventBase = 0;
        constexpr uint32_t kKVEventBase = kNumQStages;
        set_flags<PIPE_M, PIPE_MTE1, 2>();
        set_flags<PIPE_MTE1, PIPE_MTE2, kNumQStages, kQEventBase>();
        set_flags<PIPE_MTE1, PIPE_MTE2, kNumKVStages, kKVEventBase>();
        // Scale slots retain their AIV owner across query blocks.
        if constexpr (kIsPaged)
            for (uint32_t stage_idx = 0; stage_idx < kNumKVStages; ++stage_idx)
                set_intra_block(PIPE_MTE1, (stage_idx ^ 1u) & 1u, kNumAccumStages + stage_idx);

        uint32_t q_stage_idx = 0;
        uint32_t kv_stage_idx = 0;
        uint32_t cd_stage_idx = 0;
        while (scheduler.next_q_block(q_block_idx, kv_token_base, num_kv_splits)) {
            const uint32_t q_base = scheduler.get_q_base(q_block_idx);
            const uint32_t num_q_tokens_in_block = scheduler.get_num_q_tokens_in_block(q_block_idx);
            const uint32_t num_qh_rows_in_block = num_q_tokens_in_block * kNumHeads;
            const uint32_t num_qh_tiles_in_block = ceil_div(num_qh_rows_in_block, MAD_M);

            asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(kQEventBase + q_stage_idx));
            if constexpr (MAD_M == MAD_M_PADDED) {
                fill_mn_tail_block(l1q[q_stage_idx], num_qh_rows_in_block, aligned(num_qh_rows_in_block, FRAC_MN));
                copy_gm_to_l1(l1q[q_stage_idx], q_t.index(q_base * kNumHeads, 0), num_qh_rows_in_block, kHeadDim,
                              asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                copy_gm_to_l1_mx(l1_sf_q[q_stage_idx], sf_q_t.index(q_base * kNumHeads, 0), num_qh_rows_in_block,
                                 kSFPairs, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
            } else {
                const l1_ptr<qk_dtype_t> l1q_m_tiles(MAD_M_PADDED, get_frac_k<qk_dtype_t>(), l1q[q_stage_idx].addr);
                for (uint32_t qh_tile_idx = 0; qh_tile_idx < num_qh_tiles_in_block; ++qh_tile_idx) {
                    const uint32_t qh_base = qh_tile_idx * MAD_M;
                    const uint32_t num_m_rows = min(MAD_M, num_qh_rows_in_block - qh_base);
                    const uint32_t sf_q_tile_idx = q_stage_idx * kNumQHTilesPerBlock + qh_tile_idx;
                    const auto q_tile = l1q_m_tiles[qh_tile_idx].as_mad_aligned(kNumPaddedQHRowsPerBlock, kHeadDim);
                    const auto sf_q_tile = l1_sf_q_tiles[sf_q_tile_idx];
                    fill_mn_tail_block(q_tile, num_m_rows, MAD_M_PADDED);
                    copy_gm_to_l1(q_tile, q_t.index(q_base * kNumHeads + qh_base, 0), num_m_rows, kHeadDim,
                                  asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                    copy_gm_to_l1_mx(sf_q_tile, sf_q_t.index(q_base * kNumHeads + qh_base, 0), num_m_rows, kSFPairs,
                                     asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                }
            }
            asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(kQEventBase + q_stage_idx));

            asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(kQEventBase + q_stage_idx));
            asc_sync_wait(PIPE_M, PIPE_MTE1, EVENT_ID1);
            for (uint32_t qh_tile_idx = 0; qh_tile_idx < num_qh_tiles_in_block; ++qh_tile_idx) {
                const uint32_t qh_base = qh_tile_idx * MAD_M;
                const uint32_t qh_padded_base = qh_tile_idx * MAD_M_PADDED;
                const uint32_t num_m_rows = min(MAD_M, num_qh_rows_in_block - qh_base);
                const auto l0a_tile = l0a[qh_tile_idx].as_mad_aligned(num_m_rows, MAD_K);
                copy_l1_to_l0a(l0a_tile, l1q[q_stage_idx], qh_padded_base, 0);
                copy_l1_to_l0a_mx(l0a_tile, l1_sf_q[q_stage_idx], qh_padded_base, 0, MAD_K);
            }
            asc_sync_notify(PIPE_MTE1, PIPE_M, EVENT_ID1);

            // GM-backed tile check; the AIV branch defines the UB-backed twin
            const auto is_tile_active = [&](uint32_t q_tile_base, uint32_t kv_idx, uint32_t kv_end) __aicore__ -> bool {
                for (uint32_t q_idx = q_tile_base; q_idx < min(q_tile_base + kNumQTokensPerTile, num_q_tokens_in_block); ++q_idx) {
                    const uint32_t row = q_base + q_idx;
                    if constexpr (kIsPaged) {
                        if (kv_idx < scheduler.context_lens[row])
                            return true;
                    } else if (kv_idx < scheduler.cu_seq_len_k_end[row] &&
                               scheduler.cu_seq_len_k_start[row] < kv_end) {
                        return true;
                    }
                }
                return false;
            };

            asc_sync_wait(PIPE_MTE1, PIPE_M, EVENT_ID1);
            for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++kv_split_idx) {
                const uint32_t kv_idx = scheduler.get_kv_token_offset(kv_token_base, kv_split_idx);
                const uint32_t actual_kv = scheduler.get_num_kv_block_tokens(kv_token_base, kv_split_idx);
                const uint32_t issued_n = aligned(actual_kv, kNumVecElems);
                asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(kKVEventBase + kv_stage_idx));
                fill_mn_tail_block(l1kv[kv_stage_idx], actual_kv, issued_n);
                if constexpr (kIsPaged) {
                    const asc_load_l2_cache_mode kv_load_hint = scheduler.has_multiple_q_blocks()
                        ? asc_load_l2_cache_mode::NORMAL_LAST_VICTIM : asc_load_l2_cache_mode::NOTALLOC_KEEP;
                    const uint32_t num_pages = ceil_div(actual_kv, kPageSize);
                    for (uint32_t compact_page = 0; compact_page < num_pages; compact_page += 2) {
                        uint32_t first_page, page_gap;
                        const uint32_t num_batch = scheduler.get_physical_page_batch(kv_token_base, kv_split_idx, compact_page, num_pages, first_page, page_gap);
                        const auto page_src = kv_t + scheduler.get_kv_page_offset_bytes(first_page);
                        const auto page_dst = l1kv[kv_stage_idx].offset_bytes(compact_page * kPageSize * 32u);
                        const uint32_t valid_rows = min(kPageSize, actual_kv - compact_page * kPageSize);
                        copy_gm_to_l1<kUseAscendKVLayout>(page_dst, page_src, valid_rows, kHeadDim, kv_load_hint,
                                                          num_batch, kPageSize,
                                                          page_gap * scheduler.kv_page_stride_bytes);
                    }
                } else {
                    const gm_ptr<int16_t, Major::MN> sf_kv_t(scheduler.get_kv_sf_stride(), reinterpret_cast<uintptr_t>(sf_kv));
                    copy_gm_to_l1(l1kv[kv_stage_idx], kv_t.index(kv_idx, 0), actual_kv, kHeadDim,
                                  asc_load_l2_cache_mode::NOTALLOC_KEEP);
                    copy_gm_to_l1_mx(l1_sf_kv[kv_stage_idx], sf_kv_t.index(kv_idx, 0), actual_kv, kSFPairs,
                                     asc_load_l2_cache_mode::NOTALLOC_KEEP);
                }
                asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(kKVEventBase + kv_stage_idx));
                asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(kKVEventBase + kv_stage_idx));

                asc_sync_wait(PIPE_M, PIPE_MTE1, EVENT_ID0);
                const auto l0b_tile = l0b.as_mad_aligned(issued_n, MAD_K);
                copy_l1_to_l0b(l0b_tile, l1kv[kv_stage_idx], 0, 0);
                if constexpr (kIsPaged)
                    wait_intra_block(PIPE_MTE1, (kv_stage_idx ^ 1u) & 1u, kNumAccumStages + kv_stage_idx);
                copy_l1_to_l0b_mx(l0b_tile, l1_sf_kv[kv_stage_idx], 0, 0, MAD_K);
                if constexpr (kIsPaged)
                    set_intra_block(PIPE_MTE1, (kv_stage_idx ^ 1u) & 1u, kNumAccumStages + kv_stage_idx);
                asc_sync_notify(PIPE_MTE1, PIPE_M, EVENT_ID0);
                asc_sync_wait(PIPE_MTE1, PIPE_M, EVENT_ID0);

                // Alternate an unpaired tile between AIVs across KV splits.
                const uint32_t owner_offset = num_qh_tiles_in_block & kv_split_idx & 1;
                for (uint32_t qh_tile_idx = 0; qh_tile_idx < num_qh_tiles_in_block; ++qh_tile_idx) {
                    const uint32_t qh_base = qh_tile_idx * MAD_M;
                    const uint32_t q_tile_base = qh_base / kNumHeads;
                    // Skip tiles with no row window overlapping this split
                    if (!is_tile_active(q_tile_base, kv_idx, kv_idx + actual_kv))
                        continue;
                    const uint32_t num_m_rows = min(MAD_M, num_qh_rows_in_block - qh_base);
                    const auto l0c_tile = l0c[cd_stage_idx].as_mad_aligned(num_m_rows, issued_n);
                    const uint32_t accum_owner = (qh_tile_idx + owner_offset) & 1u;
                    const uint32_t accum_stage_idx = ((kv_split_idx * num_qh_tiles_in_block + qh_tile_idx) / 2) % kNumAccumStages;
                    mad(l0c_tile, l0a[qh_tile_idx], l0b_tile, num_m_rows, issued_n, MAD_K,
                        asc_unit_flag_mode::ENABLE_UPDATE, true);
                    wait_intra_block(PIPE_FIX, static_cast<uint8_t>(accum_owner),
                                     static_cast<uint8_t>(accum_stage_idx));
                    const auto accum_tile = ub_accum[accum_stage_idx].as_mad_aligned(num_m_rows, issued_n);
                    copy_l0c_to_ub(accum_tile, l0c_tile, num_m_rows, issued_n, asc_dual_dst_mode::DUAL_DST_DISABLE,
                                   accum_owner, infer_quant_mode<bfloat16_t, float>(), true, false,
                                   asc_unit_flag_mode::ENABLE_UPDATE);
                    set_intra_block(PIPE_FIX, accum_owner, accum_stage_idx);
                    cd_stage_idx = (cd_stage_idx + 1) % kNumCDStages;
                }
                asc_sync_notify(PIPE_M, PIPE_MTE1, EVENT_ID0);
                asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(kKVEventBase + kv_stage_idx));
                kv_stage_idx = (kv_stage_idx + 1) % kNumKVStages;
            }
            asc_sync_notify(PIPE_M, PIPE_MTE1, EVENT_ID1);
            asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(kQEventBase + q_stage_idx));
            q_stage_idx = (q_stage_idx + 1) % kNumQStages;
        }
    } else {
        const uint32_t sub_id = subblock_id;
        constexpr uint32_t kSFLoadEventIdx = kNumWeightStages;
        constexpr uint32_t kSFStoreEventIdx = kNumLogitsStages + 1;
        static_assert(!kIsPaged || kSFStoreEventIdx + kNumSFUBStages <= 8, "Scale event overflow");
        // Release accum stage on PIPE_V so it is not serialized behind the GM store.
        set_intra_blocks<PIPE_V, kNumAccumStages>(static_cast<uint8_t>(sub_id));
        set_flags<PIPE_MTE3, PIPE_V, kNumLogitsStages>();
        // Keep the shared head scratch single-buffered across MTE3 stores.
        static_assert(kNumLogitsStages < 8, "Scratch event id must fit the 8-id flag space");
        asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(kNumLogitsStages));
        set_flags<PIPE_V, PIPE_MTE2, kNumWeightStages>();
        if constexpr (kIsPaged)
            set_flags<PIPE_MTE3, PIPE_MTE2, kNumSFUBStages, kSFLoadEventIdx>();

        uint32_t logits_stage_idx = 0;
        uint32_t weight_stage_idx = 0;
        uint32_t kv_stage_idx = 0;
        // Keep the SF producer ahead across requests; the consumer retains its own cursor.
        auto sf_scheduler = scheduler;
        uint32_t sf_q_block_idx = 0, sf_kv_token_base = 0, num_sf_splits = 0, sf_split_idx = 0;
        uint32_t sf_num_pages[kNumSFUBStages] = {};
        const auto prefetch_sf_kv = [&](uint32_t kv_stage_idx) __aicore__ {
            if constexpr (kIsPaged) {
                const uint32_t sf_stage_idx = kv_stage_idx / 2;
                if (sf_split_idx == num_sf_splits) {
                    if (!sf_scheduler.next_q_block(sf_q_block_idx, sf_kv_token_base, num_sf_splits)) {
                        // An empty future slot issues no copy or ready notification.
                        if (sub_id == ((kv_stage_idx ^ 1u) & 1u)) sf_num_pages[sf_stage_idx] = 0;
                        return;
                    }
                    sf_split_idx = 0;
                }
                // Both AIVs advance; only the owner transfers this stage.
                const uint32_t kv_split_idx = sf_split_idx++;
                if (sub_id != ((kv_stage_idx ^ 1u) & 1u)) return;
                const uint32_t num_pages = ceil_div(
                    sf_scheduler.get_num_kv_block_tokens(sf_kv_token_base, kv_split_idx), kPageSize);
                sf_num_pages[sf_stage_idx] = num_pages;
                const asc_load_l2_cache_mode sf_load_hint = sf_scheduler.has_multiple_q_blocks()
                    ? asc_load_l2_cache_mode::NORMAL_LAST_VICTIM : asc_load_l2_cache_mode::NOTALLOC_KEEP;
                asc_sync_wait(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(kSFLoadEventIdx + sf_stage_idx));
                for (uint32_t compact_page = 0; compact_page < num_pages; compact_page += 2) {
                    uint32_t first_page, page_gap;
                    const uint32_t num_batch = sf_scheduler.get_physical_page_batch(
                        sf_kv_token_base, kv_split_idx, compact_page, num_pages, first_page, page_gap);
                    const auto sf_page_src = reinterpret_cast<__gm__ int16_t*>(
                        reinterpret_cast<uintptr_t>(sf_kv) + sf_scheduler.get_sf_page_offset_bytes(first_page));
                    copy_gm_to_ub(ub_sf_kv[sf_stage_idx].ptr() + compact_page * kPageSize * kSFPairs,
                        sf_page_src, num_batch, kPageSize * kSFPairs,
                        page_gap * sf_scheduler.sf_page_stride_bytes, sf_load_hint);
                }
                asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(kSFLoadEventIdx + sf_stage_idx));
            }
        };
        const auto publish_sf_kv = [&](uint32_t kv_stage_idx) __aicore__ {
            if constexpr (kIsPaged) {
                if (sub_id != ((kv_stage_idx ^ 1u) & 1u)) return;
                const uint32_t sf_stage_idx = kv_stage_idx / 2;
                const uint32_t num_pages = sf_num_pages[sf_stage_idx];
                if (!num_pages) return;
                asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(kSFLoadEventIdx + sf_stage_idx));
                if constexpr (!kUseAscendKVLayout && kSFPairs == 2)
                    vf_mqa_logits_pack_scales(ub_sf_kv[sf_stage_idx].ptr(), num_pages * kPageSize * kSFPairs);
                asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(kSFStoreEventIdx + sf_stage_idx));
                asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(kSFStoreEventIdx + sf_stage_idx));
                wait_intra_block(PIPE_MTE3, sub_id, kNumAccumStages + kv_stage_idx);
                asc_copy_ub2l1(l1_sf_kv[kv_stage_idx].ptr(), ub_sf_kv[sf_stage_idx].ptr(),
                    num_pages * kPageSize * kSFPairs * sizeof(int16_t));
                set_intra_block(PIPE_MTE3, sub_id, kNumAccumStages + kv_stage_idx);
                asc_sync_notify(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(kSFLoadEventIdx + sf_stage_idx));
            }
        };
        if constexpr (kIsPaged) {
            for (uint32_t stage_idx = 0; stage_idx < kNumKVStages; ++stage_idx)
                prefetch_sf_kv(stage_idx);
            for (uint32_t stage_idx = 0; stage_idx < kNumKVStages; ++stage_idx)
                publish_sf_kv(stage_idx);
        }

        while (scheduler.next_q_block(q_block_idx, kv_token_base, num_kv_splits)) {
            const uint32_t q_base = scheduler.get_q_base(q_block_idx);
            const uint32_t num_q_tokens_in_block = scheduler.get_num_q_tokens_in_block(q_block_idx);
            const uint32_t num_qh_rows_in_block = num_q_tokens_in_block * kNumHeads;
            const uint32_t num_qh_tiles_in_block = ceil_div(num_qh_rows_in_block, MAD_M);
            // Per-Q-block seq bounds via MTE2, reused across all KV splits
            if constexpr (kIsPaged) {
                copy_gm_to_ub(ub_seq_end.ptr(), scheduler.context_lens + q_base, 1, num_q_tokens_in_block,
                              num_q_tokens_in_block * sizeof(uint32_t), asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
            } else {
                copy_gm_to_ub(ub_seq_start.ptr(), scheduler.cu_seq_len_k_start + q_base, 1, num_q_tokens_in_block,
                              num_q_tokens_in_block * sizeof(uint32_t), asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                copy_gm_to_ub(ub_seq_end.ptr(), scheduler.cu_seq_len_k_end + q_base, 1, num_q_tokens_in_block,
                              num_q_tokens_in_block * sizeof(uint32_t), asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
            }
            // Scalar UB reads run on PIPE_S, outside the V/MTE2/MTE3 event triads; order them
            // after the seq copies with a (MTE2, PIPE_S) flag pair (set/wait share arg order).
            asc_sync_notify(PIPE_MTE2, PIPE_S, EVENT_ID0);
            // UB-backed seq bounds; serve the tile check and store paths
            const auto seq_ks = [&](uint32_t q_in_block) __aicore__ -> uint32_t {
                return kIsPaged ? 0u : ub_seq_start.ptr()[q_in_block];
            };
            const auto seq_ke = [&](uint32_t q_in_block) __aicore__ -> uint32_t {
                return ub_seq_end.ptr()[q_in_block];
            };
            const auto is_tile_active = [&](uint32_t q_tile_base, uint32_t kv_idx, uint32_t kv_end) __aicore__ -> bool {
                for (uint32_t q_idx = q_tile_base; q_idx < min(q_tile_base + kNumQTokensPerTile, num_q_tokens_in_block); ++q_idx) {
                    if (kv_idx >= seq_ke(q_idx))
                        continue;
                    if constexpr (kIsPaged)
                        return true;
                    else if (seq_ks(q_idx) < kv_end)
                        return true;
                }
                return false;
            };
            asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(weight_stage_idx));
            copy_gm_to_ub(ub_weights[weight_stage_idx].ptr(), weights.index(q_base, 0).ptr(), num_q_tokens_in_block,
                          kNumHeads, weights.stride_outer * sizeof(bfloat16_t),
                          asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
            asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(weight_stage_idx));
            asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(weight_stage_idx));
            mqa_logits_normalize_weights<kNumQTokensPerBlock * kNumHeads>(
                ub_weights[weight_stage_idx].ptr(), ub_weight_fp32.ptr(), num_q_tokens_in_block * kNumHeads);
            asc_sync_notify(PIPE_V, PIPE_S, EVENT_ID1);
            asc_sync_wait(PIPE_V, PIPE_S, EVENT_ID1);
            asc_sync_wait(PIPE_MTE2, PIPE_S, EVENT_ID0);

            for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++kv_split_idx) {
                if constexpr (kIsPaged)
                    prefetch_sf_kv(kv_stage_idx);
                const uint32_t kv_idx = scheduler.get_kv_token_offset(kv_token_base, kv_split_idx);
                const uint32_t actual_kv = scheduler.get_num_kv_block_tokens(kv_token_base, kv_split_idx);
                const uint32_t issued_n = aligned(actual_kv, kNumVecElems);
                uint32_t reversed_pairs = 0;
                if constexpr (kIsPaged) {
                    const uint32_t num_pages = ceil_div(actual_kv, kPageSize);
                    for (uint32_t compact_page = 0; compact_page + 1 < num_pages; compact_page += 2) {
                        const uint32_t physical_page = scheduler.get_physical_page(kv_token_base, kv_split_idx, compact_page);
                        const uint32_t partner_physical_page = scheduler.get_physical_page(kv_token_base, kv_split_idx, compact_page + 1);
                        reversed_pairs |= static_cast<uint32_t>(physical_page > partner_physical_page)
                                          << (compact_page / 2);
                    }
                }
                const uint32_t owner_offset = num_qh_tiles_in_block & kv_split_idx & 1;
                for (uint32_t qh_tile_idx = sub_id ^ owner_offset; qh_tile_idx < num_qh_tiles_in_block; qh_tile_idx += 2) {
                    const uint32_t qh_base = qh_tile_idx * MAD_M;
                    const uint32_t q_tile_base = qh_base / kNumHeads;
                    // Mirror the AIC's skip: run only when a row window overlaps this split
                    if (!is_tile_active(q_tile_base, kv_idx, kv_idx + actual_kv))
                        continue;
                    const uint32_t accum_stage_idx = ((kv_split_idx * num_qh_tiles_in_block + qh_tile_idx) / 2) % kNumAccumStages;
                    wait_intra_block(PIPE_V, static_cast<uint8_t>(sub_id), static_cast<uint8_t>(accum_stage_idx));
                    asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(logits_stage_idx));
                    const auto ub_logits_stage = ub_logits[logits_stage_idx];
                    auto score = ub_accum[accum_stage_idx].ptr();
                    auto weight = ub_weights[weight_stage_idx].ptr();
                    for (uint32_t q_local = 0; q_local < kNumQTokensPerTile;) {
                        const uint32_t q_idx = q_tile_base + q_local;
                        if (q_idx >= num_q_tokens_in_block)
                            break;
                        auto line = ub_logits_stage.ptr() + q_local * kSplitKV;
                        q_local += mqa_logits_reduce_queries<kPageSize, kNumQTokensPerTile, kNumHeads, kSplitKV>(score, weight, line, q_local, q_idx,
                                                                                                                 num_q_tokens_in_block, issued_n, reversed_pairs);
                    }
                    // Release accum on PIPE_V: V-pipe order guarantees it runs after the reduce
                    // that consumed ub_accum, so no extra V->MTE3 done-fence is needed.
                    set_intra_block(PIPE_V, static_cast<uint8_t>(sub_id), static_cast<uint8_t>(accum_stage_idx));
                    if constexpr (kEmitBlockMax) {
                        for (uint32_t q_store = 0; q_store < kNumQTokensPerTile; ++q_store) {
                            const uint32_t q_in_block = q_tile_base + q_store;
                            if (q_in_block >= num_q_tokens_in_block)
                                break;
                            const uint32_t ke = seq_ke(q_in_block);
                            if (kv_idx >= ke)
                                continue;
                            const uint32_t valid_tokens = min(actual_kv, ke - kv_idx);
                            vf_mqa_logits_block_max(
                                ub_block_scores[logits_stage_idx].ptr() + q_store * (kSplitKV / 8),
                                ub_logits_stage.ptr() + q_store * kSplitKV,
                                ub_block_scratch.ptr(),
                                valid_tokens, kv_idx / 8, (ke - 1) / 8);
                        }
                    }
                    asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(logits_stage_idx));

                    // Store current logits directly.
                    // MTE3 needs a 32B-aligned UB source, so the unaligned head of a
                    // row range is staged through the aligned head scratch first.
                    asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(logits_stage_idx));
                    for (uint32_t q_store = 0; q_store < kNumQTokensPerTile; ++q_store) {
                        const uint32_t q_in_block = q_tile_base + q_store;
                        if (q_in_block >= num_q_tokens_in_block)
                            break;
                        const uint32_t row = q_base + q_in_block;
                        const uint32_t ks = seq_ks(q_in_block);
                        const uint32_t ke = seq_ke(q_in_block);
                        const uint32_t store_begin = max(kv_idx, ks);
                        const uint32_t store_end = min(kv_idx + actual_kv, ke);
                        if (store_begin >= store_end)
                            continue;
                        const uint32_t dst_col = store_begin - ks;
                        const auto line = ub_logits_stage.ptr() + q_store * kSplitKV;
                        mqa_logits_store<kNumLogitsStages>(logits.index(row, dst_col).ptr(),
                                                           line + store_begin - kv_idx, ub_logits_head.ptr(),
                                                           store_end - store_begin);
                        if constexpr (kEmitBlockMax) {
                            const uint32_t num_block_bytes = ceil_div(store_end - store_begin, 8u) * sizeof(bfloat16_t);
                            asc_copy_ub2gm_align(block_scores.index(row, dst_col / 8).ptr(),
                                ub_block_scores[logits_stage_idx].ptr() + q_store * (kSplitKV / 8),
                                1, num_block_bytes, asc_store_l2_cache_mode::NOTALLOC_CLEAN,
                                num_block_bytes, num_block_bytes);
                        }
                    }
                    asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(logits_stage_idx));
                    logits_stage_idx = (logits_stage_idx + 1) % kNumLogitsStages;
                }
                if constexpr (kIsPaged)
                    publish_sf_kv(kv_stage_idx);
                kv_stage_idx = (kv_stage_idx + 1) % kNumKVStages;
            }
            asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(weight_stage_idx));
            weight_stage_idx = (weight_stage_idx + 1) % kNumWeightStages;
        }
    }
}

} // namespace deep_gemm

namespace {
template <uint32_t kPageRows, bool kEmitBlockMax>
__aicore__ inline void score_pages(__gm__ int64_t* args, uint32_t active,
                                  uint32_t core_id, uint32_t subblock_id) {
    using namespace deep_gemm;
    auto key = reinterpret_cast<__gm__ Tensor*>(args[3]);
    auto scale = reinterpret_cast<__gm__ Tensor*>(args[4]);
    auto table = reinterpret_cast<__gm__ Tensor*>(args[5]);
    auto output = reinterpret_cast<__gm__ Tensor*>(args[9]);
    constexpr uint32_t heads = 32, dim = 128, split = 512, mad_m = 64;
    const auto make_scheduler = [&](uint32_t id) __aicore__ {
        return PagedMQALogitsScheduler<8 * mad_m / heads, split, kPageRows, heads>(
            id, active, data<uint32_t>(args, 7), data<uint32_t>(args, 6),
            data<uint32_t>(args, 8), data<uint32_t>(args, 5), table->strides[0],
            key->strides[0], scale->strides[0]);
    };
    const gm_ptr<uint8_t, Major::K> q(dim / 2, reinterpret_cast<uintptr_t>(data<uint8_t>(args, 0)));
    const gm_ptr<uint8_t, Major::K> kv(dim / 2, reinterpret_cast<uintptr_t>(data<uint8_t>(args, 3)));
    const gm_ptr<bfloat16_t, Major::K> weights(heads, reinterpret_cast<uintptr_t>(data<bfloat16_t>(args, 2)));
    const gm_ptr<bfloat16_t, Major::K> logits(output->strides[0], reinterpret_cast<uintptr_t>(data<bfloat16_t>(args, 9)));
    const auto block_output = kEmitBlockMax ? reinterpret_cast<__gm__ Tensor*>(args[10]) : nullptr;
    const gm_ptr<bfloat16_t, Major::K> block_scores(
        kEmitBlockMax ? block_output->strides[0] : 0,
        kEmitBlockMax ? reinterpret_cast<uintptr_t>(data<bfloat16_t>(args, 10)) : 0);
    mqa_logits_core_impl<true, heads, dim, split, mad_m, 2, 2, 4, 3, 3, kPageRows, false, kEmitBlockMax>(
        q, kv, reinterpret_cast<__gm__ int16_t*>(data<uint8_t>(args, 1)),
        reinterpret_cast<__gm__ int16_t*>(data<uint8_t>(args, 4)),
        weights, logits, block_scores, make_scheduler, core_id, subblock_id);
}
}

template <bool kEmitBlockMax>
static __aicore__ __attribute__((noinline)) void score(__gm__ int64_t* args) {
    using namespace deep_gemm;
    auto key = reinterpret_cast<__gm__ Tensor*>(args[3]);
    const uint32_t active = args[kEmitBlockMax ? 12 : 10];
    const uint32_t core_id = get_block_idx(args);
    uint32_t subblock_id = 0;
#if defined(__DAV_VEC__)
    subblock_id = get_sub_block_id(args);
#endif
    asc_sync_pipe(PIPE_ALL);
#if defined(__DAV_CUBE__)
    asc_sync_intra_arrive(PIPE_FIX, 8);
    asc_sync_intra_arrive(PIPE_FIX, 24);
    asc_sync_intra_wait(PIPE_FIX, 9);
    asc_sync_intra_wait(PIPE_FIX, 25);
    asc_sync_notify(PIPE_FIX, PIPE_MTE2, EVENT_ID7);
    asc_sync_wait(PIPE_FIX, PIPE_MTE2, EVENT_ID7);
#elif defined(__DAV_VEC__)
    asc_sync_intra_wait(PIPE_V, 8);
    asc_sync_intra_arrive(PIPE_V, 9);
    asc_sync_notify(PIPE_V, PIPE_MTE2, EVENT_ID7);
    asc_sync_wait(PIPE_V, PIPE_MTE2, EVENT_ID7);
#endif
    asc_init();
    if (key->shapes[1] == 64) score_pages<64, kEmitBlockMax>(args, active, core_id, subblock_id);
    else score_pages<128, kEmitBlockMax>(args, active, core_id, subblock_id);
    // Consume free pipeline slots before the persistent runtime reuses the core.
#if defined(__DAV_CUBE__)
    wait_flags<PIPE_M, PIPE_MTE1, 2>();
    wait_flags<PIPE_MTE1, PIPE_MTE2, 6>();
    for (uint32_t lane = 0; lane < 2; ++lane)
        for (uint32_t stage = 0; stage < 3; ++stage)
            wait_intra_block(PIPE_FIX, lane, stage);
#elif defined(__DAV_VEC__)
    wait_flags<PIPE_MTE3, PIPE_V, 4>();
    wait_flags<PIPE_V, PIPE_MTE2, 2>();
    wait_flags<PIPE_MTE3, PIPE_MTE2, 2, 2>();
    for (uint32_t stage = 0; stage < 4; ++stage)
        if (subblock_id == ((stage ^ 1u) & 1u))
            wait_intra_block(PIPE_MTE3, subblock_id, 3 + stage);
#endif
    asc_sync();
}
#if defined(__DAV_VEC__)
#include <c_api/asc_simd.h>

// DeepSelect: Ascend configuration and row selection.
#include <cstdint>
#include <type_traits>

#include <kernel_operator.h>

namespace topk_select_ascend {

inline constexpr uint32_t MAX_VOCAB_SIZE = 2 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Compile-time constants
// ---------------------------------------------------------------------------

inline constexpr uint32_t TOPK_COMPACT_PADDING = 256;
inline constexpr uint32_t TOPK_HISTOGRAM_BINS = 256;
inline constexpr uint32_t TOPK_UB_LIMIT_BYTES = 248 * 1024;
inline constexpr uint32_t VST_VLD_ALIGN_BYTES = 32;
inline constexpr uint32_t VST_VLD_ALIGN_ELEMENTS = VST_VLD_ALIGN_BYTES / sizeof(uint16_t);

template <typename T1, typename T2>
__aicore__ constexpr auto ceil_div(T1 a, T2 b) -> decltype(a + b) {
    return (a + b - 1) / b;
}

template <typename T1, typename T2>
__aicore__ constexpr auto aligned(T1 a, T2 b) -> decltype(a + b) {
    return ceil_div(a, b) * b;
}

struct TopkPage {
    uint32_t start_offset;
    uint32_t len;
};

// Map each logical scan page to a physical input range. Short inputs are scanned sequentially.
// For longer inputs, full pages are visited in a randomized order, while the physical tail is
// split at a 128-element boundary. The last tail segment is placed at the end of warmup, and the
// preceding segment is visited last, keeping page starts aligned and avoiding out-of-bounds loads.
template <uint32_t kWarmupPages, uint32_t kShufflePageSize>
struct TopkPermGenerator {
    uint32_t valid_len;
    uint32_t num_pages;
    uint32_t size;
    uint32_t seed0;
    uint32_t seed1;

    __aicore__ TopkPermGenerator(uint32_t valid_len, uint64_t seed)
        : valid_len(valid_len),
          num_pages(ceil_div(valid_len, kShufflePageSize)),
          size(num_pages > 1 ? num_pages - 2 : 0),
          seed0(static_cast<uint16_t>(seed)),
          seed1(static_cast<uint16_t>(seed >> 16) % (size == 0 ? 1 : size)) {}

    __aicore__ inline uint32_t reflect(uint32_t value, uint32_t pivot, uint32_t size) const {
        return value <= pivot ? pivot - value : size + pivot - value;
    }

    __aicore__ inline TopkPage get_page_id(uint32_t logical_page_id) const {
        if (num_pages == 1)
            return {0, valid_len};

        if (num_pages <= kWarmupPages) {
            const uint32_t start_offset = logical_page_id * kShufflePageSize;
            const uint32_t len = start_offset + kShufflePageSize <= valid_len
                                     ? kShufflePageSize
                                     : valid_len - start_offset;
            return {start_offset, len};
        }

        const uint32_t last_start = aligned(valid_len - kShufflePageSize, 128u);
        if (logical_page_id == kWarmupPages - 1)
            return {last_start, valid_len - last_start};
        if (logical_page_id == num_pages - 1) {
            const uint32_t penultimate_start = (num_pages - 2) * kShufflePageSize;
            return {penultimate_start, last_start - penultimate_start};
        }

        uint32_t value = logical_page_id < kWarmupPages - 1 ? logical_page_id : logical_page_id - 1;

        value = (value * 65537u + seed0) % size;
        value = reflect(value, seed1, size);
        const uint32_t page_id = (value * 65521u + seed1) % size;
        return {page_id * kShufflePageSize, kShufflePageSize};
    }
};

// ---------------------------------------------------------------------------
// Algorithm parameters
// ---------------------------------------------------------------------------
// Scan randomized shuffle pages and select a warmup top-k by radix histograms.
// Later blocks retain values above the current threshold. Full candidate buffers
// merge with the retained top-k and update the threshold before scanning onward.
template <uint32_t kK = 4 * 1024, uint32_t kFilteredCapacity = 4096,
          uint32_t kWarmupPages = 3, uint32_t kNumStages = 3,
          uint32_t kBlockSize = 16 * 1024, uint32_t kShufflePageSize = 2 * 1024,
          uint32_t kBlockSizeBits = 14>
struct TopkSelectConfig {
    static constexpr uint32_t topk_capacity = kK;
    static constexpr uint32_t filtered_capacity = kFilteredCapacity;
    static constexpr uint32_t warmup_pages = kWarmupPages;
    static constexpr uint32_t block_size = kBlockSize;
    static constexpr uint32_t shuffle_page_size = kShufflePageSize;
    static constexpr uint32_t shuffle_page_size_bits =
        __builtin_ctz(kShufflePageSize);  // Base-2 shuffle-page size.
    static constexpr uint32_t block_size_bits = kBlockSizeBits;  // Base-2 scan-block size.
    static constexpr uint32_t num_stages = kNumStages;  // Software pipeline stages.
};

// Rows no longer than the pivot use LowerConfig; longer rows use UpperConfig.
template <uint32_t kPivot, typename kLowerConfig, typename kUpperConfig>
struct TopkSelectPolicyConfig {
    static constexpr uint32_t pivot = kPivot;
    using LowerConfig = kLowerConfig;
    using UpperConfig = kUpperConfig;
};

using Topk512Config = TopkSelectPolicyConfig<
    32832, TopkSelectConfig<512, 4096, 2>, TopkSelectConfig<512, 1536, 2>>;
using Topk1024Config = TopkSelectPolicyConfig<
    36800, TopkSelectConfig<1024, 3072, 3>, TopkSelectConfig<1024, 2560, 3>>;
using Topk4096Config = TopkSelectPolicyConfig<
    10432, TopkSelectConfig<4096, 2560, 4>, TopkSelectConfig<4096, 3072, 4>>;

// ---------------------------------------------------------------------------
// Unified-buffer (UB) memory plan
// ---------------------------------------------------------------------------
// Every offset below is a compile-time constant derived from the algorithm
// parameters. The plan is keyed by the algorithm config instead of
// `Kernel<CONFIG>` because the runtime pivot selects between the lower and
// upper config.
template <typename AlgoConfig, bool kSortedIndex, typename value_t>
struct UnifiedBufferMemoryPlan {
    static constexpr uint32_t kTopkCapacity = AlgoConfig::topk_capacity;
    static constexpr uint32_t kFilteredCapacity = AlgoConfig::filtered_capacity;
    static constexpr uint32_t kWarmupPages = AlgoConfig::warmup_pages;
    static constexpr uint32_t kBlockSize = AlgoConfig::block_size;
    static constexpr uint32_t kShufflePageSize = AlgoConfig::shuffle_page_size;
    static constexpr uint32_t kBlockSizeBits = AlgoConfig::block_size_bits;
    static constexpr uint32_t kNumStages = AlgoConfig::num_stages;

    static_assert(kWarmupPages > 0 && kNumStages > 0);
    static_assert(kNumStages <= 8, "num_stages exceeds the available event IDs");
    static_assert(kWarmupPages * kShufflePageSize - 128 >= kTopkCapacity,
                  "warmup pages cannot hold top-k candidates");
    static_assert(kBlockSize == (uint32_t(1) << kBlockSizeBits),
                  "block size and block size bits must match");
    static_assert(kBlockSize % kShufflePageSize == 0,
                  "block size must contain whole shuffle pages");
    static_assert((kShufflePageSize & (kShufflePageSize - 1)) == 0,
                  "shuffle page size must be a power of two");
    static constexpr uint32_t kShufflePageCapacity = MAX_VOCAB_SIZE / kShufflePageSize;
    static_assert(uint64_t(kShufflePageCapacity) * kShufflePageSize <= (uint64_t(1) << 32),
                  "shuffle page ids cannot reconstruct uint32 indices");
    static_assert(TOPK_COMPACT_PADDING >= VST_VLD_ALIGN_ELEMENTS,
                  "compact padding must cover filtered-count alignment");

    static constexpr uint32_t kWarmupElements = kWarmupPages * kShufflePageSize;
    static constexpr uint32_t kWarmupBytes = kWarmupElements * sizeof(value_t);
    static constexpr uint32_t kCandidateCapacity =
        kTopkCapacity + kBlockSize + kFilteredCapacity;
    static constexpr uint32_t kCandidatePartBytes =
        aligned((kCandidateCapacity + TOPK_COMPACT_PADDING) * sizeof(uint16_t), 32u);
    static constexpr uint32_t kCandidateBytes = 3 * kCandidatePartBytes;
    static constexpr uint32_t kReuseBytes =
        kWarmupBytes > kCandidateBytes ? kWarmupBytes : kCandidateBytes;
    static constexpr uint32_t kInputBytes = kNumStages * kBlockSize * sizeof(value_t);
    static constexpr uint32_t kHistogramBytes = TOPK_HISTOGRAM_BINS * sizeof(uint16_t);
    static constexpr uint32_t kTemporaryPositionBytes =
        aligned(kTopkCapacity * sizeof(uint16_t), 32u);
    static constexpr uint32_t kThresholdBytes = 32;
    static constexpr uint32_t kStateBytes = 32 + 64 * sizeof(uint32_t);
    static constexpr uint32_t kShufflePageIdBytes = kShufflePageCapacity * sizeof(uint16_t);

    // The leading region is deliberately reused by two phases: warmup owns it
    // first; candidate value/low/high own it after warmup selection starts.
    static constexpr uintptr_t kCandidateValueOffset = 0;
    static constexpr uintptr_t kCandidateIndexLowOffset =
        kCandidateValueOffset + kCandidatePartBytes;
    static constexpr uintptr_t kCandidateIndexHighOffset =
        kCandidateIndexLowOffset + kCandidatePartBytes;
    static constexpr uintptr_t kHistogramOffset = kReuseBytes;
    static constexpr uintptr_t kHighHistogramOffset = kHistogramOffset + kHistogramBytes;
    static constexpr uintptr_t kThresholdOffset = kHighHistogramOffset + kHistogramBytes;
    static constexpr uintptr_t kStateOffset = kThresholdOffset + kThresholdBytes;
    static constexpr uintptr_t kTemporaryPositionOffset = kStateOffset + kStateBytes;
    static constexpr uintptr_t kShufflePageIdOffset =
        kTemporaryPositionOffset + kTemporaryPositionBytes;
    static constexpr uintptr_t kMetadataEnd = kShufflePageIdOffset + kShufflePageIdBytes;
    static constexpr uintptr_t kMinInputOffset = (uint32_t(1) << 16) * sizeof(value_t);
    static constexpr uintptr_t kInputOffset =
        kMetadataEnd > kMinInputOffset ? kMetadataEnd : kMinInputOffset;
    static constexpr uintptr_t kUbEnd = kInputOffset + kInputBytes;

    static constexpr uint32_t kSortIndexBytes = kTopkCapacity * sizeof(uint32_t);
    static constexpr uint32_t kSortTemporaryBytes = 512 + 11 * kTopkCapacity;
    static constexpr uintptr_t kSortedIndexOffset = kInputOffset + kSortIndexBytes;
    static constexpr uintptr_t kSortPermutationOffset = kSortedIndexOffset + kSortIndexBytes;
    static constexpr uintptr_t kSortTemporaryOffset = kSortPermutationOffset + kSortIndexBytes;
    static constexpr uintptr_t kSortEnd = kSortTemporaryOffset + kSortTemporaryBytes;

    // Sorting reuses the input staging region after all loads end.
    static_assert(kUbEnd <= TOPK_UB_LIMIT_BYTES, "topk_select UB layout overflow");
    static_assert(!kSortedIndex || kSortIndexBytes <= kInputBytes,
                  "sorted-index source does not fit in reused input staging");
    static_assert(!kSortedIndex || kSortEnd <= TOPK_UB_LIMIT_BYTES,
                  "sorted-index workspace exceeds UB capacity");
};

}  // namespace topk_select_ascend

#include <cstdint>
#include <type_traits>

#include <adv_api/sort/sort.h>
#include <c_api/asc_simd.h>
#include <kernel_operator.h>


using namespace AscendC;

namespace topk_select_ascend {

inline constexpr AscendC::SortConfig kIndexMergeSortConfig{AscendC::SortType::MERGE_SORT, false};

// Accumulate a 256-bin BF16 byte histogram into two 128-lane vectors. For the low byte, only
// values whose high byte matches prefix are counted; kDescending controls the bucket order.
template <bool kHighByte, bool kDescending = false>
__simd_callee__ inline void topk_accumulate_histogram(vector_uint16_t& hist_low,
                                                      vector_uint16_t& hist_high,
                                                      __ubuf__ bfloat16_t* values, uint32_t count,
                                                      vector_uint8_t prefix) {
    constexpr uint32_t kValuesPerRound = 256;

    vector_bool mask_u8 = asc_create_mask_b8(PAT_ALL);
    vector_uint8_t positive_flip, negative_flip;
    asc_duplicate_scalar(positive_flip, uint8_t(kDescending ? 0x7f : 0x80), mask_u8);
    asc_duplicate_scalar(negative_flip, uint8_t(kDescending ? 0x00 : 0xff), mask_u8);
    vector_uint8_t raw_prefix;
    vector_bool negative_prefix;
    if constexpr (!kHighByte) {
        asc_lt_scalar(negative_prefix, prefix, uint8_t(0x80), mask_u8);
        vector_uint8_t prefix_flip;
        asc_select(prefix_flip, negative_flip, positive_flip, negative_prefix);
        asc_xor(raw_prefix, prefix, prefix_flip, mask_u8);
    }
    uint32_t remaining = count;
    const uint32_t rounds = ceil_div(count, kValuesPerRound);
    auto* source = reinterpret_cast<__ubuf__ uint16_t*>(values);
    for (uint32_t round = 0; round < rounds; ++round) {
        vector_uint8_t low_byte, high_byte;
        asc_loadalign_deintlv(
            low_byte, high_byte,
            reinterpret_cast<__ubuf__ uint8_t*>(source + round * kValuesPerRound));

        vector_bool active = asc_update_mask_b8(remaining);
        if constexpr (kHighByte) {
            if constexpr (kDescending) {
                vector_bool nonnegative;
                asc_ge_scalar(nonnegative, reinterpret_cast<vector_int8_t&>(high_byte), int8_t(0),
                         mask_u8);
                // No asc_* C-API exposes MODE_MERGING for vxor; keep the raw intrinsic.
                vxor(high_byte, high_byte, positive_flip, nonnegative, MODE_MERGING);
            } else {
                vector_bool negative;
                asc_lt_scalar(negative, reinterpret_cast<vector_int8_t&>(high_byte), int8_t(0), mask_u8);
                vector_uint8_t high_flip;
                asc_select(high_flip, negative_flip, positive_flip, negative);
                asc_xor(high_byte, high_byte, high_flip, mask_u8);
            }
            asc_cumulative_histogram_bin0(hist_low, high_byte, active);
            asc_cumulative_histogram_bin1(hist_high, high_byte, active);
        } else {
            // No asc_* C-API exposes MODE_MERGING for vxor; keep the raw intrinsic.
            vxor(low_byte, low_byte, negative_flip, negative_prefix, MODE_MERGING);
            vector_bool selected;
            asc_eq(selected, high_byte, raw_prefix, active);
            asc_cumulative_histogram_bin0(hist_low, low_byte, selected);
            asc_cumulative_histogram_bin1(hist_high, low_byte, selected);
        }
    }
}

__simd_callee__ inline void topk_accumulate_nan_count(__ubuf__ bfloat16_t* values, uint32_t count,
                                                       __ubuf__ uint32_t* nan_count,
                                                       bool initialize) {
    constexpr uint32_t kValuesPerRound = 128;
    vector_bool mask_u16 = asc_create_mask_b16(PAT_ALL);
    vector_bool all_not_nan = mask_u16;
    uint32_t remaining = count;
    auto* values_ptr = values;
    for (uint32_t offset = 0; offset < count; offset += kValuesPerRound) {
        vector_bool active = asc_update_mask_b16(remaining);
        vector_bfloat16_t value;
        asc_loadalign_postupdate(value, values_ptr, static_cast<int32_t>(kValuesPerRound));
        vector_bool not_nan;
        asc_eq(not_nan, value, value, active);
        vector_bool round_not_nan;
        asc_select(round_not_nan, not_nan, mask_u16, active);
        asc_select(all_not_nan, round_not_nan, all_not_nan, all_not_nan);
    }
    vector_bool has_nan;
    asc_not(has_nan, all_not_nan, mask_u16);
    vector_uint16_t nan_as_u16;
    asc_duplicate_scalar(nan_as_u16, uint16_t(1), has_nan);
    vector_uint32_t current_nan_count;
    asc_reduce_sum(current_nan_count, nan_as_u16, mask_u16);
    asc_duplicate(current_nan_count, current_nan_count, asc_create_mask_b32(PAT_ALL));
    vector_uint32_t accumulated;
    if (initialize)
        asc_duplicate_scalar(accumulated, uint32_t(0), asc_create_mask_b32(PAT_ALL));
    else
        asc_loadalign_brc(accumulated, nan_count);
    asc_add(accumulated, accumulated, current_nan_count, asc_create_mask_b32(PAT_ALL));
    asc_storealign(nan_count, accumulated, asc_create_mask_b32(PAT_ALL));
}

// Find the first bucket whose cumulative histogram count reaches rank, then broadcast its index.
[[maybe_unused]]
__simd_callee__ inline void topk_find_bucket(vector_uint16_t& bucket, vector_uint16_t hist_low,
                                             vector_uint16_t hist_high, vector_uint16_t rank) {
    vector_bool mask = asc_create_mask_b16(PAT_ALL);
    vector_bool found_low, found_high;
    asc_ge(found_low, hist_low, rank, mask);
    asc_ge(found_high, hist_high, rank, mask);

    vector_int16_t index_low_s16, index_high_s16;
    asc_arange(index_low_s16, int16_t(0));
    asc_arange(index_high_s16, int16_t(128));
    vector_uint16_t not_found, candidate_low, candidate_high, candidate;
    asc_duplicate_scalar(not_found, uint16_t(0xffff), mask);
    asc_select(candidate_low, reinterpret_cast<vector_uint16_t&>(index_low_s16), not_found, found_low);
    asc_select(candidate_high, reinterpret_cast<vector_uint16_t&>(index_high_s16), not_found, found_high);
    asc_min(candidate, candidate_low, candidate_high, mask);
    asc_reduce_min(bucket, candidate, mask);
    asc_duplicate(bucket, bucket, mask);
}

// Reconstruct the position within the input row from the split virtual index, then apply
// output_idx_offset to produce the final output index:
// x = index_low | (index_high << 16)
// input_row_index = shuffle_page_ids[x >> shuffle_page_size_bits] * 128
//                   + x % shuffle_page_size
// output_index = input_row_index + output_idx_offset
template <typename Config>
__simd_callee__ inline void topk_restore_index(vector_uint16_t index_low,
                                                vector_uint16_t index_high,
                                                __ubuf__ uint16_t* shuffle_page_ids,
                                                __ubuf__ int32_t* destination, uint32_t offset,
                                                int32_t output_idx_offset, vector_bool mask) {
    constexpr uint32_t kShufflePageSizeBits = Config::shuffle_page_size_bits;
    constexpr int16_t kPageIdHighShift = 16 - kShufflePageSizeBits;
    vector_uint16_t page_in_low, page_from_high, virtual_page_id, physical_page_id;
    asc_shiftright_scalar(page_in_low, index_low, static_cast<int16_t>(kShufflePageSizeBits), mask);
    asc_shiftleft_scalar(page_from_high, index_high, kPageIdHighShift, mask);
    asc_or(virtual_page_id, page_from_high, page_in_low, mask);
    asc_gather(physical_page_id, shuffle_page_ids, virtual_page_id, mask);
    asc_shiftleft_scalar(index_low, index_low, kPageIdHighShift, mask);
    asc_shiftright_scalar(index_low, index_low, kPageIdHighShift, mask);

    vector_uint16_t zero, offset0, offset1, base0, base1;
    asc_duplicate_scalar(zero, uint16_t(0), mask);
    asc_intlv(offset0, offset1, index_low, zero);
    asc_intlv(base0, base1, physical_page_id, zero);

    vector_uint32_t result0, result1;
    asc_shiftleft_scalar(result0, reinterpret_cast<vector_uint32_t&>(base0), int16_t(7), asc_create_mask_b32(PAT_ALL));
    asc_shiftleft_scalar(result1, reinterpret_cast<vector_uint32_t&>(base1), int16_t(7), asc_create_mask_b32(PAT_ALL));
    asc_add(result0, result0, reinterpret_cast<vector_uint32_t&>(offset0), asc_create_mask_b32(PAT_ALL));
    asc_add(result1, result1, reinterpret_cast<vector_uint32_t&>(offset1), asc_create_mask_b32(PAT_ALL));
    vector_int32_t index_offset;
    asc_duplicate_scalar(index_offset, output_idx_offset, asc_create_mask_b32(PAT_ALL));
    asc_add(reinterpret_cast<vector_int32_t&>(result0), reinterpret_cast<vector_int32_t&>(result0),
         index_offset, asc_create_mask_b32(PAT_ALL));
    asc_add(reinterpret_cast<vector_int32_t&>(result1), reinterpret_cast<vector_int32_t&>(result1),
         index_offset, asc_create_mask_b32(PAT_ALL));
    vector_bool mask0, mask1;
    asc_intlv_b16(mask0, mask1, mask, asc_create_mask_b16(PAT_ALLF));
    asc_storealign(destination + offset, reinterpret_cast<vector_int32_t&>(result0), mask0);
    asc_storealign(destination + offset + 64, reinterpret_cast<vector_int32_t&>(result1), mask1);
}

template <typename Config>
__simd_vf__ inline void topk_prepare_index(__ubuf__ uint16_t* index_low,
                                           __ubuf__ uint16_t* index_high,
                                           __ubuf__ uint16_t* shuffle_page_ids,
                                            __ubuf__ int32_t* prepared_index,
                                            __ubuf__ int32_t* destination, uint32_t topk,
                                            uint32_t valid_count, bool restore_index,
                                            int32_t output_idx_offset) {
    constexpr uint32_t kValuesPerReg = 128;
    constexpr uint32_t kIndicesPerReg = 64;
    if (restore_index) {
        uint32_t remaining = topk;
        auto* index_low_ptr = index_low;
        auto* index_high_ptr = index_high;
        for (uint32_t offset = 0; offset < topk; offset += kValuesPerReg) {
            vector_bool active = asc_update_mask_b16(remaining);
            vector_uint16_t low, high;
            asc_loadalign_postupdate(low, index_low_ptr, static_cast<int32_t>(kValuesPerReg));
            asc_loadalign_postupdate(high, index_high_ptr, static_cast<int32_t>(kValuesPerReg));
            topk_restore_index<Config>(low, high, shuffle_page_ids, destination, offset,
                                       output_idx_offset, active);
        }
    } else {
        uint32_t remaining = topk;
        uint32_t valid_remaining = valid_count;
        auto* prepared_index_ptr = prepared_index;
        for (uint32_t offset = 0; offset < topk; offset += kIndicesPerReg) {
            vector_bool active = asc_update_mask_b32(remaining);
            vector_bool valid = asc_update_mask_b32(valid_remaining);
            vector_int32_t index;
            asc_loadalign_postupdate(index, prepared_index_ptr,
                                     static_cast<int32_t>(kIndicesPerReg));
            vector_int32_t index_offset;
            asc_duplicate_scalar(index_offset, output_idx_offset, active);
            // No asc_* C-API exposes MODE_MERGING for vadd; keep the raw intrinsic.
            vadd(index, index, index_offset, valid, MODE_MERGING);
            asc_storealign(destination + offset, index, active);
        }
    }
}

// Accumulate every value that has entered the buffer into the high-byte histogram without
// clearing it, which still yields the correct high-byte bucket. On each rebuild, only newly
// added values in the filtered buffer are accumulated. Clamp counters to avoid overflow.
[[maybe_unused]]
__simd_vf__ inline void topk_build_high_histogram(
    __ubuf__ bfloat16_t* values, uint32_t count, __ubuf__ uint16_t* histogram, bool initialize,
    uint32_t keep_count, __ubuf__ uint32_t* nan_count) {
    vector_bool mask_u16 = asc_create_mask_b16(PAT_ALL);
    vector_uint16_t hist_low, hist_high;
    if (initialize) {
        asc_duplicate_scalar(hist_low, uint16_t(0), mask_u16);
        asc_duplicate_scalar(hist_high, uint16_t(0), mask_u16);
    } else {
        asc_loadalign(hist_low, histogram);
        asc_loadalign(hist_high, histogram + 128);
    }
    vector_uint8_t unused_prefix;
    topk_accumulate_histogram<true, true>(hist_low, hist_high, values, count, unused_prefix);
    topk_accumulate_nan_count(values, count, nan_count, initialize);
    vector_uint16_t counter_limit;
    asc_duplicate_scalar(counter_limit, static_cast<uint16_t>(keep_count + 1), mask_u16);
    asc_min(hist_low, hist_low, counter_limit, mask_u16);
    asc_min(hist_high, hist_high, counter_limit, mask_u16);
    asc_storealign(histogram, hist_low, mask_u16);
    asc_storealign(histogram + 128, hist_high, mask_u16);
}

[[maybe_unused]]
__simd_vf__ inline void topk_get_sorted_value(
    __ubuf__ bfloat16_t* destination_value, __ubuf__ bfloat16_t* source_value,
    __ubuf__ uint32_t* permutation, uint32_t topk) {
    constexpr uint32_t kValuesPerReg = 64;
    uint32_t remaining = topk;
    auto* permutation_ptr = permutation;
    for (uint32_t offset = 0; offset < topk; offset += kValuesPerReg) {
        uint32_t count = min(remaining, kValuesPerReg);
        remaining -= count;
        vector_bool mask = asc_update_mask_b16(count);
        vector_uint32_t position_u32;
        vector_uint16_t position;
        asc_loadalign_postupdate(position_u32, permutation_ptr,
                                 static_cast<int32_t>(kValuesPerReg));
        asc_pack_to_low(position, position_u32);
        vector_bfloat16_t value;
        asc_gather(value, source_value, position, mask);
        asc_storealign(destination_value + offset, value, mask);
    }
}

// During warmup, select the initial top-k candidates and establish the threshold.
// In subsequent scan rounds, append values satisfying !(value <= threshold) to the filtered
// buffer. This deliberately selects every NaN so it is retained and detected.
// When the filtered buffer exceeds the safe trigger or on the final round, reduce it to k
// candidates and update the threshold. Split virtual indices are carried with the candidates;
// topk_prepare_index restores the real indices later.
template <typename Config, bool kIsWarmup, bool kGenerateIndex = false,
          uint32_t kGeneratedIndexCapacity = 0, bool kFinalize = false>
__simd_vf__ inline void topk_select_candidates(
    __ubuf__ bfloat16_t* source_value, __ubuf__ uint16_t* source_index_low,
    __ubuf__ uint16_t* source_index_high, uint32_t source_count,
    __ubuf__ bfloat16_t* destination_value, __ubuf__ uint16_t* destination_index_low,
    __ubuf__ uint16_t* destination_index_high, __ubuf__ uint16_t* temporary_position,
    __ubuf__ uint16_t* histogram, __ubuf__ uint16_t* high_histogram, __ubuf__ uint16_t* threshold,
    uint32_t keep_count, __ubuf__ uint32_t* nan_count,
    __ubuf__ uint32_t* filtered_count = nullptr, uint32_t safe_trigger = 0,
    __ubuf__ bfloat16_t* filter_source = nullptr, uint32_t filter_source_count = 0,
    uint32_t filter_index_base = 0) {
    if constexpr (!kIsWarmup) {
        source_count = filtered_count[0];
    }
    // A scan round first appends the low indices (the low 2 bytes of each index) of values above
    // the threshold to the filtered buffer. All values in the same block share one high index, so
    // gather their values by low index, fill in that high index, and pad the result to 32B for
    // subsequent vector loads/stores.
    if constexpr (!kIsWarmup) {
        constexpr uint32_t kFilterValuesPerReg = 128;
        const uint32_t destination_count = source_count;
        vector_bool filter_mask_u16 = asc_create_mask_b16(PAT_ALL);
        auto* index_low_output = destination_index_low + destination_count;
        vector_store_unalign index_low_state;
        asc_squeeze_and_storeunalign_init();
        vector_uint16_t threshold_bits;
        asc_loadalign_brc(threshold_bits, threshold);
        const uint32_t virtual_index_base = filter_index_base;
        auto* filter_source_ptr = filter_source;

        vector_uint16_t index_low, packed_low;
        for (uint32_t offset = 0; offset < filter_source_count; offset += kFilterValuesPerReg) {
            vector_bfloat16_t value;
            asc_loadalign_postupdate(value, filter_source_ptr,
                                     static_cast<int32_t>(kFilterValuesPerReg));
            vector_bool selected;
            vector_bool less_equal;
            asc_le(less_equal, value, reinterpret_cast<vector_bfloat16_t&>(threshold_bits),
                    filter_mask_u16);
            asc_not(selected, less_equal, filter_mask_u16);
            asc_arange(reinterpret_cast<vector_int16_t&>(index_low),
                static_cast<int16_t>(virtual_index_base + offset));
            asc_squeeze_and_storeunalign(index_low_output, packed_low, index_low_state, index_low,
                                          selected);
        }
        asc_squeeze_and_storeunalign_finalize(index_low_output, index_low_state);
        auto* count_scratch = reinterpret_cast<__ubuf__ uint32_t*>(histogram);
        // asc_get_squeeze_status() is __aicore__-only and cannot be called from a SIMD VF,
        // so the AR status read stays a raw intrinsic.
        sprsts(SPR_AR, count_scratch, 0);
        asc_mem_bar(VST_LD);
        asc_mem_bar(VST_VLD);

        source_count = destination_count + count_scratch[0] / sizeof(uint16_t);
        uint32_t remaining = source_count - destination_count;
        auto* gather_base = filter_source - static_cast<uint16_t>(virtual_index_base);
        vector_uint16_t index_high;
        asc_duplicate_scalar(index_high, static_cast<uint16_t>(virtual_index_base >> 16), filter_mask_u16);

        auto* index_low_in_ptr = destination_index_low + destination_count;
        auto* destination_value_out_ptr = destination_value + destination_count;
        auto* destination_index_high_out_ptr = destination_index_high + destination_count;
        uint32_t offset = destination_count;
        for (; remaining >= kFilterValuesPerReg;
             offset += kFilterValuesPerReg, remaining -= kFilterValuesPerReg) {
            vector_uint16_t current_index_low;
            vector_bfloat16_t gathered_value;
            asc_loadalign_postupdate(current_index_low, index_low_in_ptr,
                                     static_cast<int32_t>(kFilterValuesPerReg));
            asc_gather(gathered_value, gather_base, current_index_low, filter_mask_u16);
            asc_storealign_postupdate(destination_value_out_ptr, gathered_value,
                                      static_cast<int32_t>(kFilterValuesPerReg), filter_mask_u16);
            asc_storealign_postupdate(destination_index_high_out_ptr, index_high,
                                      static_cast<int32_t>(kFilterValuesPerReg), filter_mask_u16);
        }
        if (remaining != 0) {
            uint32_t aligned_remaining = aligned(remaining, VST_VLD_ALIGN_ELEMENTS);
            vector_bool tail = asc_update_mask_b16(remaining);
            vector_bool aligned_tail = asc_update_mask_b16(aligned_remaining);
            vector_uint16_t current_index_low;
            vector_bfloat16_t gathered_value;
            asc_loadalign(current_index_low, destination_index_low + offset);
            asc_gather(gathered_value, gather_base, current_index_low, tail);
            vector_uint16_t padding_value;
            asc_duplicate_scalar(padding_value, uint16_t(0xff80), filter_mask_u16);
            asc_select(reinterpret_cast<vector_uint16_t&>(gathered_value),
                 reinterpret_cast<vector_uint16_t&>(gathered_value), padding_value, tail);
            asc_storealign(destination_value + offset, gathered_value, aligned_tail);
            asc_storealign(destination_index_high + offset, index_high, tail);
            source_count = aligned(source_count, VST_VLD_ALIGN_ELEMENTS);
        }
    }
    if (kIsWarmup || source_count > safe_trigger || kFinalize) {
        if constexpr (!kIsWarmup)
            asc_mem_bar(VST_VLD);
        vector_bool mask_u16 = asc_create_mask_b16(PAT_ALL);
        vector_uint16_t hist_low, hist_high;
        asc_duplicate_scalar(hist_low, uint16_t(0), mask_u16);
        asc_duplicate_scalar(hist_high, uint16_t(0), mask_u16);
        // For radix selection, maintain a clamped high-byte histogram of all values scanned so far.
        // Each rebuild adds only the newly appended filtered values; values discarded from the
        // candidate buffer are intentionally not removed. This historical histogram determines the
        // global high-byte threshold bucket.
        asc_loadalign(hist_low, high_histogram);
        asc_loadalign(hist_high, high_histogram + TOPK_HISTOGRAM_BINS / 2);
        if constexpr (!kIsWarmup) {
            const uint32_t histogram_begin = aligned(keep_count, VST_VLD_ALIGN_ELEMENTS);
            if (source_count > histogram_begin) {
                vector_uint8_t unused_prefix;
                topk_accumulate_histogram<true, true>(
                    hist_low, hist_high, source_value + histogram_begin,
                    source_count - histogram_begin, unused_prefix);
                topk_accumulate_nan_count(source_value + histogram_begin,
                                          source_count - histogram_begin, nan_count, false);
                vector_uint16_t counter_limit;
                asc_duplicate_scalar(counter_limit, static_cast<uint16_t>(keep_count + 1), mask_u16);
                asc_min(hist_low, hist_low, counter_limit, mask_u16);
                asc_min(hist_high, hist_high, counter_limit, mask_u16);
                asc_storealign(high_histogram, hist_low, mask_u16);
                asc_storealign(high_histogram + TOPK_HISTOGRAM_BINS / 2, hist_high, mask_u16);
            }
        }

        // Radix-select the BF16 threshold from the high- and low-byte histograms, then compute how
        // many values equal to it must be kept. This runs only once per rebuild, so these SIMD
        // operations are not aggressively optimized.
        vector_uint16_t rank, high_bucket;
        asc_duplicate_scalar(rank, static_cast<uint16_t>(keep_count), mask_u16);
        topk_find_bucket(high_bucket, hist_low, hist_high, rank);

        vector_uint16_t previous_count, zero;
        asc_duplicate_scalar(zero, uint16_t(0), mask_u16);
        vector_uint16_t one, lane_mask, scratch, bucket_lane;
        asc_duplicate_scalar(one, uint16_t(1), mask_u16);
        asc_duplicate_scalar(lane_mask, uint16_t(127), mask_u16);
        // Locate high_bucket - 1 in the two 128-lane histogram vectors.
        asc_add_scalar(bucket_lane, high_bucket, uint16_t(-1), mask_u16);
        asc_and(bucket_lane, bucket_lane, lane_mask, mask_u16);
        vector_bool bucket_condition;
        // Select its half and extract the cumulative count from its lane.
        asc_le_scalar(bucket_condition, high_bucket, uint16_t(128), mask_u16);
        asc_select(scratch, hist_low, hist_high, bucket_condition);
        asc_gather(previous_count, scratch, bucket_lane);
        // previous_count = high_bucket == 0 ? 0 : histogram[high_bucket - 1]
        asc_eq_scalar(bucket_condition, high_bucket, uint16_t(0), mask_u16);
        asc_select(previous_count, zero, previous_count, bucket_condition);
        // rank -= previous_count
        asc_sub(rank, rank, previous_count, mask_u16);
        // high_prefix = 0xff - high_bucket
        asc_duplicate_scalar(scratch, uint16_t(0xff), mask_u16);
        asc_sub(high_bucket, scratch, high_bucket, mask_u16);

        vector_uint8_t high_prefix, unused_high_byte;
        asc_deintlv(high_prefix, unused_high_byte, reinterpret_cast<vector_uint8_t&>(high_bucket),
                reinterpret_cast<vector_uint8_t&>(high_bucket));
        asc_duplicate_scalar(hist_low, uint16_t(0), mask_u16);
        asc_duplicate_scalar(hist_high, uint16_t(0), mask_u16);
        topk_accumulate_histogram<false>(hist_low, hist_high, source_value, source_count,
                                         high_prefix);

        vector_uint16_t low_bucket, threshold_key;
        vector_uint16_t high_bucket_count;
        // low_rank = histogram[255] - rank + 1
        // low_bucket = first b where histogram[b] >= low_rank
        asc_duplicate_highest(high_bucket_count, hist_high, mask_u16);
        asc_sub(rank, high_bucket_count, rank, mask_u16);
        asc_add(rank, rank, one, mask_u16);
        topk_find_bucket(low_bucket, hist_low, hist_high, rank);
        vector_uint16_t pivot_low_count, greater_count, equal_limit;
        // greater_count = previous_count + histogram[255] - histogram[low_bucket]
        // equal_limit = keep_count - greater_count
        asc_and(bucket_lane, low_bucket, lane_mask, mask_u16);
        asc_lt_scalar(bucket_condition, low_bucket, uint16_t(128), mask_u16);
        asc_select(scratch, hist_low, hist_high, bucket_condition);
        asc_gather(pivot_low_count, scratch, bucket_lane);

        asc_sub(rank, high_bucket_count, pivot_low_count, mask_u16);
        asc_add(greater_count, previous_count, rank, mask_u16);
        asc_duplicate_scalar(rank, static_cast<uint16_t>(keep_count), mask_u16);
        asc_sub(equal_limit, rank, greater_count, mask_u16);

        // threshold_key = (high_bucket << 8) | low_bucket
        // threshold_key ^= high_bucket >= 0x80 ? 0x8000 : 0xffff
        asc_shiftleft_scalar(threshold_key, high_bucket, int16_t(8), mask_u16);
        asc_or(threshold_key, threshold_key, low_bucket, mask_u16);
        asc_ge_scalar(bucket_condition, high_bucket, uint16_t(0x80), mask_u16);
        asc_duplicate_scalar(lane_mask, uint16_t(0x8000), mask_u16);
        asc_duplicate_scalar(scratch, uint16_t(0xffff), mask_u16);
        asc_select(scratch, lane_mask, scratch, bucket_condition);
        asc_xor(threshold_key, threshold_key, scratch, mask_u16);
        asc_storealign(threshold, threshold_key, asc_create_mask_b16(PAT_VL1));

        constexpr uint32_t kValuesPerReg = 128;
        vector_bool threshold_negative;
        asc_lt_scalar(threshold_negative, reinterpret_cast<vector_int16_t&>(threshold_key), int16_t(0),
                 mask_u16);
        vector_uint16_t compare_mask, negative_compare_mask;
        asc_duplicate_scalar(compare_mask, uint16_t(0), mask_u16);
        asc_duplicate_scalar(negative_compare_mask, uint16_t(0x7fff), mask_u16);
        asc_select(compare_mask, negative_compare_mask, compare_mask, threshold_negative);
        vector_uint16_t threshold_compare;
        asc_xor(threshold_compare, threshold_key, compare_mask, mask_u16);

        // Collect the positions of k candidates from the top-k/filtered buffer: all values above
        // the threshold and the required number equal to it. These positions are then used to
        // gather the values and their low/high indices.
        vector_uint16_t equal_seen = zero;
        auto* position_output = temporary_position;
        vector_store_unalign position_state;
        asc_squeeze_and_storeunalign_init();
        uint32_t remaining = source_count;
        auto* source_value_ptr = source_value;
        vector_uint16_t position;
        asc_arange(reinterpret_cast<vector_int16_t&>(position), static_cast<int16_t>(0));
        for (uint32_t offset = 0; offset < source_count; offset += kValuesPerReg) {
            vector_bool active = asc_update_mask_b16(remaining);
            vector_bfloat16_t value;
            asc_loadalign_postupdate(value, source_value_ptr, static_cast<int32_t>(kValuesPerReg));
            vector_uint16_t value_compare;
            // Compare value > threshold (with +0 > -0). Let mask = threshold >= 0 ? 0 : 0x7fff,
            // value_compare = reinterpret<int16>(value) ^ mask, and
            // threshold_compare = reinterpret<int16>(threshold) ^ mask. For finite BF16 values,
            // value > threshold iff value_compare > threshold_compare, with +0 ordered above -0.
            asc_xor(value_compare, reinterpret_cast<vector_uint16_t&>(value), compare_mask, active);
            vector_bool greater, equal;
            asc_gt(greater, reinterpret_cast<vector_int16_t&>(value_compare),
                    reinterpret_cast<vector_int16_t&>(threshold_compare), active);
            asc_eq(equal, reinterpret_cast<vector_int16_t&>(value_compare),
                    reinterpret_cast<vector_int16_t&>(threshold_compare), active);
            vector_uint16_t equal_prefix = zero;
            asc_unsqueeze(equal_prefix, equal);
            // No asc_* C-API exposes MODE_MERGING for vadd; keep the raw intrinsic.
            vadd(equal_prefix, equal_prefix, one, equal, MODE_MERGING);
            asc_add(equal_prefix, equal_prefix, equal_seen, mask_u16);
            vector_bool take_equal;
            asc_ge(take_equal, equal_limit, equal_prefix, equal);
            vector_bool selected;
            asc_or(selected, greater, take_equal, active);
            vector_uint16_t packed_position;
            asc_squeeze_and_storeunalign(position_output, packed_position, position_state, position,
                                          selected);
            asc_duplicate_highest(equal_seen, equal_prefix, mask_u16);
            asc_add_scalar(position, position, static_cast<uint16_t>(kValuesPerReg), mask_u16);
        }
        asc_squeeze_and_storeunalign_finalize(position_output, position_state);
        asc_mem_bar(VST_VLD);

        remaining = keep_count;
        vector_uint16_t index_high;
        if constexpr (kGenerateIndex) {
            static_assert(kGeneratedIndexCapacity <= (uint32_t(1) << 16),
                          "generated virtual index high must be zero");
            asc_duplicate_scalar(index_high, uint16_t(0), mask_u16);
        }
        auto* temporary_position_ptr = temporary_position;
        auto* keep_destination_value_ptr = destination_value;
        auto* keep_destination_index_low_ptr = destination_index_low;
        auto* keep_destination_index_high_ptr = destination_index_high;
        for (uint32_t offset = 0; offset < keep_count; offset += kValuesPerReg) {
            vector_bool mask = asc_update_mask_b16(remaining);
            vector_uint16_t index_low;
            asc_loadalign_postupdate(index_low, temporary_position_ptr,
                                     static_cast<int32_t>(kValuesPerReg));
            vector_bfloat16_t value;
            asc_gather(value, source_value, index_low, mask);
            if (offset + kValuesPerReg > keep_count) {
                vector_int16_t lane;
                asc_arange(lane, static_cast<int16_t>(offset));
                vector_bool out_of_bounds;
                asc_ge_scalar(out_of_bounds, lane, static_cast<int16_t>(keep_count), mask_u16);
                vector_uint16_t padding;
                asc_duplicate_scalar(padding, uint16_t(0x7fff), mask_u16);
                asc_select(reinterpret_cast<vector_uint16_t&>(value), padding,
                     reinterpret_cast<vector_uint16_t&>(value), out_of_bounds);
            }
            asc_storealign_postupdate(keep_destination_value_ptr, value,
                                      static_cast<int32_t>(kValuesPerReg), mask_u16);
            if constexpr (!kGenerateIndex) {
                asc_gather(index_high, source_index_high, index_low, mask);
                asc_gather(index_low, source_index_low, index_low, mask);
            }
            asc_storealign_postupdate(keep_destination_index_low_ptr, index_low,
                                      static_cast<int32_t>(kValuesPerReg), mask);
            asc_storealign_postupdate(keep_destination_index_high_ptr, index_high,
                                      static_cast<int32_t>(kValuesPerReg), mask);
        }
        const uint32_t aligned_keep_count = aligned(keep_count, VST_VLD_ALIGN_ELEMENTS);
        const uint32_t aligned_offset =
            keep_count / VST_VLD_ALIGN_ELEMENTS * VST_VLD_ALIGN_ELEMENTS;
        uint32_t valid_count = keep_count - aligned_offset;
        uint32_t store_count = aligned_keep_count - aligned_offset;
        vector_bool valid_mask = asc_update_mask_b16(valid_count);
        vector_bool store_mask = asc_update_mask_b16(store_count);
        vector_uint16_t value, padding_value;
        asc_mem_bar(VST_VLD);
        asc_loadalign(value, reinterpret_cast<__ubuf__ uint16_t*>(destination_value) + aligned_offset);
        asc_duplicate_scalar(padding_value, uint16_t(0xffff), asc_create_mask_b16(PAT_ALL));
        asc_select(value, value, padding_value, valid_mask);
        asc_storealign(reinterpret_cast<__ubuf__ uint16_t*>(destination_value) + aligned_offset, value, store_mask);
        if constexpr (!kIsWarmup)
            source_count = aligned_keep_count;
    }
    if constexpr (!kIsWarmup)
        filtered_count[0] = source_count;
}

// Prepare the output directly when the number of valid elements does not exceed top-k.
template <bool kReturnValue>
__simd_vf__ inline void topk_prepare_short_row(__ubuf__ bfloat16_t* source,
                                               __ubuf__ bfloat16_t* dst_value,
                                               __ubuf__ int32_t* dst_index, uint32_t valid_count,
                                               uint32_t topk, uint16_t value_fill_bits,
                                                int32_t index_fill,
                                                __ubuf__ uint32_t* nan_count) {
    constexpr uint32_t kValuesPerReg = 128;
    constexpr uint32_t kIndicesPerReg = 64;
    topk_accumulate_nan_count(source, valid_count, nan_count, true);
    if constexpr (kReturnValue) {
        vector_bool mask = asc_create_mask_b16(PAT_ALL);
        auto* source_ptr = source;
        auto* dst_value_out_ptr = dst_value;
        for (uint32_t offset = 0; offset < topk; offset += kValuesPerReg) {
            uint32_t store_count = min(topk - offset, kValuesPerReg);
            vector_bool store_mask = asc_update_mask_b16(store_count);
            vector_bfloat16_t value;
            if (offset < valid_count) {
                uint32_t valid = min(valid_count - offset, kValuesPerReg);
                const bool needs_fill = valid < kValuesPerReg;
                vector_bool valid_mask = asc_update_mask_b16(valid);
                asc_loadalign_postupdate(value, source_ptr, static_cast<int32_t>(kValuesPerReg));
                if (needs_fill) {
                    vector_uint16_t fill;
                    asc_duplicate_scalar(fill, value_fill_bits, mask);
                    asc_select(reinterpret_cast<vector_uint16_t&>(value),
                         reinterpret_cast<vector_uint16_t&>(value), fill, valid_mask);
                }
            } else {
                asc_duplicate_scalar(reinterpret_cast<vector_uint16_t&>(value), value_fill_bits, mask);
            }
            asc_storealign_postupdate(dst_value_out_ptr, value,
                                      static_cast<int32_t>(kValuesPerReg), store_mask);
        }
    }

    vector_bool mask = asc_create_mask_b32(PAT_ALL);
    auto* dst_index_out_ptr = dst_index;
    for (uint32_t offset = 0; offset < topk; offset += kIndicesPerReg) {
        uint32_t store_count = min(topk - offset, kIndicesPerReg);
        vector_bool store_mask = asc_update_mask_b32(store_count);
        vector_int32_t index;
        if (offset < valid_count) {
            uint32_t valid = min(valid_count - offset, kIndicesPerReg);
            const bool needs_fill = valid < kIndicesPerReg;
            vector_bool valid_mask = asc_update_mask_b32(valid);
            asc_arange(index, static_cast<int32_t>(offset));
            if (needs_fill) {
                vector_int32_t fill;
                asc_duplicate_scalar(fill, index_fill, mask);
                asc_select(index, index, fill, valid_mask);
            }
        } else {
            asc_duplicate_scalar(index, index_fill, mask);
        }
        asc_storealign_postupdate(dst_index_out_ptr, index,
                                  static_cast<int32_t>(kIndicesPerReg), store_mask);
    }
}

struct TopkOutputStore {
    template <typename output_idx_t>
    __aicore__ inline void operator()(__gm__ output_idx_t* destination,
                                      __ubuf__ output_idx_t* source, uint32_t topk) const {
        copy_ubuf_to_gm_align_v2(
            destination, source, 0, 1, topk * sizeof(output_idx_t),
            static_cast<uint8_t>(ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV),
            topk * sizeof(output_idx_t), topk * sizeof(output_idx_t));
    }
};

// The FP32 merge sorter operates on complete 32-index groups.
__simd_vf__ inline void topk_pad_index_sort(__ubuf__ int32_t* tail, uint32_t valid) {
    vector_int32_t index, padding;
    asc_loadalign(index, tail);
    asc_duplicate_scalar(padding, int32_t(MAX_VOCAB_SIZE), asc_create_mask_b32(PAT_ALL));
    vector_bool keep = asc_update_mask_b32(valid);
    asc_select(index, index, padding, keep);
    uint32_t group = 32;
    asc_storealign(tail, index, asc_update_mask_b32(group));
}

// The selected physical indices are distinct. Mark their positions in the released
// selection scratch, then emit them in ascending order directly into the index buffer.
__simd_vf__ inline void topk_compact_sorted_indices(__ubuf__ int32_t* selected, uint32_t topk,
                                                   uint32_t valid_len) {
    constexpr uint32_t kLanes = 64;
    auto* flags = reinterpret_cast<__ubuf__ uint32_t*>(0);
    const auto full = pset_b32(PAT_ALL);
    vector_uint32_t zero, one, indices, flag;
    vdup(zero, 0u, full, MODE_ZEROING);
    vdup(one, 1u, full, MODE_ZEROING);
    const uint16_t flag_chunks = ceil_div(valid_len, kLanes);
    mem_bar(VLD_VST);
    for (uint16_t chunk = 0; chunk < flag_chunks; ++chunk) {
        vsts(zero, flags, chunk * kLanes, NORM_B32, full);
    }
    mem_bar(VST_VST);
    mem_bar(VST_VLD);
    for (uint16_t chunk = 0; chunk < ceil_div(topk, kLanes); ++chunk) {
        uint32_t remaining = topk - chunk * kLanes;
        const auto active = plt_b32(remaining, POST_UPDATE);
        vlds(indices, reinterpret_cast<__ubuf__ uint32_t*>(selected), chunk * kLanes, NORM);
        vscatter(one, flags, indices, active);
    }
    mem_bar(VST_VLD);
    vector_store_unalign state;
    vector_int32_t positions, packed;
    asc_squeeze_and_storeunalign_init();
    for (uint16_t chunk = 0; chunk < flag_chunks; ++chunk) {
        vector_bool present;
        vlds(flag, flags, chunk * kLanes, NORM);
        vcmp_eq(present, flag, one, full);
        asc_arange(positions, static_cast<int32_t>(chunk * kLanes));
        asc_squeeze_and_storeunalign(selected, packed, state, positions, present);
    }
    asc_squeeze_and_storeunalign_finalize(selected, state);
}

template <typename Config, bool kSortedIndex, typename value_t,
          typename output_idx_t, typename OutputStore = TopkOutputStore>
__aicore__ inline void topk_select_run(uint32_t valid_len, uint32_t topk, __gm__ value_t* row_in,
                                       __gm__ value_t* row_out_value,
                                       __gm__ output_idx_t* row_out_index, bool return_value,
                                       __gm__ output_idx_t* row_output_idx_offset,
                                       output_idx_t idx_oob_fill,
                                       uint16_t value_oob_fill_bits,
                                       OutputStore output_store = {}) {
    using Plan = UnifiedBufferMemoryPlan<Config, kSortedIndex, value_t>;
    constexpr uint32_t kTopkCapacity = Plan::kTopkCapacity;
    constexpr uint32_t kFilteredCapacity = Plan::kFilteredCapacity;
    constexpr uint32_t kWarmupPages = Plan::kWarmupPages;
    constexpr uint32_t kBlockSize = Plan::kBlockSize;
    constexpr uint32_t kShufflePageSize = Plan::kShufflePageSize;
    constexpr uint32_t kNumStages = Plan::kNumStages;
    constexpr uint32_t kWarmupElements = Plan::kWarmupElements;
    constexpr uintptr_t kInputOffset = Plan::kInputOffset;
    constexpr uintptr_t kCandidateValueOffset = Plan::kCandidateValueOffset;
    constexpr uintptr_t kCandidateIndexLowOffset = Plan::kCandidateIndexLowOffset;
    constexpr uintptr_t kCandidateIndexHighOffset = Plan::kCandidateIndexHighOffset;
    constexpr uintptr_t kHistogramOffset = Plan::kHistogramOffset;
    constexpr uintptr_t kHighHistogramOffset = Plan::kHighHistogramOffset;
    constexpr uintptr_t kThresholdOffset = Plan::kThresholdOffset;
    constexpr uintptr_t kStateOffset = Plan::kStateOffset;
    constexpr uintptr_t kTemporaryPositionOffset = Plan::kTemporaryPositionOffset;
    constexpr uintptr_t kShufflePageIdOffset = Plan::kShufflePageIdOffset;
    // Only consulted by `ascendc_assert`, which is a no-op outside the device pass.
    [[maybe_unused]] constexpr uint32_t kShufflePageCapacity = Plan::kShufflePageCapacity;
    constexpr uint32_t kSortTemporaryBytes = Plan::kSortTemporaryBytes;
    constexpr uintptr_t kSortedIndexOffset = Plan::kSortedIndexOffset;
    constexpr uintptr_t kSortPermutationOffset = Plan::kSortPermutationOffset;
    constexpr uintptr_t kSortTemporaryOffset = Plan::kSortTemporaryOffset;

    static_assert(std::is_same_v<value_t, bfloat16_t>);
    static_assert(std::is_same_v<output_idx_t, int32_t>);

    auto* ub_topk_value = reinterpret_cast<__ubuf__ value_t*>(kCandidateValueOffset);
    auto* ub_topk_index_low = reinterpret_cast<__ubuf__ uint16_t*>(kCandidateIndexLowOffset);
    auto* ub_topk_index_high = reinterpret_cast<__ubuf__ uint16_t*>(kCandidateIndexHighOffset);
    auto* ub_warmup = reinterpret_cast<__ubuf__ value_t*>(kCandidateValueOffset);
    auto* ub_input = reinterpret_cast<__ubuf__ value_t*>(kInputOffset);
    auto* ub_histogram = reinterpret_cast<__ubuf__ uint16_t*>(kHistogramOffset);
    auto* ub_high_histogram = reinterpret_cast<__ubuf__ uint16_t*>(kHighHistogramOffset);
    auto* ub_threshold = reinterpret_cast<__ubuf__ uint16_t*>(kThresholdOffset);
    auto* ub_result_count = reinterpret_cast<__ubuf__ uint32_t*>(kStateOffset);
    auto* ub_nan_count = reinterpret_cast<__ubuf__ uint32_t*>(kStateOffset + 32);
    auto* ub_temporary_position = reinterpret_cast<__ubuf__ uint16_t*>(kTemporaryPositionOffset);
    auto* ub_shuffle_page_ids = reinterpret_cast<__ubuf__ uint16_t*>(kShufflePageIdOffset);
    auto* ub_output_index = reinterpret_cast<__ubuf__ int32_t*>(ub_input);

    event_t load_events[kNumStages];
    event_t warmup_events[kWarmupPages];
    for (uint32_t stage = 0; stage < kNumStages; ++stage) {
        load_events[stage] = event_t(stage);
        set_flag(PIPE_V, PIPE_MTE2, load_events[stage]);
    }
    for (uint32_t page = 0; page < kWarmupPages; ++page)
        warmup_events[page] = event_t(page);

    if (valid_len <= topk) {
        if (valid_len != 0) {
            wait_flag(PIPE_V, PIPE_MTE2, load_events[0]);
            copy_gm_to_ubuf_align_v2(ub_input, row_in, 0, 1, valid_len * sizeof(value_t), 0, 0,
                                     false,
                                     static_cast<uint8_t>(
                                         LD_L2CacheType::L2_CACHE_HINT_NOTALLOC_CLEAN),
                                     valid_len * sizeof(value_t),
                                     valid_len * sizeof(value_t));
            set_flag(PIPE_MTE2, PIPE_V, load_events[0]);
            wait_flag(PIPE_MTE2, PIPE_V, load_events[0]);
        }
        ub_output_index = reinterpret_cast<__ubuf__ int32_t*>(ub_topk_index_low);
        if (return_value)
            topk_prepare_short_row<true>(
                ub_input, ub_topk_value, ub_output_index, valid_len, topk, value_oob_fill_bits,
                static_cast<int32_t>(idx_oob_fill), ub_nan_count);
        else
            topk_prepare_short_row<false>(
                ub_input, ub_topk_value, ub_output_index, valid_len, topk, value_oob_fill_bits,
                static_cast<int32_t>(idx_oob_fill), ub_nan_count);
        if (valid_len != 0)
            set_flag(PIPE_V, PIPE_MTE2, load_events[0]);
    } else {
        constexpr uint32_t kPagesPerBlock = kBlockSize / kShufflePageSize;
        const uint32_t num_shuffle_pages = ceil_div(valid_len, kShufflePageSize);
        ascendc_assert(num_shuffle_pages <= kShufflePageCapacity, "topk shuffle page map overflow");
        const uint64_t input_hash = ld_dev(reinterpret_cast<__gm__ uint64_t*>(row_in), int16_t(0));
        const uint64_t permutation_seed = input_hash ^ uint64_t(valid_len);
        const TopkPermGenerator<kWarmupPages, kShufflePageSize> permutation{valid_len,
                                                                           permutation_seed};
        const uint32_t warmup_pages =
            num_shuffle_pages < kWarmupPages ? num_shuffle_pages : kWarmupPages;

        uint32_t warmup_elements = 0;
        bool initialize_histogram = true;
        for (uint32_t logical_page_id = 0; logical_page_id < warmup_pages; ++logical_page_id) {
            const TopkPage physical_page = permutation.get_page_id(logical_page_id);
            const uint32_t page_offset = physical_page.start_offset;
            const uint32_t page_count = physical_page.len;
            auto* warmup_destination = ub_warmup + logical_page_id * kShufflePageSize;
            copy_gm_to_ubuf_align_v2(warmup_destination, row_in + page_offset, 0, 1,
                                     page_count * sizeof(value_t), 0, 0, false,
                                     static_cast<uint8_t>(
                                         LD_L2CacheType::L2_CACHE_HINT_NOTALLOC_CLEAN),
                                     page_count * sizeof(value_t),
                                     page_count * sizeof(value_t));
            set_flag(PIPE_MTE2, PIPE_V, warmup_events[logical_page_id]);
            wait_flag(PIPE_MTE2, PIPE_V, warmup_events[logical_page_id]);
            topk_build_high_histogram(
                warmup_destination, page_count, ub_high_histogram, initialize_histogram, topk,
                ub_nan_count);
            ub_shuffle_page_ids[logical_page_id] = static_cast<uint16_t>(page_offset / 128);
            initialize_histogram = false;
            warmup_elements += page_count;
        }

        const uint32_t stable_pages = num_shuffle_pages - warmup_pages;
        const uint32_t stable_blocks = ceil_div(stable_pages, kPagesPerBlock);
        if (stable_blocks == 0)
            topk_select_candidates<Config, true, true, kWarmupElements, true>(
                ub_warmup, nullptr, nullptr, warmup_elements, ub_topk_value, ub_topk_index_low,
                ub_topk_index_high, ub_temporary_position, ub_histogram, ub_high_histogram,
                ub_threshold, topk, ub_nan_count);
        else
            topk_select_candidates<Config, true, true, kWarmupElements>(
                ub_warmup, nullptr, nullptr, warmup_elements, ub_topk_value, ub_topk_index_low,
                ub_topk_index_high, ub_temporary_position, ub_histogram, ub_high_histogram,
                ub_threshold, topk, ub_nan_count);
        ub_result_count[0] = aligned(topk, VST_VLD_ALIGN_ELEMENTS);
        constexpr uint32_t kSafeTrigger = kTopkCapacity + kFilteredCapacity;
        for (uint32_t stable_idx = 0; stable_idx < stable_blocks; ++stable_idx) {
            const uint32_t stage = stable_idx % kNumStages;
            wait_flag(PIPE_V, PIPE_MTE2, load_events[stage]);
            auto* block_destination = ub_input + stage * kBlockSize;
            uint32_t count = 0;
            for (uint32_t page = 0; page < kPagesPerBlock; ++page) {
                const uint32_t logical_page_id = warmup_pages + stable_idx * kPagesPerBlock + page;
                if (logical_page_id >= num_shuffle_pages)
                    break;
                const TopkPage physical_page = permutation.get_page_id(logical_page_id);
                const uint32_t page_offset = physical_page.start_offset;
                const uint32_t page_count = physical_page.len;
                copy_gm_to_ubuf_align_v2(
                    block_destination + page * kShufflePageSize, row_in + page_offset, 0, 1,
                    page_count * sizeof(value_t), 0, 0, false,
                    static_cast<uint8_t>(LD_L2CacheType::L2_CACHE_HINT_NOTALLOC_CLEAN),
                    page_count * sizeof(value_t), page_count * sizeof(value_t));
                const uint32_t virtual_page_id = (stable_idx + 1) * kPagesPerBlock + page;
                ub_shuffle_page_ids[virtual_page_id] = static_cast<uint16_t>(page_offset / 128);
                count += page_count;
            }
            set_flag(PIPE_MTE2, PIPE_V, load_events[stage]);
            wait_flag(PIPE_MTE2, PIPE_V, load_events[stage]);
            // Keep virtual indices block-aligned so one filtered block never crosses
            // a 64K boundary and can share one index-high value.
            const uint32_t index_base = (stable_idx + 1) * kBlockSize;
            const bool finalize = stable_idx + 1 == stable_blocks;
            if (finalize)
                topk_select_candidates<Config, false, false, 0, true>(
                    ub_topk_value, ub_topk_index_low, ub_topk_index_high, topk, ub_topk_value,
                    ub_topk_index_low, ub_topk_index_high, ub_temporary_position, ub_histogram,
                    ub_high_histogram, ub_threshold, topk, ub_nan_count, ub_result_count,
                    kSafeTrigger, ub_input + stage * kBlockSize, count, index_base);
            else
                topk_select_candidates<Config, false>(
                    ub_topk_value, ub_topk_index_low, ub_topk_index_high, topk, ub_topk_value,
                    ub_topk_index_low, ub_topk_index_high, ub_temporary_position, ub_histogram,
                    ub_high_histogram, ub_threshold, topk, ub_nan_count, ub_result_count,
                    kSafeTrigger, ub_input + stage * kBlockSize, count, index_base);
            set_flag(PIPE_V, PIPE_MTE2, load_events[stage]);
        }
    }

    // Reject the row before publishing attention indices or candidate counts.
    set_flag(PIPE_V, PIPE_S, event_t(0));
    wait_flag(PIPE_V, PIPE_S, event_t(0));
    if (ub_nan_count[0] != 0) {
        printf("[topk_select] NaN detected at batch index %u\n", get_block_idx());
        trap();
    }
    const bool restore_index = valid_len > topk;
    const bool sort_index =
        topk > 1 && valid_len > topk && ceil_div(valid_len, kShufflePageSize) > kWarmupPages;
    const output_idx_t output_idx_offset =
        row_output_idx_offset == nullptr ? 0 : row_output_idx_offset[0];
    auto* output_index = reinterpret_cast<__ubuf__ int32_t*>(kInputOffset);
    topk_prepare_index<Config>(
        ub_topk_index_low, ub_topk_index_high, ub_shuffle_page_ids, ub_output_index, output_index,
        topk, min(valid_len, topk), restore_index, output_idx_offset);

    if constexpr (kSortedIndex) {
        if (sort_index && !return_value && output_idx_offset == 0 && kTopkCapacity >= 2048 &&
            aligned(valid_len, 64u) <= kInputOffset / sizeof(uint32_t)) {
            topk_compact_sorted_indices(output_index, topk, valid_len);
        } else if (sort_index) {
            LocalTensor<int32_t> source_index_tensor(TPosition::VECCALC, kInputOffset,
                                                     kTopkCapacity);
            LocalTensor<int32_t> sorted_index_tensor(TPosition::VECCALC, kSortedIndexOffset,
                                                     kTopkCapacity);
            LocalTensor<uint32_t> permutation_tensor(TPosition::VECCALC, kSortPermutationOffset,
                                                     kTopkCapacity);
            LocalTensor<uint8_t> temporary_tensor(TPosition::VECCALC, kSortTemporaryOffset,
                                                  kSortTemporaryBytes);
            if (return_value) {
                AscendC::Sort<int32_t, true, DEFAULT_SORT_CONFIG>(
                    sorted_index_tensor, permutation_tensor, source_index_tensor, temporary_tensor,
                    topk);
            } else {
                static_assert(MAX_VOCAB_SIZE <= (1u << 24));
                static_assert(kSortTemporaryBytes >= 8 * kTopkCapacity);
                static_assert(kSortedIndexOffset + 8 * kTopkCapacity <= kSortTemporaryOffset);
                LocalTensor<float> source_float_tensor(TPosition::VECCALC, kInputOffset,
                                                        kTopkCapacity);
                LocalTensor<float> sorted_float_tensor(TPosition::VECCALC, kSortedIndexOffset,
                                                        2 * kTopkCapacity);
                const uint32_t sort_count = aligned(topk, 32u);
                if (sort_count != topk) {
                    topk_pad_index_sort(output_index + sort_count - 32, topk % 32);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Cast<float, int32_t>(source_float_tensor, source_index_tensor,
                                             AscendC::RoundMode::CAST_NONE, sort_count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sort<float, true, kIndexMergeSortConfig>(
                    sorted_float_tensor, source_float_tensor, temporary_tensor, sort_count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast<int32_t, float>(sorted_index_tensor, sorted_float_tensor,
                                             AscendC::RoundMode::CAST_RINT, topk);
                AscendC::PipeBarrier<PIPE_V>();
            }

            auto* permutation = reinterpret_cast<__ubuf__ uint32_t*>(kSortPermutationOffset);
            auto* sorted_index = reinterpret_cast<__ubuf__ int32_t*>(kSortedIndexOffset);
            auto* sorted_value = reinterpret_cast<__ubuf__ value_t*>(ub_topk_index_high);
            if (return_value) {
                topk_get_sorted_value(sorted_value, ub_topk_value, permutation, topk);
                ub_topk_value = sorted_value;
            }
            output_index = sorted_index;
        }
    }

    event_t store_event = event_t(0);
    set_flag(PIPE_V, PIPE_MTE3, store_event);
    wait_flag(PIPE_V, PIPE_MTE3, store_event);
    if (return_value) {
        copy_ubuf_to_gm_align_v2(
            row_out_value, ub_topk_value, 0, 1, topk * sizeof(value_t),
            static_cast<uint8_t>(ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV),
            topk * sizeof(value_t), topk * sizeof(value_t));
    }
    output_store(row_out_index, output_index, topk);
    set_flag(PIPE_MTE3, PIPE_V, store_event);
    wait_flag(PIPE_MTE3, PIPE_V, store_event);

    for (uint32_t stage = 0; stage < kNumStages; ++stage)
        wait_flag(PIPE_V, PIPE_MTE2, load_events[stage]);
}


}  // namespace topk_select_ascend

// PyPTO attention and candidate-pool writeback.
namespace {
enum Argument { kScoresArg, kLengthsArg, kIndicesArg, kTableArg, kRequestsArg, kCountsArg, kModeArg, kActiveArg, kSortedArg, kPageRowsArg };
template <typename T>
__aicore__ inline __gm__ T* tensor_data(__gm__ int64_t* args, int slot) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[slot]);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

template <typename T>
__aicore__ inline __gm__ T* tensor_row(__gm__ int64_t* args, int slot, int64_t row) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[slot]);
    return tensor_data<T>(args, slot) + row * tensor->strides[0];
}

__aicore__ inline int32_t read_scalar(__gm__ int32_t* address) {
    dcci(address, cache_line_t::SINGLE_CACHE_LINE);
    return *address;
}

#if defined(__DAV_VEC__)
__simd_vf__ inline void map_indices(__ubuf__ uint32_t* indices, __ubuf__ uint32_t* pages,
                                   uint32_t visible, uint32_t page_rows) {
    vector_uint32_t position, page_index, page, offset, physical, invalid, page_mask;
    vector_bool valid;
    auto all = asc_create_mask_b32(PAT_ALL);
    asc_duplicate_scalar(invalid, uint32_t(-1), all);
    asc_duplicate_scalar(page_mask, page_rows - 1, all);
    const int16_t page_shift = page_rows == 64 ? 6 : 7;
    for (uint16_t chunk = 0; chunk < 8; ++chunk) {
        asc_loadalign(position, indices + chunk * 64);
        asc_lt_scalar(valid, position, visible, all);
        asc_shiftright_scalar(page_index, position, page_shift, all);
        vgather2(page, pages, page_index, valid);
        asc_shiftleft_scalar(physical, page, page_shift, all);
        asc_and(offset, position, page_mask, all);
        asc_add(physical, physical, offset, all);
        asc_select(physical, physical, invalid, valid);
        asc_storealign(indices + chunk * 64, physical, all);
    }
}

__simd_vf__ inline void fill_indices(__ubuf__ int32_t* output, uint32_t count, int32_t value) {
    vector_int32_t filled;
    auto all = asc_create_mask_b32(PAT_ALL);
    asc_duplicate_scalar(filled, value, all);
    for (uint32_t begin = 0; begin < count; begin += 64) {
        uint32_t remaining = count - begin;
        auto valid = asc_update_mask_b32(remaining);
        asc_storealign(output + begin, filled, valid);
    }
}

__aicore__ inline void store_indices(__gm__ int32_t* target, __ubuf__ int32_t* source,
                                    uint32_t count) {
    if (count == 0) return;
    copy_ubuf_to_gm_align_v2(
        target, source, 0, 1, count * sizeof(int32_t),
        static_cast<uint8_t>(ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV),
        count * sizeof(int32_t), count * sizeof(int32_t));
}

__aicore__ inline void store_count(__gm__ int32_t* target, uint32_t count) {
    set_flag(PIPE_V, PIPE_S, event_t(6));
    wait_flag(PIPE_V, PIPE_S, event_t(6));
    auto scratch = reinterpret_cast<__ubuf__ int32_t*>(0);
    fill_indices(scratch, 1, count);
    set_flag(PIPE_V, PIPE_MTE3, event_t(1));
    wait_flag(PIPE_V, PIPE_MTE3, event_t(1));
    store_indices(target, scratch, 1);
    set_flag(PIPE_MTE3, PIPE_V, event_t(1));
    wait_flag(PIPE_MTE3, PIPE_V, event_t(1));
}

__aicore__ inline void clear_pool(__gm__ int32_t* target, int64_t capacity) {
    auto scratch = reinterpret_cast<__ubuf__ int32_t*>(0);
    fill_indices(scratch, 2048, -1);
    set_flag(PIPE_V, PIPE_MTE3, event_t(1));
    wait_flag(PIPE_V, PIPE_MTE3, event_t(1));
    for (int64_t begin = 0; begin < capacity; begin += 2048) {
        uint32_t count = capacity - begin < 2048 ? capacity - begin : 2048;
        store_indices(target + begin, scratch, count);
    }
    set_flag(PIPE_MTE3, PIPE_V, event_t(1));
    wait_flag(PIPE_MTE3, PIPE_V, event_t(1));
}

template <bool kAttentionOutput>
struct DirectOutputStore {
    __gm__ int64_t* args;
    int64_t row;
    uint32_t visible;
    int32_t request;
    __ubuf__ int32_t* prefetched_table;

    __aicore__ inline void operator()(__gm__ int32_t* target, __ubuf__ int32_t* source,
                                      uint32_t topk) const {
        if constexpr (kAttentionOutput) {
            const uint32_t page_rows = args[kPageRowsArg];
            auto table = reinterpret_cast<__gm__ Tensor*>(args[kTableArg]);
            auto table_row = reinterpret_cast<__ubuf__ int32_t*>(0);
            if (prefetched_table != nullptr) {
                table_row = prefetched_table;
                wait_flag(PIPE_MTE2, PIPE_V, event_t(7));
            } else if (visible > 0) {
                table_row = reinterpret_cast<__ubuf__ int32_t*>(0);
                uint32_t pages = (visible + page_rows - 1) >> (page_rows == 64 ? 6 : 7);
                set_flag(PIPE_V, PIPE_MTE2, event_t(7));
                wait_flag(PIPE_V, PIPE_MTE2, event_t(7));
                copy_gm_to_ubuf_align_v2(
                    table_row, tensor_data<int32_t>(args, kTableArg) + int64_t(request) * table->strides[0],
                    0, 1, pages * sizeof(int32_t), 0, 0, false,
                    static_cast<uint8_t>(LD_L2CacheType::L2_CACHE_HINT_NOTALLOC_CLEAN),
                    pages * sizeof(int32_t), pages * sizeof(int32_t));
                set_flag(PIPE_MTE2, PIPE_V, event_t(7));
                wait_flag(PIPE_MTE2, PIPE_V, event_t(7));
            }
            map_indices(reinterpret_cast<__ubuf__ uint32_t*>(source),
                        reinterpret_cast<__ubuf__ uint32_t*>(table_row), visible, page_rows);
            set_flag(PIPE_V, PIPE_MTE3, event_t(7));
            wait_flag(PIPE_V, PIPE_MTE3, event_t(7));
            store_indices(target, source, topk);
        } else {
            auto pool = reinterpret_cast<__gm__ Tensor*>(args[kIndicesArg]);
            store_indices(target, source, topk);
            store_count(tensor_row<int32_t>(args, kCountsArg, row), visible < topk ? visible : topk);
            if (pool->shapes[1] > topk) clear_pool(target + topk, pool->shapes[1] - topk);
        }
    }
};

template <typename Policy, bool kAttentionOutput, bool kSortedIndex>
static __aicore__ __attribute__((noinline)) void select_rows(__gm__ int64_t* args, uint32_t topk) {
    using namespace topk_select_ascend;
    auto scores_tensor = reinterpret_cast<__gm__ Tensor*>(args[kScoresArg]);
    auto indices_tensor = reinterpret_cast<__gm__ Tensor*>(args[kIndicesArg]);
    auto scores = tensor_data<bfloat16_t>(args, kScoresArg);
    auto indices = tensor_data<int32_t>(args, kIndicesArg);
    const int64_t active = args[kActiveArg];
    const int64_t tokens = scores_tensor->shapes[0];
    const int64_t score_stride = scores_tensor->strides[0];
    const int64_t index_stride = indices_tensor->strides[0];
    const int64_t workers = get_block_num(args);
    for (int64_t row = get_block_idx(args); row < tokens; row += workers) {
        int64_t visible = row < active ? read_scalar(tensor_row<int32_t>(args, kLengthsArg, row)) : 0;
        if constexpr (!kAttentionOutput) {
            visible = (visible + 7) / 8;
        }
        int32_t request = -1;
        if constexpr (kAttentionOutput) {
            request = visible > 0 ? read_scalar(tensor_row<int32_t>(args, kRequestsArg, row)) : 0;
        }
        auto row_scores = scores + row * score_stride;
        auto row_indices = indices + row * index_stride;
        AscendC::InitSocState();
        __ubuf__ int32_t* prefetched_table = nullptr;
        if constexpr (kAttentionOutput) {
            using LowerPlan = UnifiedBufferMemoryPlan<typename Policy::LowerConfig, kSortedIndex, bfloat16_t>;
            using UpperPlan = UnifiedBufferMemoryPlan<typename Policy::UpperConfig, kSortedIndex, bfloat16_t>;
            constexpr uintptr_t kTableOffset = LowerPlan::kUbEnd > UpperPlan::kUbEnd
                ? LowerPlan::kUbEnd : UpperPlan::kUbEnd;
            constexpr uint32_t kTableCapacity = (TOPK_UB_LIMIT_BYTES - kTableOffset) / sizeof(int32_t);
            const uint32_t page_rows = args[kPageRowsArg];
            uint32_t pages = (visible + page_rows - 1) >> (page_rows == 64 ? 6 : 7);
            if (pages > 0 && pages <= kTableCapacity) {
                auto table = reinterpret_cast<__gm__ Tensor*>(args[kTableArg]);
                prefetched_table = reinterpret_cast<__ubuf__ int32_t*>(kTableOffset);
                copy_gm_to_ubuf_align_v2(
                    prefetched_table,
                    tensor_data<int32_t>(args, kTableArg) + int64_t(request) * table->strides[0],
                    0, 1, pages * sizeof(int32_t), 0, 0, false,
                    static_cast<uint8_t>(LD_L2CacheType::L2_CACHE_HINT_NOTALLOC_CLEAN),
                    pages * sizeof(int32_t), pages * sizeof(int32_t));
                set_flag(PIPE_MTE2, PIPE_V, event_t(7));
            }
        }
        DirectOutputStore<kAttentionOutput> store{args, row, static_cast<uint32_t>(visible), request, prefetched_table};
        if (visible <= Policy::pivot) {
            topk_select_run<typename Policy::LowerConfig, kSortedIndex, bfloat16_t, int32_t>(
                visible, topk, row_scores, nullptr, row_indices, false, nullptr, -1, 0, store);
        } else {
            topk_select_run<typename Policy::UpperConfig, kSortedIndex, bfloat16_t, int32_t>(
                visible, topk, row_scores, nullptr, row_indices, false, nullptr, -1, 0, store);
        }
        asc_sync();
    }
}
#endif
}  // namespace

#endif  // __DAV_VEC__

// SIMT headers introduce math overloads for SIMT lanes. Keep them local
// to the metadata section, after the scalar/SIMD scoring and TopK code.
#if defined(__DAV_VEC__)
#include <simt_api/asc_simt.h>

namespace deep_gemm {

template <typename T>
__simt_callee__ __aicore__ inline T warp_prefix_sum(T value, uint32_t lane_idx) {
    constexpr uint32_t kWarpSize = 32;
    for (uint32_t delta = 1; delta < kWarpSize; delta <<= 1) {
        const T previous = asc_shfl_up(value, delta);
        if (lane_idx >= delta)
            value += previous;
    }
    return value;
}

// Each input is strictly increasing; equal endpoints advance together across inputs.
template <uint32_t kNumInputs>
__simt_callee__ __aicore__ inline uint32_t sorted_union_partition(
    __ubuf__ volatile uint32_t** ub_inputs, const uint32_t* num_items, uint32_t thread_idx,
    uint32_t num_threads, uint32_t* input_begins, uint32_t& num_input_items) {
    constexpr uint32_t kNoKey = 0xffffffffu;
    uint32_t num_active_inputs = 0;
    num_input_items = 0;
    #pragma unroll
    for (uint32_t input_idx = 0; input_idx < kNumInputs; ++input_idx) {
        input_begins[input_idx] = 0;
        num_input_items += num_items[input_idx];
        num_active_inputs += num_items[input_idx] != 0;
    }
    uint32_t partition_begin = thread_idx * num_input_items / num_threads;
    int32_t num_remaining = partition_begin;
    #pragma unroll
    for (uint32_t input_idx = 0; input_idx < kNumInputs; ++input_idx) {
        const uint32_t num_other_items = num_input_items - num_items[input_idx];
        input_begins[input_idx] = max(partition_begin, num_other_items) - num_other_items;
        num_remaining -= input_begins[input_idx];
    }
    const uint32_t shift = (num_active_inputs >= 2) + (num_active_inputs >= 3);
    while (num_remaining > 0) {
        const uint32_t step = max(static_cast<uint32_t>(num_remaining) >> shift, 1u);
        uint32_t keys[kNumInputs], num_steps[kNumInputs], min_key = kNoKey;
        #pragma unroll
        for (uint32_t input_idx = 0; input_idx < kNumInputs; ++input_idx) {
            num_steps[input_idx] = min(step, num_items[input_idx] - input_begins[input_idx]);
            keys[input_idx] = num_steps[input_idx] ? ub_inputs[input_idx][input_begins[input_idx] + num_steps[input_idx] - 1] : kNoKey;
            min_key = min(min_key, keys[input_idx]);
        }
        #pragma unroll
        for (uint32_t input_idx = 0; input_idx < kNumInputs; ++input_idx) {
            const uint32_t num_advanced = keys[input_idx] == min_key ? num_steps[input_idx] : 0;
            input_begins[input_idx] += num_advanced;
            num_remaining -= num_advanced;
        }
    }
    return partition_begin - num_remaining;
}

template <uint32_t kNumInputs>
__simt_callee__ __aicore__ inline uint32_t sorted_union_advance(
    __ubuf__ volatile uint32_t** ub_inputs, const uint32_t* num_items, uint32_t* input_positions,
    uint32_t* item_positions, uint32_t& key) {
    constexpr uint32_t kNoKey = 0xffffffffu;
    uint32_t keys[kNumInputs];
    key = kNoKey;
    #pragma unroll
    for (uint32_t input_idx = 0; input_idx < kNumInputs; ++input_idx) {
        keys[input_idx] = input_positions[input_idx] < num_items[input_idx] ? ub_inputs[input_idx][input_positions[input_idx]] : kNoKey;
        key = min(key, keys[input_idx]);
    }
    uint32_t input_mask = 0;
    #pragma unroll
    for (uint32_t input_idx = 0; input_idx < kNumInputs; ++input_idx) {
        item_positions[input_idx] = input_positions[input_idx];
        const bool advance = keys[input_idx] == key;
        input_mask |= static_cast<uint32_t>(advance) << input_idx;
        input_positions[input_idx] += advance;
    }
    return input_mask;
}

} // namespace deep_gemm
#endif

#if defined(__DAV_VEC__)
// DeepGEMM-Ascend: paged_mqa_logits_metadata.hpp
namespace deep_gemm {

// The scalar scorer and SIMT threads require separate callable helpers.
__simt_callee__ __aicore__ inline uint32_t simt_ceil_div(uint32_t n, uint32_t d) {
    return (n + d - 1) / d;
}

// Read the current producer values when the persistent AIV reuses a GM address.
__simt_callee__ __aicore__ inline uint32_t load_metadata_input(__gm__ uint32_t* address) {
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address);
}

__simt_callee__ __aicore__ inline uint32_t simt_context_len(uint32_t q, uint32_t count, __gm__ uint32_t* lengths) {
    uint32_t context_len = 0;
    for (uint32_t row = q; row < q + count; ++row)
        context_len = max(context_len, load_metadata_input(lengths + row));
    return context_len;
}

inline constexpr uint32_t kNumPagedMQAMetadataThreads = 128;
inline constexpr uint32_t kPagedMQAWarpSize = 32;
inline constexpr uint32_t kNumPagedMQAMetadataWarps = kNumPagedMQAMetadataThreads / kPagedMQAWarpSize;

template <uint32_t kSplitKV, uint32_t kNumQTokensPerTile>
__simt_callee__ __aicore__ inline uint32_t paged_mqa_split_cost(uint32_t num_q_tokens, uint32_t num_kv_tokens) {
    const uint32_t num_q_tiles = simt_ceil_div(num_q_tokens, kNumQTokensPerTile);
    // Measured H8/H16/H32 model: fixed per-split cost plus per-tile cost; H64 uses the tile count.
    const uint32_t cost = kNumQTokensPerTile > 1 ? 2 + 2 * num_q_tiles : num_q_tiles;
    return simt_ceil_div(num_kv_tokens, kSplitKV / 4) * cost + 2;
}

template <uint32_t kSplitKV, uint32_t kNumQTokensPerTile>
__simt_callee__ __aicore__ inline uint32_t paged_mqa_request_cost(uint32_t q_token_start, uint32_t num_q_tokens,
                                                                  __gm__ uint32_t* context_lens) {
    const uint32_t context_len = simt_context_len(q_token_start, num_q_tokens, context_lens);
    const uint32_t num_kv_splits = simt_ceil_div(context_len, kSplitKV);
    if (num_kv_splits == 0)
        return 0;
    const uint32_t full_split_cost = paged_mqa_split_cost<kSplitKV, kNumQTokensPerTile>(num_q_tokens, kSplitKV);
    const uint32_t last_tokens = context_len - (num_kv_splits - 1) * kSplitKV;
    return (num_kv_splits - 1) * full_split_cost +
           paged_mqa_split_cost<kSplitKV, kNumQTokensPerTile>(num_q_tokens, last_tokens);
}

__simt_callee__ __aicore__ inline uint32_t paged_mqa_thread_exclusive_sum(uint32_t value, uint32_t lane_idx,
                                                                          uint32_t warp_idx,
                                                                          __ubuf__ uint32_t* warp_sums) {
    const uint32_t lane_prefix = warp_prefix_sum(value, lane_idx);
    if (lane_idx == kPagedMQAWarpSize - 1)
        warp_sums[warp_idx] = lane_prefix;
    asc_syncthreads();

    if (warp_idx == 0) {
        value = lane_idx < kNumPagedMQAMetadataWarps ? warp_sums[lane_idx] : 0;
        value = warp_prefix_sum(value, lane_idx);
        if (lane_idx < kNumPagedMQAMetadataWarps)
            warp_sums[lane_idx] = value;
    }
    asc_syncthreads();

    const uint32_t warp_offset = warp_idx == 0 ? 0 : warp_sums[warp_idx - 1];
    const uint32_t previous_prefix = asc_shfl_up(lane_prefix, 1);
    return warp_offset + (lane_idx == 0 ? 0 : previous_prefix);
}

__simt_callee__ __aicore__ inline void paged_mqa_prefix_scan(uint32_t num_items, __ubuf__ uint32_t* prefix,
                                                             __ubuf__ uint32_t* warp_sums) {
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t lane_idx = thread_idx % kPagedMQAWarpSize;
    const uint32_t warp_idx = thread_idx / kPagedMQAWarpSize;
    // Keep adjacent threads' chunks on different UB banks by using an odd lane stride.
    const uint32_t num_items_per_thread = simt_ceil_div(num_items, kNumPagedMQAMetadataThreads) | 1u;
    const uint32_t item_begin_idx = min(thread_idx * num_items_per_thread, num_items);
    const uint32_t item_end_idx = min(item_begin_idx + num_items_per_thread, num_items);

    uint32_t thread_sum = 0;
    for (uint32_t item_idx = item_begin_idx; item_idx < item_end_idx; ++item_idx) {
        thread_sum += prefix[item_idx];
        prefix[item_idx] = thread_sum;
    }

    const uint32_t thread_offset = paged_mqa_thread_exclusive_sum(thread_sum, lane_idx, warp_idx, warp_sums);
    for (uint32_t item_idx = item_begin_idx; item_idx < item_end_idx; ++item_idx)
        prefix[item_idx] += thread_offset;
    asc_syncthreads();
}

template <uint32_t kSplitKV, uint32_t kNumQTokensPerTile, uint32_t kNumCores>
__simt_vf__ __aicore__ __launch_bounds__(kNumPagedMQAMetadataThreads) inline void paged_mqa_logits_metadata_vf(
    uint32_t num_q_tokens,
    __gm__ uint32_t* context_lens, __gm__ uint32_t* indices,
    __ubuf__ uint32_t* metadata_ub,
    __ubuf__ uint32_t* prefix, __ubuf__ uint32_t* request_starts,
    __ubuf__ uint32_t* warp_sums
) {
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t lane_idx = thread_idx % kPagedMQAWarpSize;
    const uint32_t warp_idx = thread_idx / kPagedMQAWarpSize;
    const uint32_t metadata_idx = thread_idx * 2;
    if (thread_idx <= kNumCores) {
        metadata_ub[metadata_idx] = num_q_tokens;
        metadata_ub[metadata_idx + 1] = 0;
    }

    const uint32_t num_q_tokens_per_thread = simt_ceil_div(num_q_tokens, kNumPagedMQAMetadataThreads);
    const uint32_t q_token_begin_idx = min(thread_idx * num_q_tokens_per_thread, num_q_tokens);
    const uint32_t q_token_end_idx = min(q_token_begin_idx + num_q_tokens_per_thread, num_q_tokens);
    uint32_t num_request_starts = 0;
    for (uint32_t q_token_idx = q_token_begin_idx; q_token_idx < q_token_end_idx; ++q_token_idx) {
        const bool is_request_start = q_token_idx == 0 ||
            load_metadata_input(indices + q_token_idx) != load_metadata_input(indices + q_token_idx - 1);
        prefix[q_token_idx] = is_request_start;
        num_request_starts += is_request_start;
    }
    uint32_t request_idx = paged_mqa_thread_exclusive_sum(num_request_starts, lane_idx, warp_idx, warp_sums);
    const uint32_t num_requests = warp_sums[kNumPagedMQAMetadataWarps - 1];
    for (uint32_t q_token_idx = q_token_begin_idx; q_token_idx < q_token_end_idx; ++q_token_idx) {
        if (prefix[q_token_idx])
            request_starts[request_idx++] = q_token_idx;
    }
    asc_syncthreads();
    for (uint32_t request_idx = thread_idx; request_idx < num_requests; request_idx += kNumPagedMQAMetadataThreads) {
        const uint32_t q_token_start = request_starts[request_idx];
        const uint32_t q_token_end = request_idx + 1 < num_requests ? request_starts[request_idx + 1] : num_q_tokens;
        const uint32_t num_q_tokens_in_request = q_token_end - q_token_start;
        prefix[request_idx] =
            paged_mqa_request_cost<kSplitKV, kNumQTokensPerTile>(q_token_start, num_q_tokens_in_request, context_lens);
    }
    asc_syncthreads();
    if (num_requests == 0)
        return;

    paged_mqa_prefix_scan(num_requests, prefix, warp_sums);
    const uint32_t total_cost = prefix[num_requests - 1];
    const uint32_t target_cost = thread_idx * (total_cost / kNumCores) + min(thread_idx, total_cost % kNumCores);
    if (thread_idx <= kNumCores && target_cost < total_cost) {
        uint32_t request_idx = 0;
        uint32_t end = num_requests;
        while (request_idx < end) {
            const uint32_t mid = (request_idx + end) / 2;
            if (prefix[mid] <= target_cost)
                request_idx = mid + 1;
            else
                end = mid;
        }

        const uint32_t q_token_start = request_starts[request_idx];
        const uint32_t q_token_end = request_idx + 1 < num_requests ? request_starts[request_idx + 1] : num_q_tokens;
        const uint32_t num_q_tokens_in_request = q_token_end - q_token_start;
        const uint32_t previous_cost = request_idx == 0 ? 0 : prefix[request_idx - 1];
        const uint32_t context_len = simt_context_len(q_token_start, num_q_tokens_in_request, context_lens);
        const uint32_t full_split_cost = paged_mqa_split_cost<kSplitKV, kNumQTokensPerTile>(num_q_tokens_in_request, kSplitKV);
        metadata_ub[metadata_idx] = q_token_start;
        metadata_ub[metadata_idx + 1] = min((target_cost - previous_cost) / full_split_cost,
                                            simt_ceil_div(context_len, kSplitKV) - 1);
    }
    asc_syncthreads();
}

template <uint32_t kSplitKV, uint32_t kNumQTokensPerTile, uint32_t kNumCores>
__simt_vf__ __aicore__ __launch_bounds__(kNumPagedMQAMetadataThreads) inline void paged_mqa_logits_metadata_streaming_vf(
    uint32_t num_q_tokens,
    __gm__ uint32_t* context_lens, __gm__ uint32_t* indices,
    __ubuf__ uint32_t* metadata_ub
) {
    const uint32_t thread_idx = threadIdx.x;
    auto total_cost_ub = metadata_ub + (kNumCores + 1) * 2;
    if (thread_idx == 0) {
        uint32_t total_cost = 0;
        uint32_t q_token_start = 0;
        while (q_token_start < num_q_tokens) {
            uint32_t q_token_end = q_token_start + 1;
            while (q_token_end < num_q_tokens &&
                   load_metadata_input(indices + q_token_end) == load_metadata_input(indices + q_token_start))
                ++q_token_end;
            total_cost += paged_mqa_request_cost<kSplitKV, kNumQTokensPerTile>(
                q_token_start, q_token_end - q_token_start, context_lens);
            q_token_start = q_token_end;
        }
        total_cost_ub[0] = total_cost;
    }
    asc_syncthreads();

    if (thread_idx <= kNumCores) {
        const uint32_t metadata_idx = thread_idx * 2;
        metadata_ub[metadata_idx] = num_q_tokens;
        metadata_ub[metadata_idx + 1] = 0;
        const uint32_t total_cost = total_cost_ub[0];
        const uint32_t target_cost = thread_idx * (total_cost / kNumCores) + min(thread_idx, total_cost % kNumCores);
        if (target_cost < total_cost) {
            uint32_t previous_cost = 0;
            uint32_t q_token_start = 0;
            while (q_token_start < num_q_tokens) {
                uint32_t q_token_end = q_token_start + 1;
                while (q_token_end < num_q_tokens &&
                       load_metadata_input(indices + q_token_end) == load_metadata_input(indices + q_token_start))
                    ++q_token_end;
                const uint32_t num_q_tokens_in_request = q_token_end - q_token_start;
                const uint32_t request_cost = paged_mqa_request_cost<kSplitKV, kNumQTokensPerTile>(
                    q_token_start, num_q_tokens_in_request, context_lens);
                const uint32_t next_cost = previous_cost + request_cost;
                if (target_cost < next_cost) {
                    const uint32_t context_len = simt_context_len(q_token_start, num_q_tokens_in_request, context_lens);
                    const uint32_t full_split_cost = paged_mqa_split_cost<kSplitKV, kNumQTokensPerTile>(num_q_tokens_in_request, kSplitKV);
                    metadata_ub[metadata_idx] = q_token_start;
                    metadata_ub[metadata_idx + 1] = min((target_cost - previous_cost) / full_split_cost,
                                                      simt_ceil_div(context_len, kSplitKV) - 1);
                    break;
                }
                previous_cost = next_cost;
                q_token_start = q_token_end;
            }
        }
    }
    asc_syncthreads();
}

} // namespace deep_gemm

static __aicore__ __attribute__((noinline)) void prepare_metadata(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
    using namespace deep_gemm;
    const uint32_t active = args[4];
    asc_init();
    constexpr uint32_t metadata_bytes = 33 * 2 * sizeof(uint32_t);
    // The persistent AIV SIMT entry provides an 8 KiB shared UB window.
    constexpr uint32_t scratch_bytes = 8 * 1024;
    static_assert(scratch_bytes <= UBSizeBytes, "UB overflow");
    const uint64_t offset = aligned((2ull * active + kNumPagedMQAMetadataWarps) * sizeof(uint32_t), 32ull);
    auto metadata_ub = reinterpret_cast<__ubuf__ uint32_t*>(0);
    if (offset + metadata_bytes <= scratch_bytes) {
        auto prefix = reinterpret_cast<__ubuf__ uint32_t*>(0);
        auto starts = prefix + active;
        auto sums = starts + active;
        metadata_ub = reinterpret_cast<__ubuf__ uint32_t*>(offset);
        asc_vf_call<paged_mqa_logits_metadata_vf<512, 2, 32>>(
            cce::dim3(kNumPagedMQAMetadataThreads), active, data<uint32_t>(args, 1),
            data<uint32_t>(args, 0), metadata_ub, prefix, starts, sums);
    } else {
        static_assert(metadata_bytes + sizeof(uint32_t) <= scratch_bytes, "UB overflow");
        asc_vf_call<paged_mqa_logits_metadata_streaming_vf<512, 2, 32>>(
            cce::dim3(kNumPagedMQAMetadataThreads), active, data<uint32_t>(args, 1),
            data<uint32_t>(args, 0), metadata_ub);
    }
    asc_sync_pipe(PIPE_ALL);
    asc_copy_ub2gm_align(data<uint32_t>(args, 2), metadata_ub, 1, metadata_bytes,
                        asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, 0, 0);
    asc_sync();
#endif
}

#endif  // __DAV_VEC__
// Externs pack Tensor pointers before scalar arguments. A shorter signature
// stores its mode where longer signatures have a non-null Tensor pointer.
// Both scoring signatures place the mode in slot 11.
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
    if (args[3] == kMetadata) {
        prepare_metadata(args);
        return;
    }
    if (args[kModeArg] == kAttention) {
        if (args[kSortedArg]) select_rows<topk_select_ascend::Topk512Config, true, true>(args, 512);
        else select_rows<topk_select_ascend::Topk512Config, true, false>(args, 512);
        return;
    }
    if (args[kModeArg] == kPool) {
        using namespace topk_select_ascend;
        const int64_t capacity = reinterpret_cast<__gm__ Tensor*>(args[kIndicesArg])->shapes[1];
        if (capacity <= 512) select_rows<Topk512Config, false, true>(args, capacity);
        else if (capacity <= 1024) select_rows<Topk1024Config, false, true>(args, capacity);
        else select_rows<Topk4096Config, false, true>(args, capacity < 2048 ? capacity : 2048);
        return;
    }
#endif
    if (args[11] == kScoreCandidates) score<true>(args);
    else score<false>(args);
}
