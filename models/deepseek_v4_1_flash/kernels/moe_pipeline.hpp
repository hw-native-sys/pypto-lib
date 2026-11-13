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

#ifndef PYPTO_V41_MOE_PRIVATE_PIPELINE_HPP
#define PYPTO_V41_MOE_PRIVATE_PIPELINE_HPP

// BEGIN DeepSeek MIT port: include/deep_gemm/ascend.hpp
#ifndef PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_ASCEND_HPP
#define PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_ASCEND_HPP

#include <c_api/asc_simd.h>
// BEGIN DeepSeek MIT port: include/deep_gemm/common.hpp
#ifndef PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_COMMON_HPP
#define PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_COMMON_HPP

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
#endif // PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_COMMON_HPP
// END DeepSeek MIT port: include/deep_gemm/common.hpp


// Device primitives, split by abstraction layer:
//   tile_ptr — operand_ptr / result_ptr (shape-carrying device pointers) + l*_ptr aliases
//   copy     — GM/L1/L0/UB data movement, scale-factor load, mad, quant-mode inference
//   sync     — intra-block / flag / ffts synchronization
// BEGIN DeepSeek MIT port: include/deep_gemm/ascend/tile_ptr.hpp
#ifndef PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_ASCEND_TILE_PTR_HPP
#define PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_ASCEND_TILE_PTR_HPP

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
#endif // PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_ASCEND_TILE_PTR_HPP
// END DeepSeek MIT port: include/deep_gemm/ascend/tile_ptr.hpp

// BEGIN DeepSeek MIT port: include/deep_gemm/ascend/copy.hpp
#ifndef PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_ASCEND_COPY_HPP
#define PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_ASCEND_COPY_HPP

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
#endif // PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_ASCEND_COPY_HPP
// END DeepSeek MIT port: include/deep_gemm/ascend/copy.hpp

// BEGIN DeepSeek MIT port: include/deep_gemm/ascend/sync.hpp
#ifndef PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_ASCEND_SYNC_HPP
#define PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_ASCEND_SYNC_HPP

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
#endif // PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_ASCEND_SYNC_HPP
// END DeepSeek MIT port: include/deep_gemm/ascend/sync.hpp

#endif // PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_ASCEND_HPP
// END DeepSeek MIT port: include/deep_gemm/ascend.hpp

// BEGIN DeepSeek MIT port: include/deep_gemm/comm/mega_moe.hpp
#ifndef PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_COMM_MEGA_MOE_HPP
#define PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_COMM_MEGA_MOE_HPP

// BEGIN DeepSeek MIT port: include/deep_gemm/layout/mega_moe.hpp
#ifndef PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_LAYOUT_MEGA_MOE_HPP
#define PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_LAYOUT_MEGA_MOE_HPP

#include <cstdint>
#include <type_traits>

// BEGIN DeepSeek MIT port: include/deep_gemm/layout/sym_buffer.hpp
#ifndef PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_LAYOUT_SYM_BUFFER_HPP
#define PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_LAYOUT_SYM_BUFFER_HPP

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
#endif // PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_LAYOUT_SYM_BUFFER_HPP
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
#endif // PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_LAYOUT_MEGA_MOE_HPP
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

// Build expert-contiguous dispatch metadata
template <const auto& workspace, bool kOverlapReduce, bool kFullTopology>
__simt_vf__ __aicore__ __launch_bounds__(layout::kNumDispatchThreads)
void build_dispatch_metadata(__gm__ uint8_t* buffer, __gm__ const int64_t* topk_idx,
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
        const auto expert_idx = static_cast<uint64_t>(asc_ldcg(const_cast<__gm__ int64_t*>(topk_idx + token_topk_idx)));
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
            const auto expert_idx = static_cast<uint64_t>(asc_ldcg(const_cast<__gm__ int64_t*>(topk_idx + token_topk_idx)));
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
            const uint32_t weight_bits = asc_ldcg(workspace.input.topk_weights.template get_ptr<uint32_t>(buffer, token_topk_idx));
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
#endif // PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_COMM_MEGA_MOE_HPP
// END DeepSeek MIT port: include/deep_gemm/comm/mega_moe.hpp

// BEGIN DeepSeek MIT port: include/deep_gemm/scheduler/mega_moe.hpp
#ifndef PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_SCHEDULER_MEGA_MOE_HPP
#define PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_SCHEDULER_MEGA_MOE_HPP


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
            is_linear1 = state.l1_task_base < required_linear1_task_end + topology.cores();
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
#endif // PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_SCHEDULER_MEGA_MOE_HPP
// END DeepSeek MIT port: include/deep_gemm/scheduler/mega_moe.hpp

// BEGIN DeepSeek MIT port: include/deep_gemm/simd/mega_moe.hpp
#ifndef PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_SIMD_MEGA_MOE_HPP
#define PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_SIMD_MEGA_MOE_HPP


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
#endif // PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_SIMD_MEGA_MOE_HPP
// END DeepSeek MIT port: include/deep_gemm/simd/mega_moe.hpp

// BEGIN DeepSeek MIT port: include/deep_gemm/utils/math.hpp
#ifndef PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_UTILS_MATH_HPP
#define PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_UTILS_MATH_HPP


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
#endif // PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_UTILS_MATH_HPP
// END DeepSeek MIT port: include/deep_gemm/utils/math.hpp

// BEGIN PyPTO adapter: include/deep_gemm/original_weight_adapter.hpp
#ifndef PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_ORIGINAL_WEIGHT_ADAPTER_HPP
#define PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_ORIGINAL_WEIGHT_ADAPTER_HPP
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
AICORE inline void decode_routed_panel(uint32_t raw_address, uint32_t nz_address) {
    using Packed = pto::Tile<pto::TileType::Vec, float4_e2m1x2_t, BlockK, MadN>;
    using NZ = pto::Tile<pto::TileType::Vec, float8_e4m3_t, BlockK, MadN,
                         pto::BLayout::ColMajor, BlockK, MadN, pto::SLayout::RowMajor,
                         512, pto::PadValue::Null, pto::CompactMode::RowPlusOne>;
    static_assert(NZ::StorageBytes == (BlockK + 1) * MadN);
    Packed packed;
    NZ nz;
    pto::TASSIGN(packed, raw_address);
    pto::TASSIGN(nz, nz_address);
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
#endif // PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_ORIGINAL_WEIGHT_ADAPTER_HPP
// END PyPTO adapter: include/deep_gemm/original_weight_adapter.hpp

#endif // PYPTO_V41_MOE_PRIVATE_PIPELINE_HPP
