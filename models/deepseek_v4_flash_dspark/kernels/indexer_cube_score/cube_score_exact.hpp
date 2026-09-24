/* Copyright (c) PyPTO Contributors.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 * Experimental prefetched score tile with low FIX/high GEMV overlap.
 */
#pragma once
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"

namespace dspark_exact {
using namespace pto;
using ScoreAcc = TileAcc<float,16,256,1,256>;
template<typename T,int M,int N> using Global = GlobalTensor<T,Shape<1,1,1,M,N>,Stride<M*N,M*N,M*N,N,1>>;
template<typename T,int M,int N> using Mat = Tile<TileType::Mat,T,M,N,BLayout::ColMajor,M,N,SLayout::RowMajor,512>;
using KeyPageGlobal = GlobalTensor<int8_t,Shape<1,1,1,128,32>,Stride<4096,4096,4096,1,128>,Layout::DN>;
using KeyMat = Tile<TileType::Mat,int8_t,128,256,BLayout::RowMajor,128,256,SLayout::ColMajor,512>;
using KeyPageMat = Tile<TileType::Mat,int8_t,128,256,BLayout::RowMajor,128,32,SLayout::ColMajor,512>;
using BiasMat = Tile<TileType::Mat,int32_t,1,256,BLayout::RowMajor,1,256,SLayout::NoneBox>;
using BiasTile = Tile<TileType::Bias,int32_t,1,256,BLayout::RowMajor,1,256,SLayout::NoneBox>;
using CoefficientMat = Tile<TileType::Mat,half,1,256,BLayout::RowMajor,1,64,SLayout::NoneBox>;
using IntegerAcc = TileAcc<int32_t,64,256,64,256>;
using FloatAcc = TileAcc<float,64,256,64,256>;
using LimbMat = Mat<half,64,256>;

// A=16384. Constants occupy 21504 bytes of a FP32[1,5376] allocation:
// half neg_magic[64,16] @0, -I[64,64] @2048, add_offset[64,16] @10240,
// half constant_right[16,256] @12288, int32 bias[1,256] @20480.
// neg_magic[:,0]=-0.75, add_offset[:,0]=2^-14,
// constant_right[0,:]=1024, bias[:]=0x44400000-1024; other entries zero.
// Coefficients are three successive FP16 residual limbs of A*W.
class CubeScoreComputer {
public:
    __aicore__ inline CubeScoreComputer(__gm__ int8_t *keys,__gm__ float *constants)
        : keys_(keys),prefetched_mask_(0),released_mask_(0) {
        TASSIGN(query_mat_,0); TASSIGN(key_mat_,8192);
        TASSIGN(neg_magic_mat_,40960); TASSIGN(neg_identity_mat_,43008);
        TASSIGN(add_offset_mat_,51200); TASSIGN(constant_right_mat_,53248);
        TASSIGN(bias_mat_,61440); TASSIGN(coefficient_hi_mat_,62464);
        TASSIGN(coefficient_lo_mat_,62976); TASSIGN(coefficient_tail_mat_,63488); TASSIGN(hi_mat_,65536); TASSIGN(lo_mat_,98304);
        TASSIGN(query_left_,0); TASSIGN(neg_magic_left_,8192);
        TASSIGN(neg_identity_left_,10240); TASSIGN(add_offset_left_,18432);
        TASSIGN(coefficient_hi_left_,20480); TASSIGN(coefficient_lo_left_,20992);
        TASSIGN(coefficient_tail_left_,21504);
        TASSIGN(key_right_,0); TASSIGN(constant_right_,32768);
        TASSIGN(hi_right_,0); TASSIGN(lo_right_,0); TASSIGN(bias_,0);
        TASSIGN(integer_acc_,0); TASSIGN(float_acc_,0); TASSIGN(score_acc_,65536);
        auto bytes=reinterpret_cast<__gm__ uint8_t *>(constants);
        Global<half,64,16> neg_magic(reinterpret_cast<__gm__ half *>(bytes));
        Global<half,64,64> neg_identity(reinterpret_cast<__gm__ half *>(bytes+2048));
        Global<half,64,16> add_offset(reinterpret_cast<__gm__ half *>(bytes+10240));
        Global<half,16,256> constant_right(reinterpret_cast<__gm__ half *>(bytes+12288));
        Global<int32_t,1,256> bias(reinterpret_cast<__gm__ int32_t *>(bytes+20480));
        TLOAD(neg_magic_mat_,neg_magic); TLOAD(neg_identity_mat_,neg_identity);
        TLOAD(add_offset_mat_,add_offset); TLOAD(constant_right_mat_,constant_right);
        TLOAD(bias_mat_,bias);
        fence_mte2_mte1();
        TMOV(neg_magic_left_,neg_magic_mat_); TMOV(neg_identity_left_,neg_identity_mat_);
        TMOV(add_offset_left_,add_offset_mat_); TMOV(bias_,bias_mat_);
        TMOV(constant_right_,constant_right_mat_);
        fence_mte1_m();
    }

    // Constants stay resident for the worker. Query/coefficients reload per leaf.
    __aicore__ inline void set_query(__gm__ int8_t *query,__gm__ half *coefficient_hi,
                                     __gm__ half *coefficient_lo,__gm__ half *coefficient_tail) {
        // Protect L1 and L0A when a persistent worker advances to another leaf.
        set_flag(PIPE_MTE1,PIPE_MTE2,EVENT_ID0); wait_flag(PIPE_MTE1,PIPE_MTE2,EVENT_ID0);
        Global<int8_t,64,128> q(query);
        Global<half,1,64> hi(coefficient_hi),lo(coefficient_lo),tail(coefficient_tail);
        TLOAD(query_mat_,q); TLOAD(coefficient_hi_mat_,hi); TLOAD(coefficient_lo_mat_,lo);
        TLOAD(coefficient_tail_mat_,tail);
        fence_mte2_mte1();
        set_flag(PIPE_M,PIPE_MTE1,EVENT_ID0); wait_flag(PIPE_M,PIPE_MTE1,EVENT_ID0);
        TMOV(query_left_,query_mat_);
        TEXTRACT(coefficient_hi_left_,coefficient_hi_mat_,0,0);
        TEXTRACT(coefficient_lo_left_,coefficient_lo_mat_,0,0);
        TEXTRACT(coefficient_tail_left_,coefficient_tail_mat_,0,0);
        fence_mte1_m();
    }

    // Start one 256-candidate Key load. Slot 0 occupies L1 [8192,40960),
    // slot 1 [131072,163840). A slot's prior L1 reader must finish before
    // MTE2 writes it again. Ready/release events use IDs 2/3, not math ID 0.
    __aicore__ inline void prefetch(const int64_t physical_rows[8],int slot) {
        const unsigned bit=1u<<slot;
        PTO_ASSERT((prefetched_mask_&bit)==0,"Key slot already prefetched");
        if ((released_mask_&bit)!=0) {
            wait_release(slot);
            released_mask_&=~bit;
        }
        const int base=slot==0 ? 8192 : 131072;
        for (int page=0; page<8; ++page) {
            KeyPageMat part;
            TASSIGN(part,base+page*32*32);
            KeyPageGlobal global(keys_+physical_rows[page]*128);
            TLOAD(part,global);
        }
        set_ready(slot);
        prefetched_mask_|=bit;
    }

    // Consume the already-prefetched slot. If next_rows is non-null, issue
    // the next tile into the other L1 slot before this tile's Cube work.
    // Caller: prefetch(rows0,0); compute(tile&1,next_or_null); finish_leaf().
    __aicore__ inline ScoreAcc &compute(int slot,const int64_t *next_rows) {
        const unsigned bit=1u<<slot;
        PTO_ASSERT((prefetched_mask_&bit)!=0,"Key slot was not prefetched");
        wait_ready(slot);
        prefetched_mask_&=~bit;
        // Previous math may still read L0B at address zero. The alternate
        // L1 buffer is independent, so its MTE2 loads can overlap this wait.
        set_flag(PIPE_M,PIPE_MTE1,EVENT_ID0); wait_flag(PIPE_M,PIPE_MTE1,EVENT_ID0);
        TASSIGN(key_mat_,slot==0 ? 8192 : 131072);
        TMOV(key_right_,key_mat_);
        set_release(slot);
        released_mask_|=bit;
        if (next_rows!=nullptr) prefetch(next_rows,slot^1);
        // hi and lo share L0B address zero; the constant stays at 32768.
        fence_mte1_m();
        TMATMUL_BIAS(integer_acc_,query_left_,key_right_,bias_);
        // Keep the verified type-alias boundary explicitly completion-ordered.
        fence_m_fix();
        set_flag(PIPE_FIX,PIPE_M,EVENT_ID0); wait_flag(PIPE_FIX,PIPE_M,EVENT_ID0);
        TMATMUL_ACC(float_acc_,neg_magic_left_,constant_right_);
        fence_m_fix();
        TMOV<LimbMat,FloatAcc,ReluPreMode::NormalRelu>(hi_mat_,float_acc_);
        fence_fix_mte1();
        TMOV(hi_right_,hi_mat_);
        fence_mte1_m();
        TMATMUL_ACC(float_acc_,neg_identity_left_,hi_right_);
        TMATMUL_ACC(float_acc_,add_offset_left_,constant_right_);
        fence_m_fix();
        TMOV<LimbMat,FloatAcc,ReluPreMode::NormalRelu>(lo_mat_,float_acc_);
        // FIX reads R accumulator 0 while M writes independent ScoreAcc65536.
        // hi remains resident in L0B0 until these three products finish.
        TGEMV(score_acc_,coefficient_hi_left_,hi_right_);
        TGEMV_ACC(score_acc_,score_acc_,coefficient_lo_left_,hi_right_);
        TGEMV_ACC(score_acc_,score_acc_,coefficient_tail_left_,hi_right_);
        fence_fix_mte1();
        set_flag(PIPE_M,PIPE_MTE1,EVENT_ID0); wait_flag(PIPE_M,PIPE_MTE1,EVENT_ID0);
        TMOV(lo_right_,lo_mat_);
        fence_mte1_m();
        TGEMV_ACC(score_acc_,score_acc_,coefficient_hi_left_,lo_right_);
        TGEMV_ACC(score_acc_,score_acc_,coefficient_lo_left_,lo_right_);
        TGEMV_ACC(score_acc_,score_acc_,coefficient_tail_left_,lo_right_);
        fence_m_fix();
        return score_acc_;
    }

    // Consume exactly the release tokens that were produced. A one-tile leaf
    // has only slot 0 pending. There are no outstanding ready tokens when the
    // caller has consumed all tiles, and no token leaks into the next leaf.
    __aicore__ inline void finish_leaf() {
        PTO_ASSERT(prefetched_mask_==0,"Leaf finished with unread prefetched Key");
        if ((released_mask_&1u)!=0) wait_release(0);
        if ((released_mask_&2u)!=0) wait_release(1);
        released_mask_=0;
    }

private:
    __aicore__ static inline void set_ready(int slot) {
        if (slot==0) set_flag(PIPE_MTE2,PIPE_MTE1,EVENT_ID2);
        else set_flag(PIPE_MTE2,PIPE_MTE1,EVENT_ID3);
    }
    __aicore__ static inline void wait_ready(int slot) {
        if (slot==0) wait_flag(PIPE_MTE2,PIPE_MTE1,EVENT_ID2);
        else wait_flag(PIPE_MTE2,PIPE_MTE1,EVENT_ID3);
    }
    __aicore__ static inline void set_release(int slot) {
        if (slot==0) set_flag(PIPE_MTE1,PIPE_MTE2,EVENT_ID2);
        else set_flag(PIPE_MTE1,PIPE_MTE2,EVENT_ID3);
    }
    __aicore__ static inline void wait_release(int slot) {
        if (slot==0) wait_flag(PIPE_MTE1,PIPE_MTE2,EVENT_ID2);
        else wait_flag(PIPE_MTE1,PIPE_MTE2,EVENT_ID3);
    }
    __aicore__ static inline void fence_mte2_mte1() {
        set_flag(PIPE_MTE2,PIPE_MTE1,EVENT_ID0); wait_flag(PIPE_MTE2,PIPE_MTE1,EVENT_ID0);
    }
    __aicore__ static inline void fence_mte1_m() {
        set_flag(PIPE_MTE1,PIPE_M,EVENT_ID0); wait_flag(PIPE_MTE1,PIPE_M,EVENT_ID0);
    }
    __aicore__ static inline void fence_m_fix() {
        set_flag(PIPE_M,PIPE_FIX,EVENT_ID0); wait_flag(PIPE_M,PIPE_FIX,EVENT_ID0);
    }
    __aicore__ static inline void fence_fix_mte1() {
        set_flag(PIPE_FIX,PIPE_MTE1,EVENT_ID0); wait_flag(PIPE_FIX,PIPE_MTE1,EVENT_ID0);
    }
    __gm__ int8_t *keys_;
    unsigned prefetched_mask_,released_mask_;
    Mat<int8_t,64,128> query_mat_; KeyMat key_mat_;
    Mat<half,64,16> neg_magic_mat_,add_offset_mat_;
    Mat<half,64,64> neg_identity_mat_;
    Mat<half,16,256> constant_right_mat_;
    BiasMat bias_mat_; BiasTile bias_;
    CoefficientMat coefficient_hi_mat_,coefficient_lo_mat_,coefficient_tail_mat_;
    LimbMat hi_mat_,lo_mat_;
    TileLeft<int8_t,64,128,64,128> query_left_;
    TileRight<int8_t,128,256,128,256> key_right_;
    TileLeft<half,64,16,64,16> neg_magic_left_,add_offset_left_;
    TileLeft<half,64,64,64,64> neg_identity_left_;
    TileRight<half,16,256,16,256> constant_right_;
    TileRight<half,64,256,64,256> hi_right_,lo_right_;
    TileLeft<half,1,256,1,64> coefficient_hi_left_,coefficient_lo_left_,coefficient_tail_left_;
    IntegerAcc integer_acc_; FloatAcc float_acc_; ScoreAcc score_acc_;
};
} // namespace dspark_exact
