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

#ifndef PYPTO_V41_MOE_PRIVATE_PIPELINE_COMPUTE_HPP
#define PYPTO_V41_MOE_PRIVATE_PIPELINE_COMPUTE_HPP

#include "moe_pipeline.hpp"

// BEGIN DeepSeek MIT port: include/deep_gemm/mega_moe.hpp
#ifndef PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_MEGA_MOE_HPP
#define PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_MEGA_MOE_HPP

// Adapted from https://github.com/deepseek-ai/DeepGEMM-Ascend
// Revision: 3a35326271b069b8b5bf1a754435acd6cc91575b (MIT; see the local notice above).


#include <c_api/asc_simd.h>
#include <c_api/cache_ctrl/cache_ctrl.h>
#include <simt_api/device_functions.h>
#include <simt_api/common_functions.h>

namespace deep_gemm::mega_moe {

using layout::BLOCK_M;
using layout::BLOCK_N;

// Pipeline choices: double-buffer L0A/B and keep three A / four routed B stages in L1
constexpr uint32_t kNumL0Stages = 2;
constexpr uint32_t kNumL1AStages = 3, kNumL1BStages = 4, kNumSFStages = 2;

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
    static constexpr uint32_t BLOCK_K = MOE_TOKENS <= 80 ? 256 : L0ASizeBytes / kMadM;
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
// L1      A                 0      3 x 64
//         routed B        192      4 x 64
//         shared B        192      2 x 128 (M=128), 3 x 64 (M=256)
//         A SF            448      2 x 16
//         B SF            480      2 x 16
// L0A     A                 0      2 x 32
// L0B     B                 0      2 x 32
// L0C     FP32 accum        0      2 x 64 (M=128), 1 x 256 (M=256)

// AIC: L1A 0–2, SF 3–4, shared L1B 5–6 or 5–7
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
    const auto topk_idx = workspace.input.topk_idx.template get_ptr<uint64_t>(buffer);
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
    // Reduce      225      2 stages within the remaining 31 KiB
    constexpr uint32_t kNumFP4Stages = 3, kNumNZStages = 2;
    constexpr ub_ptr<uint8_t> ub_fp4(1, L0BSizeBytes / 2);
    constexpr ub_ptr<uint8_t> ub_nz(1, L0BSizeBytes + 512, ub_fp4.offset(kNumFP4Stages));
    constexpr auto ub_reduce_inputs = make_reduce_buffer<workspace>(ub_nz.offset(kNumNZStages));

    static_assert(ub_nz.offset(kNumNZStages) <= UBSizeBytes and ub_reduce_inputs.shape_n > 0, "AIV0 UB overflow");

    // Initialize UB reuse credits; AIC releases the initial L1B credits
    notify<PIPE_V, PIPE_MTE2, kNumFP4Stages>(kFP4Empty);
    notify<PIPE_MTE3, PIPE_V, kNumNZStages>(kNZEmpty);

    // Size the reduce window from local output tokens
    const uint32_t num_reduce_window_m_blocks = ceil_div(num_tokens, BLOCK_M);
    sched::LinearTaskState schedule;
    sched::ExpertCursor<workspace> expert_cursors[2]{};
    uint32_t fp4_stage_idx = 0, nz_stage_idx = 0;
    uint32_t l1b_stage_idx = 0;
    bool had_routed_task = false;

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
        const uint32_t valid_m = expert_cursors[layer_idx].advance_to(buffer, pool_block_idx);
        const auto weight_l2_hint = expert_cursors[layer_idx].num_expert_tokens > BLOCK_M ?
            asc_load_l2_cache_mode::NORMAL_LAST_VICTIM : asc_load_l2_cache_mode::NOTALLOC_KEEP;
        const uint32_t weight_expert = expert_cursors[layer_idx].local_expert_idx;
        const uint32_t weight_output_n = is_linear1 ? config.intermediate_hidden : config.hidden;
        const auto first_weight = is_linear1 ? weights.l1.data : weights.l2.data;
        const auto up_weight = weights.l1_up.data;

        const auto load_weights = [&]<uint32_t MAD_M>() __aicore__ {
            constexpr uint32_t BLOCK_K = MMAConfig<MAD_M>::BLOCK_K;
            constexpr uint32_t MAD_N = MMAConfig<MAD_M>::MAD_N;
            constexpr uint32_t kNumColGroups = MAD_N / get_frac_k<uint8_t>();
            constexpr l1_ptr<float8_e4m3_t, Major::MN> l1b(MAD_N, BLOCK_K, kNumL1AStages * L0ASizeBytes);

            for (uint32_t k_idx = 0; k_idx < num_k; k_idx += BLOCK_K) {
                // valid_k remains a multiple of MAD_K, including the final K block
                const uint32_t valid_k = min(BLOCK_K, num_k - k_idx);
                for (uint32_t n_mad_idx = 0; n_mad_idx < BLOCK_N; n_mad_idx += MAD_N) {
                    // Load packed FP4
                    wait<PIPE_V, PIPE_MTE2>(kFP4Empty, fp4_stage_idx);
#ifdef __DAV_VEC__
                    original_weights::load_routed_panel<BLOCK_K, MAD_N>(
                        ub_fp4[fp4_stage_idx].addr, first_weight, up_weight, is_linear1,
                        weight_expert, num_k, weight_output_n, n_block_idx, n_mad_idx, k_idx, valid_k, weight_l2_hint);
#endif
                    notify<PIPE_MTE2, PIPE_V>(kFP4Full, fp4_stage_idx);

                    // Dequantize to FP8 NZ
                    wait<PIPE_MTE2, PIPE_V>(kFP4Full, fp4_stage_idx);
                    wait<PIPE_MTE3, PIPE_V>(kNZEmpty, nz_stage_idx);
#ifdef __DAV_VEC__
                    original_weights::decode_routed_panel<BLOCK_K, MAD_N>(
                        ub_fp4[fp4_stage_idx].addr, ub_nz[nz_stage_idx].addr);
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
        if (valid_m <= 128)
            load_weights.template operator()<128>();
        else
            load_weights.template operator()<256>();
    }

    // Return with neutral events after all routed tasks on this core.
    wait<PIPE_V, PIPE_MTE2, kNumFP4Stages>(kFP4Empty);
    wait<PIPE_MTE3, PIPE_V, kNumNZStages>(kNZEmpty);
    if (had_routed_task)
        wait_intra_block<PIPE_MTE3, 0, kNumL1BStages>(kL1BEmpty);
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
                                   kNumL1AStages * L0ASizeBytes + kNumL1BStages * L0BSizeBytes);
    constexpr l1_ptr<int16_t> l1_sfb(BLOCK_N, kNumSFPairsPerChunk, l1_sfa.offset(kNumSFStages));
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
    if constexpr (kHasSharedExperts)
        notify<PIPE_MTE1, PIPE_MTE2, 3>(kSharedL1BEmpty);

    sched::LinearTaskState schedule;
    sched::ExpertCursor<workspace> expert_cursors[2]{};
    uint32_t l1a_stage_idx = 0;
    uint32_t l1b_stage_idx = 0;
    uint32_t sf_stage_idx = 0;
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
            constexpr l1_ptr<float8_e4m3_t, Major::MN> l1b(MAD_N, BLOCK_K, l1a.offset(kNumL1AStages));
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
                        copy_l1_to_l0b_mx(l0b[l0b_stage_idx], l1_sfb[sf_stage_idx], n_mad_idx, sf_pair_idx_in_chunk, MAD_K);
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
                                            CoreTopology<kFullTopology> topology) {
    constexpr auto& workspace = layout::kWorkspace<kNumRanks, kNumExperts, kNumSharedExperts,
                                                   kNumMaxTokensPerRank, kNumTopk, kHidden, kIntermediateHidden>;
    constexpr auto& config = workspace.config;

    // Measured 32 MiB combine-work threshold for tail reduction
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
    const auto topk_idx = workspace.input.topk_idx.get_ptr(buffer);
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
        const uint32_t num_metadata_workers = min(max(ceil_div(num_tokens * config.num_topk, layout::kNumDispatchThreads), 1u), topology.cores());
        if (core_idx < num_metadata_workers) {
            // SIMT reads UB offsets; metadata owns this scratch until the VF completes
            auto* offsets = reinterpret_cast<__ubuf__ int64_t*>(0);
            for (uint32_t i = 0; i < kNumRanks; ++ i)
                offsets[i] = sym_buffer.offsets[i];
            asc_sync_data_barrier(DSB_UB);
            asc_vf_call<comm::build_dispatch_metadata<workspace, kOverlapReduce, kFullTopology>>(
                dim3(layout::kNumDispatchThreads, 1, 1),
                buffer, topk_idx, num_tokens, rank_idx, core_idx, num_metadata_workers, offsets);
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
#endif // PYPTO_V41_MOE_PRIVATE_INCLUDE_DEEP_GEMM_MEGA_MOE_HPP
// END DeepSeek MIT port: include/deep_gemm/mega_moe.hpp

// BEGIN PyPTO adapter: pypto_sym_buffer_adapter.hpp
#ifndef PYPTO_V41_MOE_PRIVATE_PYPTO_SYM_BUFFER_ADAPTER_HPP
#define PYPTO_V41_MOE_PRIVATE_PYPTO_SYM_BUFFER_ADAPTER_HPP
/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tensor.h"
#include "intrinsic.h"
#include "comm_context.h"

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
#endif // PYPTO_V41_MOE_PRIVATE_PYPTO_SYM_BUFFER_ADAPTER_HPP
// END PyPTO adapter: pypto_sym_buffer_adapter.hpp

// BEGIN PyPTO adapter: mega_input_adapter.hpp
#ifndef PYPTO_V41_MOE_PRIVATE_MEGA_INPUT_ADAPTER_HPP
#define PYPTO_V41_MOE_PRIVATE_MEGA_INPUT_ADAPTER_HPP
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
__aicore__ inline void publish(__gm__ uint8_t* buffer, __gm__ uint8_t* x,
                               __gm__ uint8_t* scale, __gm__ int32_t* indices,
                               __gm__ float* weights, __gm__ bfloat16_t* output,
                               uint32_t active, uint32_t aiv_idx, deep_gemm::CoreTopology<kFullTopology> topology)
{
#ifdef __DAV_VEC__
    constexpr auto& config = workspace.config;
    constexpr uint32_t rows = 16;
    constexpr uint32_t width = config.hidden;
    constexpr uint32_t pairs = width / 64;
    constexpr uint32_t payload_bytes = rows * width;
    constexpr uint32_t scale_bytes = rows * pairs * sizeof(uint16_t);
    constexpr uint32_t scale_zz = payload_bytes;
    constexpr uint32_t scale_nd = scale_zz + scale_bytes;
    constexpr uint32_t scale_tmp = scale_nd + scale_bytes;
    static_assert(config.num_max_tokens_per_rank % rows == 0);
    static_assert(scale_tmp + scale_bytes <= PTO_UBUF_SIZE_BYTES);
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

    if constexpr (config.num_max_tokens_per_rank <= 80) {
        constexpr uint32_t entries = config.num_max_tokens_per_rank * config.num_topk;
        const uint32_t per_worker = deep_gemm::ceil_div(entries, topology.vectors());
        const uint32_t first_entry = aiv_idx * per_worker;
        const uint32_t last_entry = first_entry + per_worker < entries ? first_entry + per_worker : entries;
        const uint32_t active_entries = active * config.num_topk;
        auto* output_indices = workspace.input.topk_idx.get_ptr(buffer);
        auto* output_weights = workspace.input.topk_weights.template get_ptr<uint32_t>(buffer);
        auto* weight_bits = reinterpret_cast<__gm__ uint32_t*>(weights);
        for (uint32_t index = first_entry; index < last_entry; ++index) {
            const int64_t expert = index < active_entries ? static_cast<int64_t>(asc_load_dev(indices + index)) : int64_t{-1};
            const uint32_t bits = index < active_entries ? asc_load_dev(weight_bits + index) : uint32_t{0};
            asc_store_dev(output_indices + index, expert);
            asc_store_dev(output_weights + index, bits);
        }
    }

    for (uint32_t first = aiv_idx * rows; first < config.num_max_tokens_per_rank;
         first += topology.vectors() * rows) {
        Vec<uint8_t, rows, width> payload(rows, width);
        TASSIGN(payload, 0);
        auto input = global(x + first * width, rows, width, width);
        auto destination = global(workspace.input.acts.data.template get_ptr<uint8_t>(buffer, first * width),
                                  rows, width, width);
        TLOAD(payload, input);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(destination, payload);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

        Vec<uint16_t, pairs, rows> packed(pairs, rows), temporary(pairs, rows);
        Vec<uint16_t, rows, pairs> unpacked(rows, pairs);
        TASSIGN(packed, scale_zz);
        TASSIGN(unpacked, scale_nd);
        TASSIGN(temporary, scale_tmp);
        auto input_scale = global(reinterpret_cast<__gm__ uint16_t*>(scale) + first * pairs,
                                  pairs, rows, rows);
        auto output_scale = global(workspace.input.acts.sf.template get_ptr<uint16_t>(buffer, first * pairs),
                                   rows, pairs, pairs);
        TLOAD(packed, input_scale);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        TTRANS(unpacked, packed, temporary);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        TSTORE(output_scale, unpacked);

        auto* output_indices = workspace.input.topk_idx.get_ptr(buffer);
        auto* output_weights = workspace.input.topk_weights.template get_ptr<uint32_t>(buffer);
        auto* weight_bits = reinterpret_cast<__gm__ uint32_t*>(weights);
        if constexpr (config.num_max_tokens_per_rank > 80) {
            for (uint32_t row = first; row < first + rows; ++row) {
                for (uint32_t k = 0; k < config.num_topk; ++k) {
                    const uint32_t index = row * config.num_topk + k;
                    const int64_t expert = row < active ? static_cast<int64_t>(asc_load_dev(indices + index)) : int64_t{-1};
                    const uint32_t bits = row < active ? asc_load_dev(weight_bits + index) : uint32_t{0};
                    asc_store_dev(output_indices + index, expert);
                    asc_store_dev(output_weights + index, bits);
                }
            }
        }
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
#endif // PYPTO_V41_MOE_PRIVATE_MEGA_INPUT_ADAPTER_HPP
// END PyPTO adapter: mega_input_adapter.hpp

#endif // PYPTO_V41_MOE_PRIVATE_PIPELINE_COMPUTE_HPP
