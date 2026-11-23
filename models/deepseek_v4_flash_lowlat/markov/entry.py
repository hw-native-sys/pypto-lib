# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Markov model presets, vocabulary projection, and native sampling."""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import pypto.language as pl
import sys
import pypto.language.distributed as pld


@dataclass(frozen=True)
class DeepSeekV4Config:
    """Follows HuggingFace config.json"""

    name: str

    # ---- attention / hidden ----
    hidden_size: int
    num_attention_heads: int
    head_dim: int  # MLA value-head dim
    qk_rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int  # grouped output projection
    sliding_window: int
    rms_norm_eps: float
    vocab_size: int

    # ---- MoE ----
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    scoring_func: Literal["softmax", "sigmoid", "sqrtsoftplus"]
    routed_scaling_factor: float
    swiglu_limit: float

    # ---- layers ----
    num_hidden_layers: int
    num_hash_layers: int  # first layers use hash routing
    num_nextn_predict_layers: int  # multi-token-prediction layers
    compress_ratios: Tuple[int, ...]  # per-layer KV compression ratio (0 / 4 / 128)

    # ---- lightning indexer ----
    index_n_heads: int
    index_head_dim: int
    index_topk: int  # compressed positions kept by the indexer

    # ---- hyper-connections (HC) ----
    hc_mult: int  # hc-stack width
    hc_sinkhorn_iters: int
    hc_eps: float

    # ---- context length / RoPE (YaRN; rope_scaling.* flattened) ----
    max_position_embeddings: int
    rope_theta: float
    compress_rope_theta: float
    rope_factor: float  # rope_scaling.factor
    beta_fast: int  # rope_scaling.beta_fast
    beta_slow: int  # rope_scaling.beta_slow
    original_max_position_embeddings: int  # rope_scaling.original_max_position_embeddings

    # ---- precision / quantization (quantization_config.* flattened; unused by decode kernels) ----
    dtype: Literal["bf16", "fp8"]  # quantization_config.quant_method
    scale_fmt: Optional[Literal["ue8m0"]]  # quantization_config.scale_fmt
    expert_dtype: Optional[Literal["fp4"]]  # MoE-expert weight dtype (None = same as `dtype`)
    scale_dtype: Literal["fp32", "fp8"]  # dequant-scale storage dtype

    # ---- deployment (not consumed by the decode kernels) ----
    max_batch_size: int  # max supported batch size (cache sizing)

    # ---- derived ----
    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.qk_rope_head_dim

    @property
    def softmax_scale(self) -> float:
        return self.head_dim**-0.5

    @property
    def index_nope_head_dim(self) -> int:
        return self.index_head_dim - self.qk_rope_head_dim

    @property
    def index_weights_scale(self) -> float:
        return self.index_head_dim**-0.5 * self.index_n_heads**-0.5

    @property
    def hc_dim(self) -> int:
        return self.hc_mult * self.hidden_size

    @property
    def mix_hc(self) -> int:
        return (2 + self.hc_mult) * self.hc_mult


FLASH = DeepSeekV4Config(
    name="flash",
    hidden_size=4096,
    num_attention_heads=64,
    head_dim=512,
    qk_rope_head_dim=64,
    q_lora_rank=1024,
    o_lora_rank=1024,
    o_groups=8,
    sliding_window=128,
    rms_norm_eps=1e-6,
    vocab_size=129280,
    moe_intermediate_size=2048,
    n_routed_experts=256,
    n_shared_experts=1,
    num_experts_per_tok=6,
    scoring_func="sqrtsoftplus",
    routed_scaling_factor=1.5,
    swiglu_limit=10.0,
    num_hidden_layers=43,
    num_hash_layers=3,
    num_nextn_predict_layers=1,
    compress_ratios=(
        0,
        0,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        0,
    ),
    index_n_heads=64,
    index_head_dim=128,
    index_topk=512,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    max_position_embeddings=1_048_576,
    rope_theta=10000.0,
    compress_rope_theta=160000.0,
    rope_factor=16.0,
    beta_fast=32,
    beta_slow=1,
    original_max_position_embeddings=65536,
    dtype="fp8",
    scale_fmt="ue8m0",
    expert_dtype="fp4",
    scale_dtype="fp8",
    max_batch_size=4,
)

DECODE_BATCH = 16  # B: requests per decode step, per DP rank

DSPARK_SPEC_TOKENS = 7  # drafts the DSpark drafter proposes per request

DECODE_SEQ = 1 + DSPARK_SPEC_TOKENS  # S: tokens the target model verifies per step

DECODE_TOKENS = DECODE_BATCH * DECODE_SEQ

TP = 4  # tensor-parallel ranks per DP group

MOE_TOKENS = DECODE_TOKENS // TP


T_DYN = pl.dynamic("NATIVE_M_MARKOV_HEAD_T_DYN")

VOCAB_DYN = pl.dynamic("NATIVE_M_MARKOV_HEAD_VOCAB_DYN")

MARKOV_RANK = 256

T_TILE = 16

VOCAB_TILE = 128

SPMD_BLOCKS = 48


@pl.jit.inline
def markov_head(
    token_ids: pl.Tensor[[T_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB_DYN, MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB_DYN, MARKOV_RANK], pl.BF16],
    logits_bias: pl.Tensor[[T_DYN, VOCAB_DYN], pl.FP32],
    markov_embed: pl.Tensor[[T_DYN, MARKOV_RANK], pl.BF16],
    token_ids_ready_tid: pl.Scalar[pl.TASK_ID] = None,
):
    t_dim = pl.tensor.dim(token_ids, 0)
    t_linear = ((t_dim + T_TILE - 1) // T_TILE) * T_TILE

    with pl.spmd(
        t_dim,
        name_hint="markov_embedding",
        deps=[token_ids_ready_tid],
        allow_early_resolve=True,
    ) as embedding_tid:
        # Weight reads bypass L2.
        pl.set_cache_policy(markov_w1, pl.CachePolicy.BYPASS)
        token_idx = pl.tile.get_block_idx()
        token_id = pl.read(token_ids, [token_idx])
        token_row = pl.cast(token_id, target_type=pl.INDEX)
        markov_embed[token_idx : token_idx + 1, 0:MARKOV_RANK] = markov_w1[
            token_row : token_row + 1, 0:MARKOV_RANK
        ]

    vocab_dim = pl.tensor.dim(markov_w2, 0)
    work_items = (t_linear // T_TILE) * (vocab_dim // VOCAB_TILE)
    with pl.spmd(
        SPMD_BLOCKS,
        name_hint="markov_logits",
        deps=[embedding_tid],
    ) as logits_tid:
        # Weight reads bypass L2.
        pl.set_cache_policy(markov_w2, pl.CachePolicy.BYPASS)
        block = pl.tile.get_block_idx()
        for work_idx in pl.range(block, work_items, SPMD_BLOCKS):
            t0 = (work_idx // (vocab_dim // VOCAB_TILE)) * T_TILE
            vocab0 = (work_idx % (vocab_dim // VOCAB_TILE)) * VOCAB_TILE
            valid_rows = pl.min(T_TILE, t_dim - t0)
            embed_bf16 = pl.slice(
                markov_embed, [T_TILE, MARKOV_RANK], [t0, 0], valid_shape=[valid_rows, MARKOV_RANK]
            )
            weight_bf16 = markov_w2[vocab0 : vocab0 + VOCAB_TILE, 0:MARKOV_RANK]
            logits_tile = pl.matmul(embed_bf16, weight_bf16, b_trans=True, out_dtype=pl.FP32)
            logits_valid = pl.set_validshape(logits_tile, valid_rows, VOCAB_TILE)
            logits_bias[t0 : t0 + T_TILE, vocab0 : vocab0 + VOCAB_TILE] = logits_valid

    return logits_bias, markov_embed, embedding_tid, logits_tid


D = FLASH.hidden_size

VOCAB = FLASH.vocab_size

MAX_LOGIT_ROWS = MOE_TOKENS

_TP_DEFAULT = 2


def _parse_int_argv(name, default=None):
    for i, tok in enumerate(sys.argv):
        if tok == name and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if tok.startswith(f"{name}="):
            return int(tok.split("=", 1)[1])
    return default


TP_SIZE: int = _parse_int_argv("--tp") or _TP_DEFAULT

VOCAB_PER_TP = VOCAB // TP_SIZE

GROUP_LOGIT_ROWS = TP_SIZE * MAX_LOGIT_ROWS

FUSED_K_TILE = 256

FUSED_VOCAB_TILE = 128

MM_ROW_TILE = 32

HIDDEN_GATHER_TILE = 1024

HIDDEN_GATHER_ROW_TILE = min(GROUP_LOGIT_ROWS, 64)

PUSH_ROW_TILE = 16  # rows pushed per dispatch_push block (fewer, larger ring puts)

LOGITS_GATHER_ROW_TILE = min(MAX_LOGIT_ROWS, 8)

LOGITS_COMM_TILE = 2048

GREEDY_ROW_WIDTH = 808

GREEDY_BLOCK_ROWS = 8

FUSED_LM_HEAD_CORES = 24

AIV_LANES = 2

LANE_ROWS = MM_ROW_TILE // AIV_LANES

VOCAB_TAIL = VOCAB_PER_TP % FUSED_VOCAB_TILE

VOCAB_FULL_TILES = VOCAB_PER_TP // FUSED_VOCAB_TILE

LOGITS_COMM_TAIL = VOCAB_PER_TP % LOGITS_COMM_TILE

N_LOGITS_COMM_TILES = VOCAB_PER_TP // LOGITS_COMM_TILE

LOGITS_COMM_BLOCKS = min(FUSED_LM_HEAD_CORES, N_LOGITS_COMM_TILES + (1 if LOGITS_COMM_TAIL != 0 else 0))

LOGITS_TAIL_BLOCK = N_LOGITS_COMM_TILES % LOGITS_COMM_BLOCKS

GREEDY_INDEX_SENTINEL = 1073741824

DONE_VALUE = 1

assert MAX_LOGIT_ROWS % MM_ROW_TILE == 0, "each row block must be one owner's rows"

assert MAX_LOGIT_ROWS % PUSH_ROW_TILE == 0, "row block must cover whole rows"

assert MAX_LOGIT_ROWS % LOGITS_GATHER_ROW_TILE == 0, "logits row blocks must cover whole rows"

assert TP_SIZE == 1 or TP_SIZE % AIV_LANES == 0, "peer owners must pair across the AIV lanes"

assert GREEDY_BLOCK_ROWS * 4 % 32 == 0, "reduction result must clear the 32 B column floor"

assert GREEDY_ROW_WIDTH * 4 % 32 == 0, "block row must clear the 32 B row floor"

assert VOCAB < GREEDY_INDEX_SENTINEL, "sentinel must lose every row_min against a real id"


@pl.jit.inline(auto_scope=False)
def lm_head(
    hidden_states: pl.Tensor,
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16, pl.NZ],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
    hidden_ready_tid: pl.Scalar[pl.TASK_ID],
) -> tuple[
    pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    pl.Scalar[pl.TASK_ID],
]:
    # Scratch is allocated just outside the scope that first writes it: a
    # create_tensor inside a pl.at yields a tile, not a GM tensor view.
    selected_hidden = pl.create_tensor([MAX_LOGIT_ROWS, D], dtype=pl.BF16)

    # Publish this card's logit rows into every group member's window slot: the
    # window holds one slot per group member and each card writes only its own,
    # `tp_rank * MAX_LOGIT_ROWS`. One block per logit row, one [1, D] put per peer.
    # One block per PUSH_ROW_TILE rows: PUSH_ROW_TILE-row ring puts instead
    # of one [1, D] put per row. This cuts the per-dispatch put and notify
    # counts (512 -> 32 per card) to the MTP fixture's scale; the earlier
    # per-row form stalled lm_head_dispatch_gather on persistent re-dispatch
    # (SCHEDULER_TIMEOUT S1 with the gather task's core never exiting).
    with pl.spmd(
        MAX_LOGIT_ROWS // PUSH_ROW_TILE,
        name_hint="lm_head_dispatch_push",
        deps=[hidden_ready_tid],
    ) as _dispatch_push_tid:
        blk = pl.tile.get_block_idx()
        r0 = blk * PUSH_ROW_TILE
        hidden_rows = pl.tensor.dim(hidden_states, 0)
        for i in pl.range(PUSH_ROW_TILE):
            row = r0 + i
            source_row_raw = pl.read(logit_row_indices, [row])
            # Clamp so the load address is always inside hidden_states even if a
            # caller hands over a stale index; the -1 guard below decides whether
            # the row is actually used.
            safe_raw = pl.max(pl.min(source_row_raw, hidden_rows - 1), 0)
            selected_hidden[row : row + 1, :] = pl.full([1, D], dtype=pl.BF16, value=0.0)
            if source_row_raw >= 0:
                source_row = pl.cast(safe_raw, target_type=pl.INDEX)
                selected_hidden[row : row + 1, :] = hidden_states[source_row : source_row + 1, :]

        # Self-target rides the same put; put drains before the notify issues.
        for peer_tp in pl.range(TP_SIZE):
            pld.tensor.put(
                dst=hidden_window,
                peer=group_base + peer_tp,
                src=selected_hidden,
                dst_offsets=[tp_rank * MAX_LOGIT_ROWS + r0, 0],
                src_offsets=[r0, 0],
                shape=[PUSH_ROW_TILE, D],
            )

        # Notify folded into the push: one notify per block per source per epoch.
        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=hidden_done,
                    peer=group_base + peer_tp,
                    offsets=[tp_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

    # Start the wait after this rank has published. Signal credits persist until
    # the final clear, so peer notifies may safely arrive before this task starts;
    # anchoring avoids an idle wait occupying a core group while the push is pending.
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="lm_head_dispatch_wait", deps=[_dispatch_push_tid]
    ) as _dwait_tid:
        for owner_tp in pl.range(TP_SIZE):
            if owner_tp != tp_rank:
                pld.system.wait(
                    signal=hidden_done,
                    offsets=[owner_tp, 0],
                    expected=pl.cast(done_epoch * (MAX_LOGIT_ROWS // PUSH_ROW_TILE), pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )

    # Window -> matmul operand: a local copy split over k-tiles. Keeps the matmul's
    # auto-dep on owner_hiddens.
    owner_hiddens = pl.create_tensor([GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    with pl.spmd(
        (GROUP_LOGIT_ROWS // HIDDEN_GATHER_ROW_TILE) * (D // HIDDEN_GATHER_TILE),
        name_hint="lm_head_dispatch_gather",
        deps=[_dwait_tid, _dispatch_push_tid],
    ) as _dgather_tid:
        gblock = pl.tile.get_block_idx()
        gk0 = (gblock % (D // HIDDEN_GATHER_TILE)) * HIDDEN_GATHER_TILE
        gr0 = (gblock // (D // HIDDEN_GATHER_TILE)) * HIDDEN_GATHER_ROW_TILE
        owner_hiddens[gr0 : gr0 + HIDDEN_GATHER_ROW_TILE, gk0 : gk0 + HIDDEN_GATHER_TILE] = hidden_window[
            gr0 : gr0 + HIDDEN_GATHER_ROW_TILE, gk0 : gk0 + HIDDEN_GATHER_TILE
        ]

    # Project the owner0 row block on every TP vocabulary shard.
    # Fused cube+comm kernel: matmul one [MM_ROW_TILE, FUSED_VOCAB_TILE] tile,
    # carry the accumulator across the C->V edge with pl.aiv_shard, and
    # remote_store each lane's row half to its owner's window while the cube
    # projects the next tile. No GM staging: every row block is one owner's rows,
    # so each lane's shard is already the exact rows it pushes.
    with pl.spmd(
        FUSED_LM_HEAD_CORES,
        name_hint="lm_head_matmul_push",
        deps=[_dgather_tid],
        optimizations=[pl.cross_core_slot(slot_num=2)],
    ) as _push_tid:
        # Weight reads bypass L2.
        pl.set_cache_policy(lm_head_weight, pl.CachePolicy.BYPASS)
        lm_core = pl.tile.get_block_idx()
        vocab_base = tp_rank * VOCAB_PER_TP
        for mm_ob in pl.range(lm_core, VOCAB_FULL_TILES, FUSED_LM_HEAD_CORES):
            mm_o0 = mm_ob * FUSED_VOCAB_TILE
            for mm_rb in pl.range(MAX_LOGIT_ROWS // MM_ROW_TILE):
                mm_r0 = mm_rb * MM_ROW_TILE
                mm_acc = pl.create_tensor([MM_ROW_TILE, FUSED_VOCAB_TILE], dtype=pl.FP32)
                for mm_kb in pl.pipeline(0, D // FUSED_K_TILE, stage=2):
                    mm_k0 = mm_kb * FUSED_K_TILE
                    mm_hidden_tile = owner_hiddens[mm_r0 : mm_r0 + MM_ROW_TILE, mm_k0 : mm_k0 + FUSED_K_TILE]
                    mm_weight_tile = lm_head_weight[
                        mm_o0 : mm_o0 + FUSED_VOCAB_TILE, mm_k0 : mm_k0 + FUSED_K_TILE
                    ]
                    mm_acc = pl.matmul_acc(
                        mm_acc, mm_hidden_tile, mm_weight_tile, b_trans=True, init_cond=(mm_kb == 0)
                    )

                # The block is one owner's rows; the two lanes push its two
                # contiguous halves straight to that owner's window.
                for aiv_id in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.UP_DOWN):
                    mm_shard = pl.aiv_shard(mm_acc)
                    owner_tp = mm_rb // (MAX_LOGIT_ROWS // MM_ROW_TILE)
                    owner_r0 = (mm_rb % (MAX_LOGIT_ROWS // MM_ROW_TILE)) * MM_ROW_TILE + aiv_id * LANE_ROWS
                    pld.tensor.remote_store(
                        mm_shard,
                        logits_window,
                        group_base + owner_tp,
                        [owner_r0, vocab_base + mm_o0],
                    )

        # Ragged tail: its own matmul on one block; the accumulator is already
        # VOCAB_TAIL wide. Hoisted out of the strided loop: a dynamic branch would
        # give the accumulator two static shapes across the C->V edge.
        if VOCAB_TAIL != 0:
            if lm_core == VOCAB_FULL_TILES % FUSED_LM_HEAD_CORES:
                mm_tail_o0 = VOCAB_FULL_TILES * FUSED_VOCAB_TILE
                for tail_rb in pl.range(MAX_LOGIT_ROWS // MM_ROW_TILE):
                    tail_r0 = tail_rb * MM_ROW_TILE
                    tail_acc = pl.create_tensor([MM_ROW_TILE, VOCAB_TAIL], dtype=pl.FP32)
                    for tail_kb in pl.pipeline(0, D // FUSED_K_TILE, stage=2):
                        tail_k0 = tail_kb * FUSED_K_TILE
                        tail_hidden_tile = owner_hiddens[
                            tail_r0 : tail_r0 + MM_ROW_TILE, tail_k0 : tail_k0 + FUSED_K_TILE
                        ]
                        tail_weight_tile = lm_head_weight[
                            mm_tail_o0 : mm_tail_o0 + VOCAB_TAIL,
                            tail_k0 : tail_k0 + FUSED_K_TILE,
                        ]
                        tail_acc = pl.matmul_acc(
                            tail_acc,
                            tail_hidden_tile,
                            tail_weight_tile,
                            b_trans=True,
                            init_cond=(tail_kb == 0),
                        )

                    for aiv_id in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.UP_DOWN):
                        tail_shard = pl.aiv_shard(tail_acc)
                        owner_tp = tail_rb // (MAX_LOGIT_ROWS // MM_ROW_TILE)
                        owner_r0 = (
                            tail_rb % (MAX_LOGIT_ROWS // MM_ROW_TILE)
                        ) * MM_ROW_TILE + aiv_id * LANE_ROWS
                        pld.tensor.remote_store(
                            tail_shard,
                            logits_window,
                            group_base + owner_tp,
                            [owner_r0, vocab_base + mm_tail_o0],
                        )

        # Notify folded into the push: each block signals every peer after its own
        # stores, so a peer sees FUSED_LM_HEAD_CORES notifies per source per epoch.
        for aiv_id in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.NONE):
            for notify_pair in pl.range(TP_SIZE // AIV_LANES):
                owner_tp = notify_pair * AIV_LANES + aiv_id
                if owner_tp != tp_rank:
                    pld.system.notify(
                        target=logits_done,
                        peer=group_base + owner_tp,
                        offsets=[tp_rank, 0],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )

    # Wait only (the notify rides inside the push). deps on the push scope so the
    # wait runs alongside our own push; an unanchored wait dispatches immediately
    # and spins holding a core group.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_combine_wait", deps=[_push_tid]) as _cwait_tid:
        for src_tp in pl.range(TP_SIZE):
            if src_tp != tp_rank:
                pld.system.wait(
                    signal=logits_done,
                    offsets=[src_tp, 0],
                    expected=pl.cast(done_epoch * FUSED_LM_HEAD_CORES, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )

    # Assemble full-vocabulary logits, same vocab-tile split. deps on _cwait_tid for
    # the peers' stores; our own tiles ride the local RAW edge on logits_window.
    with pl.spmd(LOGITS_COMM_BLOCKS, name_hint="lm_head_combine_gather", deps=[_cwait_tid]) as _gather_tid:
        gblk = pl.tile.get_block_idx()
        for src_tp in pl.range(TP_SIZE):
            src_vocab_base = src_tp * VOCAB_PER_TP
            for ob in pl.range(gblk, N_LOGITS_COMM_TILES, LOGITS_COMM_BLOCKS):
                o0 = ob * LOGITS_COMM_TILE
                lo = src_vocab_base + o0
                for gr in pl.range(0, MAX_LOGIT_ROWS, LOGITS_GATHER_ROW_TILE):
                    if tp_rank == 0:
                        logits[gr : gr + LOGITS_GATHER_ROW_TILE, lo : lo + LOGITS_COMM_TILE] = logits_window[
                            gr : gr + LOGITS_GATHER_ROW_TILE, lo : lo + LOGITS_COMM_TILE
                        ]
                    else:
                        zero_logits = pl.full(
                            [LOGITS_GATHER_ROW_TILE, LOGITS_COMM_TILE], dtype=pl.FP32, value=0.0
                        )
                        logits[gr : gr + LOGITS_GATHER_ROW_TILE, lo : lo + LOGITS_COMM_TILE] = zero_logits

            if LOGITS_COMM_TAIL != 0:
                if gblk == LOGITS_TAIL_BLOCK:
                    tail_o0 = N_LOGITS_COMM_TILES * LOGITS_COMM_TILE
                    tl = src_vocab_base + tail_o0
                    for tr in pl.range(0, MAX_LOGIT_ROWS, LOGITS_GATHER_ROW_TILE):
                        if tp_rank == 0:
                            logits[tr : tr + LOGITS_GATHER_ROW_TILE, tl : tl + LOGITS_COMM_TAIL] = (
                                logits_window[tr : tr + LOGITS_GATHER_ROW_TILE, tl : tl + LOGITS_COMM_TAIL]
                            )
                        else:
                            zero_tail = pl.full(
                                [LOGITS_GATHER_ROW_TILE, LOGITS_COMM_TAIL], dtype=pl.FP32, value=0.0
                            )
                            logits[tr : tr + LOGITS_GATHER_ROW_TILE, tl : tl + LOGITS_COMM_TAIL] = zero_tail

    # Every local wait has observed all current-round peer notifies before the
    # logits gather can complete. Clear only this rank's counters so a retained
    # CommDomain can safely reuse the fixed done_epoch on the next forward.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="lm_head_signal_clear",
        deps=[_gather_tid],
    ) as _clear_tid:
        zero = pl.cast(0, pl.INT32)
        for src_tp in pl.range(TP_SIZE):
            pl.write(hidden_done, [src_tp, 0], zero)
            pl.write(logits_done, [src_tp, 0], zero)
    return logits, _clear_tid


B_DYN = pl.dynamic("NATIVE_M_DSPARK_MARKOV_B_DYN")

DSPARK_MARKOV_RANK = 256

DSPARK_QUERY_WIDTH = 5

DSPARK_QUERY_PAD = 8

DSPARK_SUPPORTED_BATCHES = (4,)

DSPARK_MAX_BATCH = max(DSPARK_SUPPORTED_BATCHES)

DSPARK_MOE_TOKENS = DSPARK_MAX_BATCH * DSPARK_QUERY_PAD

assert DSPARK_QUERY_WIDTH < DSPARK_QUERY_PAD



EPS = FLASH.rms_norm_eps

HIDDEN_TILE = 512

RMS_M_TILE = 8

LM_M_TILE = 16

LM_N_TILE = 128

LM_K_TILE = 256

MARKOV_M_TILE = 16

MARKOV_ID_PAD = 16

CONFIDENCE_PAD = 8

GREEDY_VOCAB_CHUNK = 256

GREEDY_NUM_CHUNKS = VOCAB // GREEDY_VOCAB_CHUNK

GREEDY_PARTITION_TILE = 16

GREEDY_PARTITIONS = (GREEDY_NUM_CHUNKS + GREEDY_PARTITION_TILE - 1) // GREEDY_PARTITION_TILE

GREEDY_CHUNK_PAD = 512

GREEDY_ARGMAX_ROWS = 8

NEG_INF = -3.402823e38

assert D % HIDDEN_TILE == 0

assert D % LM_K_TILE == 0

assert VOCAB % LM_N_TILE == 0

assert VOCAB % GREEDY_VOCAB_CHUNK == 0

assert GREEDY_NUM_CHUNKS <= GREEDY_CHUNK_PAD

assert CONFIDENCE_PAD * 4 == 32

for _batch in DSPARK_SUPPORTED_BATCHES:
    assert (_batch * DSPARK_QUERY_PAD) % LM_M_TILE == 0


@pl.jit.inline
def normalize_head_hidden(
    head_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    normalized: pl.Tensor[[DSPARK_MOE_TOKENS, D], pl.BF16],
    head_ready_tid: pl.Scalar[pl.TASK_ID],
):
    batch = pl.tensor.dim(head_hidden, 0)
    active_tokens = batch * DSPARK_QUERY_WIDTH
    padded_tokens = batch * DSPARK_QUERY_PAD
    hidden_flat = pl.reshape(head_hidden, [active_tokens, D])
    padded_hidden = pl.create_tensor([DSPARK_MOE_TOKENS, D], dtype=pl.BF16)
    hidden_blocks = D // HIDDEN_TILE
    with pl.spmd(
        (DSPARK_MOE_TOKENS // RMS_M_TILE) * hidden_blocks,
        name_hint="dspark_hidden_zero",
        deps=[head_ready_tid],
    ) as hidden_zero_tid:
        zero_task = pl.tile.get_block_idx()
        zero_row = (zero_task // hidden_blocks) * RMS_M_TILE
        zero_col = (zero_task % hidden_blocks) * HIDDEN_TILE
        padded_hidden[
            zero_row : zero_row + RMS_M_TILE,
            zero_col : zero_col + HIDDEN_TILE,
        ] = pl.full([RMS_M_TILE, HIDDEN_TILE], dtype=pl.BF16, value=0.0)
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dspark_hidden_pad",
        deps=[hidden_zero_tid],
    ) as hidden_pad_tid:
        for token in pl.range(active_tokens):
            padded_hidden[token : token + 1, :] = hidden_flat[token : token + 1, :]

    # Keep this normalization local. Passing a dynamically sized temporary to
    # the generic rms_norm inline kernel loses its inferred tensor metadata
    # during JIT specialization.
    with pl.spmd(
        padded_tokens // RMS_M_TILE,
        name_hint="dspark_final_norm",
        deps=[hidden_pad_tid],
        allow_early_resolve=True,
    ) as final_norm_tid:
        row_block = pl.tile.get_block_idx()
        row_offset = row_block * RMS_M_TILE
        square_sum = pl.full([1, RMS_M_TILE], dtype=pl.FP32, value=0.0)
        for hidden_block in pl.pipeline(D // HIDDEN_TILE, stage=2):
            hidden_offset = hidden_block * HIDDEN_TILE
            rms_hidden_tile = pl.cast(
                padded_hidden[
                    row_offset : row_offset + RMS_M_TILE,
                    hidden_offset : hidden_offset + HIDDEN_TILE,
                ],
                target_type=pl.FP32,
            )
            square_sum = pl.add(
                square_sum,
                pl.reshape(
                    pl.row_sum(pl.mul(rms_hidden_tile, rms_hidden_tile)),
                    [1, RMS_M_TILE],
                ),
            )
        inv_rms = pl.reshape(
            pl.rsqrt(
                pl.add(pl.mul(square_sum, 1.0 / D), EPS),
                high_precision=True,
            ),
            [RMS_M_TILE, 1],
        )
        for hidden_block in pl.pipeline(D // HIDDEN_TILE, stage=2):
            hidden_offset = hidden_block * HIDDEN_TILE
            rms_hidden_tile = pl.cast(
                padded_hidden[
                    row_offset : row_offset + RMS_M_TILE,
                    hidden_offset : hidden_offset + HIDDEN_TILE,
                ],
                target_type=pl.FP32,
            )
            norm_tile = pl.cast(
                pl.reshape(
                    final_norm_weight[hidden_offset : hidden_offset + HIDDEN_TILE],
                    [1, HIDDEN_TILE],
                ),
                target_type=pl.FP32,
            )
            normalized[
                row_offset : row_offset + RMS_M_TILE,
                hidden_offset : hidden_offset + HIDDEN_TILE,
            ] = pl.cast(
                pl.col_expand_mul(
                    pl.row_expand_mul(rms_hidden_tile, inv_rms),
                    norm_tile,
                ),
                target_type=pl.BF16,
                mode="rint",
            )
    return final_norm_tid


@pl.jit.inline
def greedy_markov_step(
    head_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16],
    base_logits: pl.Tensor,
    num_sampled: pl.Tensor[[B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[B_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    confidence_head_weight: pl.Tensor[[1, D + DSPARK_MARKOV_RANK], pl.FP32],
    draft_token_scratch: pl.Tensor[[MARKOV_M_TILE, MARKOV_ID_PAD], pl.INT32],
    confidence_scratch: pl.Tensor[[DSPARK_QUERY_WIDTH, MARKOV_M_TILE], pl.FP32],
    base_logits_ready_tid: pl.Scalar[pl.TASK_ID],
    start_tid: pl.Scalar[pl.TASK_ID],
    step: pl.Scalar[pl.INT32],
):
    batch = pl.tensor.dim(num_sampled, 0)
    previous_token_ids = pl.create_tensor([batch], dtype=pl.INT64)
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dspark_markov_previous_tokens",
        deps=[start_tid],
    ) as previous_tokens_tid:
        for request in pl.range(batch):
            previous_token = pl.cast(0, pl.INT64)
            if step == 0:
                previous_token = pl.read(next_prefill_tokens, [request])
                if pl.read(num_sampled, [request]) > 0:
                    previous_token = pl.read(last_sampled, [request])
            if step > 0:
                previous_token = pl.cast(
                    pl.read(draft_token_scratch, [request, step - 1]),
                    pl.INT64,
                )
            pl.write(previous_token_ids, [request], previous_token)

    markov_bias = pl.create_tensor([batch, VOCAB], dtype=pl.FP32)
    markov_embedding = pl.create_tensor([batch, DSPARK_MARKOV_RANK], dtype=pl.BF16)
    (
        markov_bias,
        markov_embedding,
        markov_embedding_tid,
        markov_logits_tid,
    ) = markov_head(
        previous_token_ids,
        markov_w1,
        markov_w2,
        markov_bias,
        markov_embedding,
        previous_tokens_tid,
    )

    confidence_blocks = (batch + CONFIDENCE_PAD - 1) // CONFIDENCE_PAD
    with pl.spmd(
        confidence_blocks,
        name_hint="dspark_confidence_head",
        deps=[markov_embedding_tid],
    ) as confidence_tid:
        # Weight reads bypass L2.
        pl.set_cache_policy(confidence_head_weight, pl.CachePolicy.BYPASS)
        confidence_block = pl.tile.get_block_idx()
        request_base = confidence_block * CONFIDENCE_PAD
        valid_rows = pl.min(CONFIDENCE_PAD, batch - request_base)
        confidence_logit = pl.full(
            [1, CONFIDENCE_PAD],
            dtype=pl.FP32,
            value=0.0,
        )
        for hidden_block in pl.range(D // HIDDEN_TILE):
            hidden_offset = hidden_block * HIDDEN_TILE
            hidden_bf16 = pl.reshape(
                pl.slice(
                    head_hidden,
                    [CONFIDENCE_PAD, 1, HIDDEN_TILE],
                    [request_base, step, hidden_offset],
                    valid_shape=[valid_rows, 1, HIDDEN_TILE],
                ),
                [CONFIDENCE_PAD, HIDDEN_TILE],
            )
            hidden_fp32 = pl.cast(hidden_bf16, target_type=pl.FP32)
            hidden_weight = confidence_head_weight[
                0:1,
                hidden_offset : hidden_offset + HIDDEN_TILE,
            ]
            confidence_logit = pl.add(
                confidence_logit,
                pl.reshape(
                    pl.row_sum(pl.col_expand_mul(hidden_fp32, hidden_weight)),
                    [1, CONFIDENCE_PAD],
                ),
            )
        markov_bf16 = pl.slice(
            markov_embedding,
            [CONFIDENCE_PAD, DSPARK_MARKOV_RANK],
            [request_base, 0],
            valid_shape=[valid_rows, DSPARK_MARKOV_RANK],
        )
        markov_fp32 = pl.cast(markov_bf16, target_type=pl.FP32)
        markov_weight = confidence_head_weight[
            0:1,
            D : D + DSPARK_MARKOV_RANK,
        ]
        confidence_logit = pl.add(
            confidence_logit,
            pl.reshape(
                pl.row_sum(pl.col_expand_mul(markov_fp32, markov_weight)),
                [1, CONFIDENCE_PAD],
            ),
        )
        confidence_prob = pl.recip(pl.add(pl.exp(pl.neg(confidence_logit)), 1.0))
        confidence_scratch[
            step : step + 1,
            request_base : request_base + CONFIDENCE_PAD,
        ] = confidence_prob

    partial_scores = pl.create_tensor([batch, GREEDY_PARTITIONS, MARKOV_ID_PAD], dtype=pl.FP32)
    partial_ids = pl.create_tensor([batch, GREEDY_PARTITIONS, MARKOV_ID_PAD], dtype=pl.INT32)
    with pl.spmd(
        batch * GREEDY_PARTITIONS,
        name_hint="greedy_vocab_partitions",
        deps=[base_logits_ready_tid, confidence_tid, markov_logits_tid],
    ) as partial_tid:
        block = pl.tile.get_block_idx()
        request = block // GREEDY_PARTITIONS
        partition = block % GREEDY_PARTITIONS
        source_row = request * DSPARK_QUERY_WIDTH + step
        chunk_maxima = pl.full([GREEDY_ARGMAX_ROWS, GREEDY_CHUNK_PAD], dtype=pl.FP32, value=NEG_INF)
        chunk_token_ids = pl.full([1, GREEDY_CHUNK_PAD], dtype=pl.INT32, value=0)
        for local_chunk in pl.range(GREEDY_PARTITION_TILE):
            chunk = partition * GREEDY_PARTITION_TILE + local_chunk
            if chunk < GREEDY_NUM_CHUNKS:
                vocab_offset = chunk * GREEDY_VOCAB_CHUNK
                scores = pl.full([GREEDY_ARGMAX_ROWS, GREEDY_VOCAB_CHUNK], dtype=pl.FP32, value=NEG_INF)
                base = pl.slice(base_logits, [1, GREEDY_VOCAB_CHUNK], [source_row, vocab_offset])
                bias = markov_bias[request : request + 1, vocab_offset : vocab_offset + GREEDY_VOCAB_CHUNK]
                scores[0:1, 0:GREEDY_VOCAB_CHUNK] = pl.add(base, bias)
                local_winner = pl.row_argmax(scores)
                local_token = pl.read(local_winner, [0, 0])
                local_value = pl.read(scores, [0, pl.cast(local_token, pl.INDEX)])
                pl.write(chunk_maxima, [0, local_chunk], local_value)
                pl.write(chunk_token_ids, [0, local_chunk], pl.cast(vocab_offset, pl.INT32) + local_token)
        winning_chunk = pl.read(pl.row_argmax(chunk_maxima), [0, 0])
        winner_col = pl.cast(winning_chunk, pl.INDEX)
        winning_score = pl.read(chunk_maxima, [0, winner_col])
        winning_token = pl.read(chunk_token_ids, [0, winner_col])
        pl.write(partial_scores, [request, partition, 0], winning_score)
        pl.write(partial_ids, [request, partition, 0], winning_token)
    with pl.spmd(MARKOV_M_TILE, name_hint="greedy_partition_reduce", deps=[partial_tid]) as greedy_tid:
        request = pl.tile.get_block_idx()
        if request < batch:
            maxima = pl.full([GREEDY_ARGMAX_ROWS, GREEDY_CHUNK_PAD], dtype=pl.FP32, value=NEG_INF)
            ids = pl.full([1, GREEDY_CHUNK_PAD], dtype=pl.INT32, value=0)
            for partition in pl.range(GREEDY_PARTITIONS):
                pl.write(maxima, [0, partition], pl.read(partial_scores, [request, partition, 0]))
                pl.write(ids, [0, partition], pl.read(partial_ids, [request, partition, 0]))
            winning_partition = pl.read(pl.row_argmax(maxima), [0, 0])
            winning_token = pl.read(ids, [0, pl.cast(winning_partition, pl.INDEX)])
            token_row = draft_token_scratch[request : request + 1, 0:MARKOV_ID_PAD]
            if step == 0:
                pl.write(token_row, [0, 0], winning_token)
            if step == 1:
                pl.write(token_row, [0, 1], winning_token)
            if step == 2:
                pl.write(token_row, [0, 2], winning_token)
            if step == 3:
                pl.write(token_row, [0, 3], winning_token)
            if step == 4:
                pl.write(token_row, [0, 4], winning_token)
            if step == 5:
                pl.write(token_row, [0, 5], winning_token)
            if step == 6:
                pl.write(token_row, [0, 6], winning_token)
            draft_token_scratch[request : request + 1, 0:MARKOV_ID_PAD] = token_row

    return greedy_tid


@pl.jit.inline
def sample_from_base_logits(
    head_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16],
    base_logits: pl.Tensor,
    num_sampled: pl.Tensor[[B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[B_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    confidence_head_weight: pl.Tensor[[1, D + DSPARK_MARKOV_RANK], pl.FP32],
    draft_token_ids: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.INT32],
    confidence_probs: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.FP32],
    base_logits_ready_tid: pl.Scalar[pl.TASK_ID],
    output_owner_tp_rank: pl.Scalar[pl.INT32],
):
    batch = pl.tensor.dim(num_sampled, 0)
    draft_token_scratch = pl.create_tensor(
        [MARKOV_M_TILE, MARKOV_ID_PAD],
        dtype=pl.INT32,
    )
    confidence_scratch = pl.create_tensor(
        [DSPARK_QUERY_WIDTH, MARKOV_M_TILE],
        dtype=pl.FP32,
    )
    with pl.spmd(
        batch,
        name_hint="dspark_markov_token_scratch_zero",
    ) as token_scratch_zero_tid:
        request = pl.tile.get_block_idx()
        draft_token_scratch[
            request : request + 1,
            0:MARKOV_ID_PAD,
        ] = pl.full([1, MARKOV_ID_PAD], dtype=pl.INT32, value=0)
    step_0_tid = greedy_markov_step(
        head_hidden,
        base_logits,
        num_sampled,
        last_sampled,
        next_prefill_tokens,
        markov_w1,
        markov_w2,
        confidence_head_weight,
        draft_token_scratch,
        confidence_scratch,
        base_logits_ready_tid,
        token_scratch_zero_tid,
        pl.cast(0, pl.INT32),
    )
    step_1_tid = greedy_markov_step(
        head_hidden,
        base_logits,
        num_sampled,
        last_sampled,
        next_prefill_tokens,
        markov_w1,
        markov_w2,
        confidence_head_weight,
        draft_token_scratch,
        confidence_scratch,
        base_logits_ready_tid,
        step_0_tid,
        pl.cast(1, pl.INT32),
    )
    step_2_tid = greedy_markov_step(
        head_hidden,
        base_logits,
        num_sampled,
        last_sampled,
        next_prefill_tokens,
        markov_w1,
        markov_w2,
        confidence_head_weight,
        draft_token_scratch,
        confidence_scratch,
        base_logits_ready_tid,
        step_1_tid,
        pl.cast(2, pl.INT32),
    )
    step_3_tid = greedy_markov_step(
        head_hidden,
        base_logits,
        num_sampled,
        last_sampled,
        next_prefill_tokens,
        markov_w1,
        markov_w2,
        confidence_head_weight,
        draft_token_scratch,
        confidence_scratch,
        base_logits_ready_tid,
        step_2_tid,
        pl.cast(3, pl.INT32),
    )
    step_4_tid = greedy_markov_step(
        head_hidden,
        base_logits,
        num_sampled,
        last_sampled,
        next_prefill_tokens,
        markov_w1,
        markov_w2,
        confidence_head_weight,
        draft_token_scratch,
        confidence_scratch,
        base_logits_ready_tid,
        step_3_tid,
        pl.cast(4, pl.INT32),
    )
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dspark_markov_token_output_copy",
        deps=[step_4_tid],
    ) as token_output_ready_tid:
        if output_owner_tp_rank == 0:
            for request in pl.range(batch):
                for output_step in pl.range(DSPARK_QUERY_WIDTH):
                    pl.write(
                        draft_token_ids,
                        [request, output_step],
                        pl.read(draft_token_scratch, [request, output_step]),
                    )
                    pl.write(
                        confidence_probs,
                        [request, output_step],
                        pl.read(confidence_scratch, [output_step, request]),
                    )
        else:
            for request in pl.range(batch):
                for output_step in pl.range(DSPARK_QUERY_WIDTH):
                    pl.write(draft_token_ids, [request, output_step], pl.cast(0, pl.INT32))
                    pl.write(confidence_probs, [request, output_step], pl.cast(0.0, pl.FP32))
    return draft_token_ids, confidence_probs, token_output_ready_tid


@pl.jit.inline(auto_scope=False)
def markov_inline(
    head_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16, pl.NZ],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    num_sampled: pl.Tensor[[B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[B_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    confidence_head_weight: pl.Tensor[[1, D + DSPARK_MARKOV_RANK], pl.FP32],
    draft_token_ids: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.INT32],
    confidence_probs: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.FP32],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    head_ready_tid: pl.Scalar[pl.TASK_ID],
):
    head_hidden.bind_dynamic(0, B_DYN)
    num_sampled.bind_dynamic(0, B_DYN)
    last_sampled.bind_dynamic(0, B_DYN)
    next_prefill_tokens.bind_dynamic(0, B_DYN)
    draft_token_ids.bind_dynamic(0, B_DYN)
    confidence_probs.bind_dynamic(0, B_DYN)
    normalized = pl.create_tensor([DSPARK_MOE_TOKENS, D], dtype=pl.BF16)
    final_norm_tid = normalize_head_hidden(
        head_hidden,
        final_norm_weight,
        normalized,
        head_ready_tid,
    )
    base_logits = pl.create_tensor([MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
    base_logits, base_logits_tid = lm_head(
        normalized,
        lm_head_weight,
        logit_row_indices,
        base_logits,
        hidden_window,
        hidden_done,
        logits_window,
        logits_done,
        group_base,
        tp_rank,
        DONE_VALUE,
        final_norm_tid,
    )
    draft_token_ids, confidence_probs, token_output_ready_tid = sample_from_base_logits(
        head_hidden,
        base_logits,
        num_sampled,
        last_sampled,
        next_prefill_tokens,
        markov_w1,
        markov_w2,
        confidence_head_weight,
        draft_token_ids,
        confidence_probs,
        base_logits_tid,
        tp_rank,
    )
    return draft_token_ids, confidence_probs, token_output_ready_tid
