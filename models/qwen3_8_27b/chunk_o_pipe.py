# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet chunk_o as an explicit cube/vector pipeline.

Same arithmetic as `chunk_o.py`, with the cube and vector programs written
separately and joined by GM mailboxes and three events:

    AIC  Q@S, Q@K^T  --qs,qk-->  AIV gate  --gated-->  AIC gated@V  --intra-->  AIV out

One chunk per block; the pipeline runs over the 48 independent value heads, the
cube `PRE_LAUNCH` heads ahead of the vector. `inter` is rebuilt at the consume
step from the `qs` mailbox slot, which the ring holds until then.
"""
import pypto.language as pl

from chunk_cumsum import CHUNK_TILE
from config import QWEN3_8_27B
from reference import frob_ok

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")                 # tokens
STATE_DYN = pl.dynamic("GDN_STATE_DYN")     # (T // CHUNK_TILE) * H * D, the snapshot rows

# model config
H = QWEN3_8_27B.linear_num_value_heads      # value heads
HG = QWEN3_8_27B.linear_num_key_heads       # QK heads; H // HG value heads share one
D = QWEN3_8_27B.linear_value_head_dim       # head dimension
GRP = H // HG                               # value heads sharing one key head
KEY_WIDTH = HG * D
VAL_WIDTH = H * D
# validation
DEFAULT_CASES = (128, 256)
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len

# No a5 path (pto-isa#612): expand_mixed_kernel_pass rejects the authored cross-core
# events below -- "Direct C2V promotion cannot mix an automatic write phase with
# authored cross-core events" -- so -p a5 fails at compile, with or without --sim.

# tiling
AIV_LANES = 2           # vector lanes per cube core, each owning half the chunk rows
# One head of run-ahead is enough to cover the hand-off, and deeper rings cost
# more in mailbox traffic than the extra overlap returns: 614.8 / 627.0 / 644.0
# us at 1 / 2 / 3 (T = 8192, a2a3), against 880.8 for the auto-synchronised form.
PRE_LAUNCH = 1          # heads the cube runs ahead of the vector
# `qs` is written at the produce step and read at the CONSUME step, so it lives
# pre_launch steps longer than the other three mailboxes, and the ring has to be
# sized for it. The cube's consume of step s waits one GATED_READY, which the
# vector sets during its produce of s -- before that iteration's own consume has
# run -- so the cube only ever knows the vector has consumed up to
# s - pre_launch - 1. That puts the boundary at 2*pre_launch + 1 exactly, which
# frees a slot in the same iteration it is read: at that depth pre_launch=1
# passed once and returned 1% of the output wrong the next run. The extra slot
# is the margin.
SLOTS = 2 * PRE_LAUNCH + 2

# event ids, one per hand-off direction
QK_READY = 0            # AIC -> AIV, FP32 Q@S and Q@K^T
GATED_READY = 1         # AIV -> AIC, FP16 gated scores
INTRA_READY = 2         # AIC -> AIV, FP32 gated @ V


# pl.split_aiv's count is the hardware AIV sub-core count, fixed at 2.
if CHUNK_TILE % AIV_LANES:
    raise ValueError(f"CHUNK_TILE={CHUNK_TILE} must divide by AIV_LANES={AIV_LANES}")
if H % HG:
    raise ValueError(f"H={H} must be a multiple of HG={HG}")
LANE_ROWS = CHUNK_TILE // AIV_LANES
MAILBOX_STRIDE = SLOTS * CHUNK_TILE          # mailbox rows one chunk block owns


def _gdn_chunk_o_pipe(
    q: pl.Tensor[[T_DYN, HG, D], pl.FP16],
    k: pl.Tensor[[T_DYN, HG, D], pl.FP16],
    v: pl.Tensor[[T_DYN, H, D], pl.FP16],
    state: pl.Tensor[[STATE_DYN, D], pl.FP16],
    g_sum: pl.Tensor[[H, T_DYN], pl.FP32],
    mask: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP32],
    o_out: pl.Out[pl.Tensor[[T_DYN, H, D], pl.FP16]],
):
    """The token count must be a multiple of CHUNK_TILE, and `state` hold (T // CHUNK_TILE) * H * D rows."""
    q.bind_dynamic(0, T_DYN)
    k.bind_dynamic(0, T_DYN)
    v.bind_dynamic(0, T_DYN)
    state.bind_dynamic(0, STATE_DYN)
    g_sum.bind_dynamic(1, T_DYN)
    o_out.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(q, 0)
    nchunk = t_dim // CHUNK_TILE                # state snapshots, one per chunk
    mailbox_rows = nchunk * MAILBOX_STRIDE
    q_flat = pl.reshape(q, [t_dim, KEY_WIDTH])
    k_flat = pl.reshape(k, [t_dim, KEY_WIDTH])
    v_flat = pl.reshape(v, [t_dim, VAL_WIDTH])
    o_flat = pl.reshape(o_out, [t_dim, VAL_WIDTH])

    # GM mailboxes: one disjoint slot ring per chunk block, so no two cores
    # and no two heads in flight share rows.
    qs_mb = pl.create_tensor([mailbox_rows, D], dtype=pl.FP32)
    qk_mb = pl.create_tensor([mailbox_rows, CHUNK_TILE], dtype=pl.FP32)
    gated_mb = pl.create_tensor([mailbox_rows, CHUNK_TILE], dtype=pl.FP16)
    intra_mb = pl.create_tensor([mailbox_rows, D], dtype=pl.FP32)
    ffts = pl.create_tensor([256], dtype=pl.INT64)

    for c0 in pl.spmd(nchunk, name_hint="chunk_o_pipe"):
        pl.system.set_ffts(ffts)
        t0 = c0 * CHUNK_TILE
        block_row = c0 * MAILBOX_STRIDE

        # ---- cube: Q@S and Q@K^T for head `tick`, gated@V for `tick - pre_launch`
        for tick in pl.range(H + PRE_LAUNCH):
            if tick < H:
                prod_row = block_row + (tick % SLOTS) * CHUNK_TILE
                dg0 = (tick // GRP) * D
                qc = pl.load(q_flat, [t0, dg0], [CHUNK_TILE, D], target_memory=pl.MemorySpace.Mat)
                s_row = c0 * VAL_WIDTH + tick * D
                s_blk = pl.load(state, [s_row, 0], [D, D], target_memory=pl.MemorySpace.Mat)
                qs = pl.matmul(qc, s_blk, out_dtype=pl.FP32)
                pl.store(qs, [prod_row, 0], qs_mb)
                kc = pl.load(k_flat, [t0, dg0], [CHUNK_TILE, D], target_memory=pl.MemorySpace.Mat)
                qk = pl.matmul(qc, pl.tile.transpose_view(kc), out_dtype=pl.FP32)
                pl.store(qk, [prod_row, 0], qk_mb)
                pl.system.sync_set(QK_READY, pipe=pl.PipeType.FIX, ffts_mode=2,
                                   core_type=pl.KernelType.AIC)
            if tick >= PRE_LAUNCH:
                step = tick - PRE_LAUNCH
                cons_row = block_row + (step % SLOTS) * CHUNK_TILE
                # V does not depend on the gate, so its load overlaps it.
                vc = pl.load(v_flat, [t0, step * D], [CHUNK_TILE, D],
                             target_memory=pl.MemorySpace.Mat)
                pl.system.sync_wait(GATED_READY, pipe=pl.PipeType.MTE2,
                                    core_type=pl.KernelType.AIC)
                gated = pl.load(gated_mb, [cons_row, 0], [CHUNK_TILE, CHUNK_TILE],
                                target_memory=pl.MemorySpace.Mat)
                intra = pl.matmul(gated, vc, out_dtype=pl.FP32)
                pl.store(intra, [cons_row, 0], intra_mb)
                pl.system.sync_set(INTRA_READY, pipe=pl.PipeType.FIX, ffts_mode=2,
                                   core_type=pl.KernelType.AIC)

        # ---- vector: gate for head `tick`, output for `tick - pre_launch`
        for lane in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.NONE):
            pl.system.set_ffts(ffts)
            lane_row = lane * LANE_ROWS
            lane_t0 = t0 + lane_row

            for tick in pl.range(H + PRE_LAUNCH):
                if tick < H:
                    prod_row = block_row + (tick % SLOTS) * CHUNK_TILE + lane_row
                    pl.system.sync_wait(QK_READY, pipe=pl.PipeType.MTE2,
                                        core_type=pl.KernelType.AIV)
                    qk_tile = pl.load(qk_mb, [prod_row, 0], [LANE_ROWS, CHUNK_TILE],
                                      target_memory=pl.MemorySpace.Vec)
                    # gate[i, j] = exp(min(g_i - g_j, 0)) * causal, the rows this
                    # lane owns against every column of the chunk.
                    g_full = pl.load(g_sum, [tick, t0], [1, CHUNK_TILE],
                                     target_memory=pl.MemorySpace.Vec)
                    g_lane = pl.load(g_sum, [tick, lane_t0], [1, LANE_ROWS],
                                     target_memory=pl.MemorySpace.Vec)
                    g_col = pl.reshape(g_lane, [LANE_ROWS, 1])
                    zero = pl.tile.full([LANE_ROWS, CHUNK_TILE], dtype=pl.FP32, value=0.0)
                    diff = pl.col_expand_sub(pl.row_expand_add(zero, g_col), g_full)
                    decay = pl.exp(pl.minimum(diff, 0.0))
                    mask_tile = pl.load(mask, [lane_row, 0], [LANE_ROWS, CHUNK_TILE],
                                        target_memory=pl.MemorySpace.Vec)
                    gate = pl.mul(decay, mask_tile)
                    gate_scores = pl.cast(pl.mul(qk_tile, gate), target_type=pl.FP16,
                                          mode="rint")
                    pl.store(gate_scores, [prod_row, 0], gated_mb)
                    pl.system.sync_set(GATED_READY, pipe=pl.PipeType.MTE3, ffts_mode=2,
                                       core_type=pl.KernelType.AIV)
                if tick >= PRE_LAUNCH:
                    step = tick - PRE_LAUNCH
                    cons_row = block_row + (step % SLOTS) * CHUNK_TILE + lane_row
                    pl.system.sync_wait(INTRA_READY, pipe=pl.PipeType.MTE2,
                                        core_type=pl.KernelType.AIV)
                    intra_tile = pl.load(intra_mb, [cons_row, 0], [LANE_ROWS, D],
                                         target_memory=pl.MemorySpace.Vec)
                    qs_tile = pl.load(qs_mb, [cons_row, 0], [LANE_ROWS, D],
                                      target_memory=pl.MemorySpace.Vec)
                    g_step = pl.load(g_sum, [step, lane_t0], [1, LANE_ROWS],
                                     target_memory=pl.MemorySpace.Vec)
                    eg = pl.exp(pl.reshape(g_step, [LANE_ROWS, 1]))
                    inter = pl.row_expand_mul(qs_tile, eg)
                    o_tile = pl.cast(pl.add(inter, intra_tile), target_type=pl.FP16,
                                     mode="rint")
                    pl.store(o_tile, [lane_t0, step * D], o_flat)
    return o_out


# The explicit pipeline keeps GM mailboxes and the rescale queue live at once, so the task
# allocator needs more than the runtime's own 256 MiB default: at T = 65536 that default
# fails with `orch_error_code=2 HEAP_RING_DEADLOCK`, which arrives as `finalize_native_run
# failed with code -2` and names nothing about the heap. **Measured floor 512 MiB at
# T = 65536**; set one power of two above it, as `mlp` and `gated_attention` are. The
# heapless entry ran at T = 8192 and 32768, which is why this was missed.
# Bytes per ring, one entry per scope depth, 0 leaving a ring at its default. The stage
# scopes allocate on ring 1; a scalar here would reserve the same bytes on all four.
PIPE_RING_HEAP = [0, 1024 * 1024 * 1024, 0, 0]


gdn_chunk_o_pipe = pl.jit.inline(auto_scope=False)(_gdn_chunk_o_pipe)
gdn_chunk_o_pipe_test = pl.jit(auto_scope=False)(_gdn_chunk_o_pipe)


def build_tensor_specs(t: int = BENCH_SEQ):
    import chunk_o

    return chunk_o.build_tensor_specs(t=t)


def golden_gdn_chunk_o_pipe(tensors):
    import chunk_o

    chunk_o.golden_gdn_chunk_o(tensors)


if __name__ == "__main__":
    import argparse
    import os

    from golden import run

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None,
                        help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--rounds", type=int, default=0)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()
    cases = [args.seq_len] if args.seq_len else list(DEFAULT_CASES)
    if len(cases) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one shape; name it with --seq-len")
    if args.seq_len and args.seq_len % CHUNK_TILE:
        parser.error(f"--seq-len must be a multiple of CHUNK_TILE={CHUNK_TILE}")
    compile_only = args.compile_only
    if args.rounds:
        os.environ["PYPTO_BENCH"] = "1"
        os.environ["PYPTO_BENCH_ROUNDS"] = str(args.rounds)
        os.environ["PYPTO_BENCH_WARMUP"] = "5"

    for t in cases:
        print(f"[case] chunk_o_pipe T={t}", flush=True)
        result = run(
            fn=gdn_chunk_o_pipe_test,
            specs=build_tensor_specs(t=t),
            golden_fn=None if compile_only else golden_gdn_chunk_o_pipe,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode,
                        device_id=args.device, ring_heap=PIPE_RING_HEAP),
            compare_fn={"o_out": frob_ok()},
            compile_only=compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(f"FAILED at T={t}")
