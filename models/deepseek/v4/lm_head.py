# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI marker: run on >=2 NPUs via $DEVICE_RANGE instead of single $DEVICE_ID
"""DeepSeek-V4 LM head projection with TP vocab all-gather.

The input hidden states are expected to have already passed the final RMSNorm,
matching cann-recipes' ``DeepseekV3Model.forward`` + ``forward_lm_head`` split.

Each TP rank owns one contiguous vocabulary shard of ``lm_head_weight`` and
computes local logits for that shard. The distributed program then publishes
every shard to every TP rank through an HCCL window so each rank's output
contains full-vocabulary logits.
"""

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

from config import DECODE_TOKENS, FLASH as M, LM_HEAD_TP_SIZE


# Tensor shapes and loop trip counts are static in the frontend, so the TP
# world size is a build-time constant. Deployment uses LM_HEAD_TP_SIZE=8; this
# demo validates on 2 NPUs.
TP_SIZE = 2

T = DECODE_TOKENS  # 128 decode tokens.
D = M.hidden_size  # 4096 hidden size.
VOCAB = M.vocab_size  # 129280 vocabulary size.

LM_HEAD_K_CHUNK = 128  # K tile width; D / 128 = 32 matmul accumulation blocks.
VOCAB_CHUNK = 160  # Vocab tile width; with TP=2, VOCAB_PER_TP / 160 = 404 blocks.
T_TILE = 16  # Token tile height; T / 16 = 8 token blocks.

assert D % LM_HEAD_K_CHUNK == 0
assert T % T_TILE == 0
assert VOCAB % VOCAB_CHUNK == 0
assert VOCAB % TP_SIZE == 0
assert TP_SIZE <= LM_HEAD_TP_SIZE

K_BLOCKS = D // LM_HEAD_K_CHUNK  # 32.
VOCAB_PER_TP = VOCAB // TP_SIZE  # 64640 when TP_SIZE=2.
VOCAB_BLOCKS_PER_TP = VOCAB_PER_TP // VOCAB_CHUNK  # 404 blocks per TP shard.

assert VOCAB_PER_TP % VOCAB_CHUNK == 0


@pl.jit.inline
def lm_head(
    hidden_states: pl.Tensor[[T, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logits_shard: pl.Out[pl.Tensor[[T, VOCAB_PER_TP], pl.FP32]],
) -> pl.Tensor[[T, VOCAB_PER_TP], pl.FP32]:
    for t0 in pl.parallel(0, T, T_TILE):
        for ob in pl.parallel(VOCAB_BLOCKS_PER_TP):
            o0 = ob * VOCAB_CHUNK
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head"):
                hidden_chunk = pl.slice(hidden_states, [T_TILE, LM_HEAD_K_CHUNK], [t0, 0])
                weight_chunk = pl.slice(
                    lm_head_weight, [VOCAB_CHUNK, LM_HEAD_K_CHUNK], [o0, 0]
                )
                acc = pl.matmul(hidden_chunk, weight_chunk, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.range(1, K_BLOCKS):
                    k0 = kb * LM_HEAD_K_CHUNK
                    hidden_chunk = pl.slice(hidden_states, [T_TILE, LM_HEAD_K_CHUNK], [t0, k0])
                    weight_chunk = pl.slice(
                        lm_head_weight, [VOCAB_CHUNK, LM_HEAD_K_CHUNK], [o0, k0]
                    )
                    acc = pl.matmul_acc(acc, hidden_chunk, weight_chunk, b_trans=True)
                logits_shard = pl.assemble(logits_shard, acc, [t0, o0])
    return logits_shard


@pl.jit.incore
def gather_step(
    logits_shard: pl.Tensor[[T, VOCAB_PER_TP], pl.FP32],
    logits: pl.Out[pl.Tensor[[T, VOCAB], pl.FP32]],
    logits_window: pld.DistributedTensor[[T, VOCAB], pl.FP32],
    gather_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T, VOCAB], pl.FP32]:
    # Publish this rank's contiguous vocab shard into every peer's full
    # logits window. The local rank writes its own shard directly to
    # the host-backed output below; routing it through the comm window
    # would add a large self-remote path that is unnecessary here.
    vocab_base = my_rank * VOCAB_PER_TP
    for peer in pl.range(TP_SIZE):
        if peer != my_rank:
            for t0 in pl.range(0, T, T_TILE):
                for ob in pl.range(VOCAB_BLOCKS_PER_TP):
                    o0 = ob * VOCAB_CHUNK
                    tile = pl.load(logits_shard, [t0, o0], [T_TILE, VOCAB_CHUNK])
                    pld.tile.remote_store(
                        tile,
                        target=logits_window,
                        peer=peer,
                        offsets=[t0, vocab_base + o0],
                    )

    for peer in pl.range(TP_SIZE):
        if peer != my_rank:
            pld.system.notify(
                target=gather_done,
                peer=peer,
                offsets=[my_rank, 0],
                value=1,
                op=pld.NotifyOp.Set,
            )

    for src in pl.range(TP_SIZE):
        if src != my_rank:
            pld.system.wait(
                signal=gather_done,
                offsets=[src, 0],
                expected=1,
                cmp=pld.WaitCmp.Ge,
            )

    for t0 in pl.range(0, T, T_TILE):
        for src in pl.range(TP_SIZE):
            src_vocab_base = src * VOCAB_PER_TP
            for ob in pl.range(VOCAB_BLOCKS_PER_TP):
                o0 = ob * VOCAB_CHUNK
                # Store inside each branch: a tile assigned in both arms of an
                # if/else can't be phi-merged ("used outside its defining
                # scope"), so write it within the arm that loaded it.
                if src == my_rank:
                    tile = pl.load(logits_shard, [t0, o0], [T_TILE, VOCAB_CHUNK])
                    pl.store(tile, [t0, src_vocab_base + o0], logits)
                else:
                    tile = pl.load(
                        logits_window,
                        [t0, src_vocab_base + o0],
                        [T_TILE, VOCAB_CHUNK],
                    )
                    pl.store(tile, [t0, src_vocab_base + o0], logits)
    return logits


@pl.jit
def lm_head_tp(
    hidden_states: pl.Tensor[[T, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logits: pl.Out[pl.Tensor[[T, VOCAB], pl.FP32]],
    logits_window: pld.DistributedTensor[[T, VOCAB], pl.FP32],
    gather_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    # scalars trailing — runtime TaskArgs requires all tensor args before any
    # scalar args (#1603-adjacent constraint).
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T, VOCAB], pl.FP32]:
    logits_shard = pl.create_tensor([T, VOCAB_PER_TP], dtype=pl.FP32)
    logits_shard = lm_head(hidden_states, lm_head_weight, logits_shard)
    # gather_step is an InCore dep (a separate IR function): capture its
    # post-call return rather than reusing the pre-call ``logits`` handle.
    logits = gather_step(logits_shard, logits, logits_window, gather_done, my_rank)
    return logits


@pl.jit.host
def host_orch(
    hidden_states: pl.Tensor[[TP_SIZE, T, D], pl.BF16],
    lm_head_weight: pl.Tensor[[TP_SIZE, VOCAB_PER_TP, D], pl.BF16],
    logits: pl.Out[pl.Tensor[[TP_SIZE, T, VOCAB], pl.FP32]],
):
    logits_window_buf = pld.alloc_window_buffer(T * VOCAB * 4)
    gather_done_buf = pld.alloc_window_buffer(TP_SIZE * 4)

    for r in pl.range(pld.world_size()):
        logits_window = pld.window(logits_window_buf, [T, VOCAB], dtype=pl.FP32)
        gather_done = pld.window(gather_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        lm_head_tp(
            hidden_states[r],
            lm_head_weight[r],
            logits[r],
            logits_window,
            gather_done,
            r,
            device=r,
        )


def golden_lm_head(tensors):
    import torch

    hidden = tensors["hidden_states"].float()
    weight = tensors["lm_head_weight"].float()
    shard_logits = []
    for r in range(weight.shape[0]):
        shard_logits.append(torch.matmul(hidden[r], weight[r].t()))
    full_logits = [torch.cat(shard_logits, dim=-1) for _ in range(weight.shape[0])]
    tensors["logits"][:] = torch.stack(full_logits, dim=0)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_hidden_states():
        # Hidden states are replicated across TP ranks (TP splits the vocab,
        # not the tokens).
        x = torch.randn(T, D) * 0.1
        return x.unsqueeze(0).expand(TP_SIZE, -1, -1).contiguous()

    def init_lm_head_weight():
        return (torch.randn(TP_SIZE, VOCAB_PER_TP, D) / D ** 0.5).to(torch.bfloat16)

    return [
        TensorSpec("hidden_states", [TP_SIZE, T, D], torch.bfloat16, init_value=init_hidden_states),
        TensorSpec(
            "lm_head_weight",
            [TP_SIZE, VOCAB_PER_TP, D],
            torch.bfloat16,
            init_value=init_lm_head_weight,
        ),
        TensorSpec("logits", [TP_SIZE, T, VOCAB], torch.float32, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=str, default="0,1",
                        help=f"comma-separated device ids; need at least {TP_SIZE}")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) >= TP_SIZE, (
        f"need at least {TP_SIZE} devices for TP, got {device_ids}"
    )

    result = run_jit(
        fn=host_orch,
        specs=build_tensor_specs(),
        golden_fn=golden_lm_head,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        compile_cfg=dict(
            distributed_config=DistributedConfig(
                device_ids=device_ids[:TP_SIZE],
                num_sub_workers=0,
            ),
        ),
        runtime_cfg=dict(
            platform=args.platform,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
