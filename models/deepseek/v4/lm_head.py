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


T = DECODE_TOKENS  # 128 decode tokens.
D = M.hidden_size  # 4096 hidden size.
VOCAB = M.vocab_size  # 129280 vocabulary size.

LM_HEAD_K_CHUNK = 128  # K tile width; D / 128 = 32 matmul accumulation blocks.
VOCAB_CHUNK = 160  # Vocab tile width; with TP=8, VOCAB_PER_TP / 160 = 101 blocks.
T_TILE = 16  # Token tile height; T / 16 = 8 token blocks.

assert D % LM_HEAD_K_CHUNK == 0
assert T % T_TILE == 0
assert VOCAB % VOCAB_CHUNK == 0

K_BLOCKS = D // LM_HEAD_K_CHUNK  # 32.
VOCAB_BLOCKS_FULL = VOCAB // VOCAB_CHUNK  # 808 full-vocab blocks.
VOCAB_PER_TP = VOCAB // LM_HEAD_TP_SIZE  # 16160 when LM_HEAD_TP_SIZE=8.
VOCAB_BLOCKS_PER_TP = VOCAB_PER_TP // VOCAB_CHUNK  # 101 blocks per TP shard.

assert VOCAB % LM_HEAD_TP_SIZE == 0
assert VOCAB_PER_TP % VOCAB_CHUNK == 0


@pl.jit.inline
def lm_head(
    hidden_states: pl.Tensor[[T, D], pl.BF16],
    lm_head_weight: pl.Tensor[["VOCAB_PER_TP", D], pl.BF16],
    logits_shard: pl.Out[pl.Tensor[[T, "VOCAB_PER_TP"], pl.FP32]],
) -> pl.Tensor[[T, "VOCAB_PER_TP"], pl.FP32]:
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


def _build_tp_lm_head_program(tp_size: int):
    """Build a TP-sized distributed LM head program.

    ``tp_size`` is deliberately a Python build-time value because tensor shapes
    and loop trip counts are static in the pypto frontend.
    """
    assert tp_size >= 1
    assert VOCAB % tp_size == 0

    VOCAB_PER_TP = VOCAB // tp_size
    assert VOCAB_PER_TP % VOCAB_CHUNK == 0
    VOCAB_BLOCKS_PER_TP = VOCAB_PER_TP // VOCAB_CHUNK
    lm_head_tp_inline = pl.inline(lm_head._func)

    @pl.program
    class TpLmHead:
        @pl.function(type=pl.FunctionType.InCore)
        def gather_step(
            self,
            logits_shard: pl.Tensor[[T, VOCAB_PER_TP], pl.FP32],
            logits: pl.Out[pl.Tensor[[T, VOCAB], pl.FP32]],
            logits_window: pld.DistributedTensor[[T, VOCAB], pl.FP32],
            gather_done: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[T, VOCAB], pl.FP32]:
            # Publish this rank's contiguous vocab shard into every peer's full
            # logits window. The local rank writes its own shard directly to
            # the host-backed output below; routing it through the comm window
            # would add a large self-remote path that is unnecessary here.
            vocab_base = my_rank * VOCAB_PER_TP
            for peer in pl.range(tp_size):
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

            for peer in pl.range(tp_size):
                if peer != my_rank:
                    pld.system.notify(
                        target=gather_done,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.Set,
                    )

            for src in pl.range(tp_size):
                if src != my_rank:
                    pld.system.wait(
                        signal=gather_done,
                        offsets=[src, 0],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )

            for t0 in pl.range(0, T, T_TILE):
                for src in pl.range(tp_size):
                    src_vocab_base = src * VOCAB_PER_TP
                    for ob in pl.range(VOCAB_BLOCKS_PER_TP):
                        o0 = ob * VOCAB_CHUNK
                        if src == my_rank:
                            tile = pl.load(logits_shard, [t0, o0], [T_TILE, VOCAB_CHUNK])
                        else:
                            tile = pl.load(
                                logits_window,
                                [t0, src_vocab_base + o0],
                                [T_TILE, VOCAB_CHUNK],
                            )
                        pl.store(tile, [t0, src_vocab_base + o0], logits)
            return logits

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            hidden_states: pl.Tensor[[T, D], pl.BF16],
            lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
            logits: pl.Out[pl.Tensor[[T, VOCAB], pl.FP32]],
            logits_window: pld.DistributedTensor[[T, VOCAB], pl.FP32],
            gather_done: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[T, VOCAB], pl.FP32]:
            logits_shard = pl.create_tensor([T, VOCAB_PER_TP], dtype=pl.FP32)
            logits_shard = lm_head_tp_inline(hidden_states, lm_head_weight, logits_shard)
            return self.gather_step(logits_shard, logits, logits_window, gather_done, my_rank)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            hidden_states: pl.Tensor[[T, D], pl.BF16],
            lm_head_weight: pl.Tensor[[tp_size, VOCAB_PER_TP, D], pl.BF16],
            logits: pl.Out[pl.Tensor[[tp_size, T, VOCAB], pl.FP32]],
        ):
            logits_window_buf = pld.alloc_window_buffer(T * VOCAB * 4)
            gather_done_buf = pld.alloc_window_buffer(tp_size * 4)

            for r in pl.range(pld.world_size()):
                logits_window = pld.window(logits_window_buf, [T, VOCAB], dtype=pl.FP32)
                gather_done = pld.window(gather_done_buf, [tp_size, 1], dtype=pl.INT32)
                self.chip_orch(
                    hidden_states,
                    lm_head_weight[r],
                    logits[r],
                    logits_window,
                    gather_done,
                    r,
                    device=r,
                )

    return TpLmHead


def golden_lm_head(tensors):
    import torch

    hidden = tensors["hidden_states"].float()
    weight = tensors["lm_head_weight"].float()
    shard_logits = []
    for r in range(weight.shape[0]):
        shard_logits.append(torch.matmul(hidden, weight[r].t()))
    full_logits = torch.cat(shard_logits, dim=-1)
    tensors["logits"][:] = full_logits.unsqueeze(0).expand(weight.shape[0], -1, -1)


def build_tensor_specs(tp_size: int):
    import torch
    from golden import TensorSpec

    vocab_per_tp = VOCAB // tp_size

    def init_hidden_states():
        return torch.randn(T, D) * 0.1

    def init_lm_head_weight():
        return (torch.randn(tp_size, vocab_per_tp, D) / D ** 0.5).to(torch.bfloat16)

    return [
        TensorSpec("hidden_states", [T, D], torch.bfloat16, init_value=init_hidden_states),
        TensorSpec(
            "lm_head_weight",
            [tp_size, vocab_per_tp, D],
            torch.bfloat16,
            init_value=init_lm_head_weight,
        ),
        TensorSpec("logits", [tp_size, T, VOCAB], torch.float32, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=str, default="0,1",
                        help="comma-separated device ids; count must match --tp-size")
    parser.add_argument("--tp-size", type=int, default=2,
                        help=f"TP world size to build; deployment default is {LM_HEAD_TP_SIZE}")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert args.tp_size >= 1
    assert args.tp_size <= LM_HEAD_TP_SIZE
    assert LM_HEAD_TP_SIZE % args.tp_size == 0
    assert VOCAB % args.tp_size == 0
    assert len(device_ids) >= args.tp_size, (
        f"need at least {args.tp_size} devices for TP, got {device_ids}"
    )

    program = _build_tp_lm_head_program(args.tp_size)
    result = run(
        program=program,
        specs=build_tensor_specs(args.tp_size),
        golden_fn=golden_lm_head,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        compile_cfg=dict(
            distributed_config=DistributedConfig(
                device_ids=device_ids[:args.tp_size],
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
