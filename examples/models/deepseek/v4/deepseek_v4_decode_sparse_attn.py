# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
DeepSeek-V4 compressed flash attention (decode, compress_ratio in {4, 128}).

Corresponds to model.py Attention.forward decode branch lines 533-534
together with the surrounding sparse_attn semantics from kernel.py:355:
    o = sparse_attn(q, kv_cache, attn_sink, topk_idxs, softmax_scale)
    apply_rotary_emb(o[..., -rope_dim:], freqs_cis, inverse=True)

Inputs:
- q                 : query tensor from MLA prolog (RoPE already applied)
- ori_kv / cmp_kv   : paged sliding-window and paged compressed KV pools
- cmp_sparse_indices: per-token absolute indices (window + compressed concat)
                      computed by orchestrator (window topk + indexer topk)
- attn_sink         : per-head sink term added inside softmax
- inverse RoPE on rope dims of o is fused into this kernel's output

Standalone contract:
- `cmp_sparse_indices[t, :]` may contain `-1` pads.
- entries in `[0, WIN)` address the logical sliding-window ring slots.
- entries in `[WIN, WIN + cmp_valid)` address compressed cache slots, where
    `cmp_valid = max(seqused_kv[b] - min(WIN, seqused_kv[b]), 0)`.
- `freqs_cos` / `freqs_sin` store the split-half RoPE tables used by the
    existing standalone examples: first half for the low rope lanes, second
    half for the high rope lanes.
"""


import pypto.language as pl


B           = 16                        # demo 4
S           = 1
T           = B * S
H           = 128                       # demo 64
HEAD_DIM    = 512
ROPE_DIM    = 64
NOPE_DIM    = HEAD_DIM - ROPE_DIM
WIN         = 128
IDX_TOPK    = 1024                      # demo 512
TOPK        = WIN + IDX_TOPK             # 1152
STANDALONE_CMP_VALID = 16
STANDALONE_SPARSE_K  = WIN + STANDALONE_CMP_VALID

BLOCK_SIZE      = 128
ORI_MAX_BLOCKS  = 1                      # WIN==BLOCK_SIZE => 1 block per batch (placeholder)
ORI_BLOCK_NUM   = B * ORI_MAX_BLOCKS
CMP_MAX_BLOCKS  = 64                     # placeholder
CMP_BLOCK_NUM   = B * CMP_MAX_BLOCKS

SOFTMAX_SCALE   = HEAD_DIM ** -0.5
HALF_ROPE       = ROPE_DIM // 2
MATMUL_ROW_PAD  = 16


def build_deepseek_v4_decode_sparse_attn_program():
    @pl.program
    class DeepSeekV4DecodeSparseAttn:
        @pl.function(type=pl.FunctionType.InCore)
        def init_kv_topk_zero(
            self,
            kv_topk_batch: pl.Out[pl.Tensor[[TOPK, HEAD_DIM], pl.BF16]],
        ) -> pl.Tensor[[TOPK, HEAD_DIM], pl.BF16]:
            for kk, (kv_topk_iter,) in pl.range(TOPK, init_values=(kv_topk_batch,)):
                kv_topk_batch_next = pl.store(
                    pl.cast(
                        pl.tile.full([1, HEAD_DIM], dtype=pl.FP32, value=0.0),
                        target_type=pl.BF16,
                    ),
                    [kk, 0],
                    kv_topk_iter,
                )
                (kv_topk_batch_carry,) = pl.yield_(kv_topk_batch_next)

            return kv_topk_batch_carry

        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v4_decode_sparse_attn(
            self,
            q:                  pl.Tensor[[T, H, HEAD_DIM],                               pl.BF16],
            ori_kv:             pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],       pl.BF16],
            ori_block_table:    pl.Tensor[[B, ORI_MAX_BLOCKS],                            pl.INT32],
            cmp_kv:             pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],       pl.BF16],
            cmp_block_table:    pl.Tensor[[B, CMP_MAX_BLOCKS],                            pl.INT32],
            cmp_sparse_indices: pl.Tensor[[T, TOPK],                                      pl.INT32],
            attn_sink:          pl.Tensor[[H],                                            pl.FP32],
            seqused_kv:         pl.Tensor[[B],                                            pl.INT32],
            freqs_cos:          pl.Tensor[[T, ROPE_DIM],                                  pl.BF16],
            freqs_sin:          pl.Tensor[[T, ROPE_DIM],                                  pl.BF16],
            o:                  pl.Out[pl.Tensor[[T, H, HEAD_DIM],                        pl.BF16]],
        ):
            # ===== Stage 0: flatten bridge views and allocate the sparse-attn stage tensor =====
            q_flat = pl.reshape(q, [T * H, HEAD_DIM])
            o_flat = pl.reshape(o, [T * H, HEAD_DIM])
            # Keep cache gathers on flattened 2D views. The backend runtime rejects
            # reshape(view(...)) on 4D sub-slices because those subviews are not
            # considered contiguous in orchestration.
            ori_kv_flat = pl.reshape(ori_kv, [ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
            ori_block_table_flat = pl.reshape(ori_block_table, [B * ORI_MAX_BLOCKS])
            cmp_kv_flat = pl.reshape(cmp_kv, [CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
            cmp_block_table_flat = pl.reshape(cmp_block_table, [B * CMP_MAX_BLOCKS])
            cmp_sparse_indices_flat = pl.reshape(cmp_sparse_indices, [T * TOPK])
            attn_stage = pl.create_tensor([T * H, HEAD_DIM], dtype=pl.BF16)

            for b in pl.range(B):
                # ===== Stage 1: rebuild this batch row's sparse KV list in decode order =====
                seq_used = pl.read(seqused_kv, [b])
                window_valid = pl.min(WIN, seq_used)
                cmp_valid = seq_used - window_valid
                cmp_topk_valid = pl.min(IDX_TOPK, cmp_valid)
                sparse_k = window_valid + cmp_topk_valid
                ori_block_base = b * ORI_MAX_BLOCKS
                cmp_block_base = b * CMP_MAX_BLOCKS
                kv_topk_batch = pl.create_tensor([TOPK, HEAD_DIM], dtype=pl.BF16)
                kv_topk_batch = self.init_kv_topk_zero(kv_topk_batch)

                # Mirror the real decode layout from model.py: valid window entries
                # come first, followed by valid compressed entries, then trailing pads.
                # Stage 1.1: gather the live sliding-window rows into dense TOPK scratch.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_gather_window_kv_topk"):
                    for kk, (kv_topk_iter,) in pl.range(window_valid, init_values=(kv_topk_batch,)):
                        raw_idx_pos = b * TOPK + kk
                        raw_idx = pl.read(cmp_sparse_indices_flat, [raw_idx_pos])
                        ori_slot = raw_idx // BLOCK_SIZE
                        ori_block_pos = ori_block_base + ori_slot
                        ori_blk = pl.cast(pl.read(ori_block_table_flat, [ori_block_pos]), pl.INDEX)
                        ori_intra = raw_idx % BLOCK_SIZE
                        ori_row = ori_blk * BLOCK_SIZE + ori_intra
                        kv_row = pl.load(
                            ori_kv_flat,
                            [ori_row, 0],
                            [1, HEAD_DIM],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        kv_topk_batch_next = pl.store(kv_row, [kk, 0], kv_topk_iter)
                        (kv_topk_batch_carry,) = pl.yield_(kv_topk_batch_next)
                    kv_topk_batch = kv_topk_batch_carry

                # Stage 1.2: append the live compressed-cache rows after the window prefix.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_gather_cmp_kv_topk"):
                    for kk, (kv_topk_iter,) in pl.range(cmp_topk_valid, init_values=(kv_topk_batch,)):
                        raw_idx_pos = b * TOPK + window_valid + kk
                        raw_idx = pl.read(cmp_sparse_indices_flat, [raw_idx_pos])
                        cmp_slot = raw_idx - WIN
                        cmp_block_slot = cmp_slot // BLOCK_SIZE
                        cmp_block_pos = cmp_block_base + cmp_block_slot
                        cmp_blk = pl.cast(pl.read(cmp_block_table_flat, [cmp_block_pos]), pl.INDEX)
                        cmp_intra = cmp_slot % BLOCK_SIZE
                        cmp_row = cmp_blk * BLOCK_SIZE + cmp_intra
                        kv_row = pl.load(
                            cmp_kv_flat,
                            [cmp_row, 0],
                            [1, HEAD_DIM],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        kv_topk_batch_next = pl.store(kv_row, [window_valid + kk, 0], kv_topk_iter)
                        (kv_topk_batch_carry,) = pl.yield_(kv_topk_batch_next)
                    kv_topk_batch = kv_topk_batch_carry

                # ===== Stage 2: consume the gathered sparse KV rows head by head =====
                for h in pl.parallel(H):
                    head_row = b * H + h
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_sparse_attn_init"):
                        # Stage 2.1: duplicate q and the first sparse KV row into the
                        # backend-safe 16-row matmul shape, then seed the online softmax
                        # recurrence from that first sparse KV entry.
                        q_batch = pl.col_expand(
                            pl.full([MATMUL_ROW_PAD, HEAD_DIM], dtype=pl.FP32, value=0.0),
                            pl.cast(q_flat[head_row : head_row + 1, 0 : HEAD_DIM], target_type=pl.FP32),
                        )

                        kv_batch = pl.col_expand(
                            pl.full([MATMUL_ROW_PAD, HEAD_DIM], dtype=pl.FP32, value=0.0),
                            pl.cast(kv_topk_batch[0 : 1, 0 : HEAD_DIM], target_type=pl.FP32),
                        )
                        oi = kv_batch
                        score0_col = pl.row_sum(pl.mul(q_batch, kv_batch))
                        score0_mat = pl.row_expand(
                            pl.full([MATMUL_ROW_PAD, MATMUL_ROW_PAD], dtype=pl.FP32, value=0.0),
                            score0_col,
                        )
                        score0 = score0_mat[0 : 1, 0 : MATMUL_ROW_PAD]
                        mi = pl.mul(score0, SOFTMAX_SCALE)
                        li = pl.exp(pl.sub(mi, mi))

                    # Stage 2.2: mirror the v3.2 scope4 recurrence shape and update
                    # mi/li/oi directly across per-iteration CORE_GROUP regions.
                    for kk in pl.range(1, sparse_k):
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_sparse_attn_accum"):
                            kv_batch = pl.col_expand(
                                pl.full([MATMUL_ROW_PAD, HEAD_DIM], dtype=pl.FP32, value=0.0),
                                pl.cast(kv_topk_batch[kk : kk + 1, 0 : HEAD_DIM], target_type=pl.FP32),
                            )
                            cur_score_col = pl.row_sum(pl.mul(q_batch, kv_batch))
                            cur_score_mat = pl.row_expand(
                                pl.full([MATMUL_ROW_PAD, MATMUL_ROW_PAD], dtype=pl.FP32, value=0.0),
                                cur_score_col,
                            )
                            cur_score = cur_score_mat[0 : 1, 0 : MATMUL_ROW_PAD]
                            cur_mi = pl.mul(cur_score, SOFTMAX_SCALE)
                            mi_new = pl.maximum(mi, cur_mi)
                            alpha = pl.exp(pl.sub(mi, mi_new))
                            beta = pl.exp(pl.sub(cur_mi, mi_new))
                            li = pl.add(pl.mul(alpha, li), beta)
                            alpha_mat = pl.col_expand(
                                pl.full([MATMUL_ROW_PAD, MATMUL_ROW_PAD], dtype=pl.FP32, value=0.0),
                                alpha,
                            )
                            beta_mat = pl.col_expand(
                                pl.full([MATMUL_ROW_PAD, MATMUL_ROW_PAD], dtype=pl.FP32, value=0.0),
                                beta,
                            )
                            oi = pl.add(
                                pl.row_expand_mul(oi, alpha_mat[0 : MATMUL_ROW_PAD, 0 : 1]),
                                pl.row_expand_mul(kv_batch, beta_mat[0 : MATMUL_ROW_PAD, 0 : 1]),
                            )
                            mi = mi_new

                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_sparse_attn_norm"):
                        # Stage 2.3: add the sink-only denominator term, normalize the
                        # FP32 accumulator, and stash the sparse-attn output as BF16.
                        # attn_sink is an extra softmax logit only. It changes the
                        # denominator but does not contribute any value vector to oi.
                        sink_tile = pl.add(pl.sub(mi, mi), pl.read(attn_sink, [h]))
                        denom = pl.add(li, pl.exp(pl.sub(sink_tile, mi)))
                        denom_mat = pl.col_expand(
                            pl.full([MATMUL_ROW_PAD, MATMUL_ROW_PAD], dtype=pl.FP32, value=0.0),
                            denom,
                        )
                        oi_out = pl.row_expand_div(oi, denom_mat[0 : MATMUL_ROW_PAD, 0 : 1])
                        attn_stage_row = pl.cast(
                            oi_out[0 : 1, 0 : HEAD_DIM],
                            target_type=pl.BF16,
                        )
                        attn_stage = pl.assemble(
                            attn_stage,
                            attn_stage_row,
                            [head_row, 0],
                        )

                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_inverse_rope"):
                        # ===== Stage 3: inverse RoPE consumes the BF16-staged sparse
                        # attention output and writes the final BF16 decode result. =====
                        o_nope = attn_stage[head_row : head_row + 1, 0 : NOPE_DIM]
                        cos_lo = pl.cast(freqs_cos[b : b + 1, 0 : HALF_ROPE], target_type=pl.FP32)
                        cos_hi = pl.cast(freqs_cos[b : b + 1, HALF_ROPE : ROPE_DIM], target_type=pl.FP32)
                        sin_lo = pl.cast(freqs_sin[b : b + 1, 0 : HALF_ROPE], target_type=pl.FP32)
                        sin_hi = pl.cast(freqs_sin[b : b + 1, HALF_ROPE : ROPE_DIM], target_type=pl.FP32)
                        x_lo = pl.cast(attn_stage[head_row : head_row + 1, NOPE_DIM : NOPE_DIM + HALF_ROPE], target_type=pl.FP32)
                        x_hi = pl.cast(attn_stage[head_row : head_row + 1, NOPE_DIM + HALF_ROPE : HEAD_DIM], target_type=pl.FP32)
                        y_lo = pl.add(
                            pl.col_expand_mul(x_lo, cos_lo),
                            pl.col_expand_mul(x_hi, sin_lo),
                        )
                        y_hi = pl.sub(
                            pl.col_expand_mul(x_hi, cos_hi),
                            pl.col_expand_mul(x_lo, sin_hi),
                        )
                        o_flat = pl.assemble(o_flat, o_nope, [head_row, 0])
                        o_flat = pl.assemble(o_flat, pl.cast(y_lo, target_type=pl.BF16), [head_row, NOPE_DIM])
                        o_flat = pl.assemble(o_flat, pl.cast(y_hi, target_type=pl.BF16), [head_row, NOPE_DIM + HALF_ROPE])

            return o

    return DeepSeekV4DecodeSparseAttn


def golden_deepseek_v4_decode_sparse_attn(tensors):
    """Torch reference: mirror gather, sparse-attn staging, and inverse RoPE flow."""
    import torch

    q                  = tensors["q"].float()                              # [T, H, HEAD_DIM]
    ori_kv             = tensors["ori_kv"].float()                         # [bn, BS, 1, D]
    ori_block_table    = tensors["ori_block_table"]
    cmp_kv             = tensors["cmp_kv"].float()
    cmp_block_table    = tensors["cmp_block_table"]
    cmp_sparse_indices = tensors["cmp_sparse_indices"]                     # [T, TOPK]
    attn_sink          = tensors["attn_sink"].float()                      # [H]
    seqused_kv         = tensors["seqused_kv"]
    cos                = tensors["freqs_cos"].float()
    sin                = tensors["freqs_sin"].float()

    # Logical layout: index in [0, WIN) -> window slot in ori_kv (per-batch ring buffer)
    #                index in [WIN, WIN + cmp_len) -> compressed slot id (cmp_sparse_indices already
    #                offsets by WIN). -1 indicates a padded / invalid slot.

    out = torch.zeros(T, H, HEAD_DIM)

    for b in range(B):
        seq_used = int(seqused_kv[b].item())
        window_valid = min(WIN, seq_used)
        cmp_valid = max(seq_used - window_valid, 0)
        idxs = cmp_sparse_indices[b]                                       # [TOPK]
        gathered = []
        for raw in idxs.tolist():
            if raw < 0:
                continue
            if raw < WIN:
                if raw >= window_valid:
                    continue
                blk_id = int(ori_block_table[b, raw // BLOCK_SIZE].item())
                intra  = raw % BLOCK_SIZE
                gathered.append(ori_kv[blk_id, intra, 0])                   # [HEAD_DIM]
            else:
                cmp_slot = raw - WIN
                if cmp_slot >= cmp_valid:
                    continue
                blk_id = int(cmp_block_table[b, cmp_slot // BLOCK_SIZE].item())
                intra  = cmp_slot % BLOCK_SIZE
                gathered.append(cmp_kv[blk_id, intra, 0])
        if not gathered:
            continue
        kv_b = torch.stack(gathered, dim=0)                                # [N, HEAD_DIM]

        q_b = q[b]                                                          # [H, HEAD_DIM]
        scores = (q_b @ kv_b.T) * SOFTMAX_SCALE                            # [H, N]
        mi = scores.max(dim=-1, keepdim=True).values                       # [H, 1]
        exp_scores = torch.exp(scores - mi)                                # [H, N]
        oi_num = exp_scores @ kv_b                                         # [H, HEAD_DIM]
        li = exp_scores.sum(dim=-1, keepdim=True)                          # [H, 1]
        denom = li + torch.exp(attn_sink.unsqueeze(-1) - mi)               # [H, 1]
        o_b = oi_num / denom                                               # [H, HEAD_DIM]
        out[b] = o_b

    rope_lo = out[..., NOPE_DIM : NOPE_DIM + HALF_ROPE]
    rope_hi = out[..., NOPE_DIM + HALF_ROPE :]
    cos_lo = cos[:, :HALF_ROPE].unsqueeze(1)
    cos_hi = cos[:, HALF_ROPE:].unsqueeze(1)
    sin_lo = sin[:, :HALF_ROPE].unsqueeze(1)
    sin_hi = sin[:, HALF_ROPE:].unsqueeze(1)
    inv_lo = rope_lo * cos_lo + rope_hi * sin_lo
    inv_hi = rope_hi * cos_hi - rope_lo * sin_hi
    out = torch.cat([out[..., :NOPE_DIM], inv_lo, inv_hi], dim=-1)

    tensors["o"][:] = out.to(torch.bfloat16)


def build_tensor_specs():
    import torch  # type: ignore[import]
    from golden import TensorSpec

    def seeded_uniform(shape, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        return torch.rand(*shape, generator=generator) - 0.5

    def init_q():
        return seeded_uniform((T, H, HEAD_DIM), 1)
    def init_ori_kv():
        return seeded_uniform((ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM), 2)
    def init_cmp_kv():
        return seeded_uniform((CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM), 3)
    def init_attn_sink():
        return torch.zeros(H)

    def init_ori_block_table():
        tbl = torch.full((B, ORI_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(ORI_MAX_BLOCKS):
                tbl[b, j] = b * ORI_MAX_BLOCKS + j
        return tbl

    def init_cmp_block_table():
        tbl = torch.full((B, CMP_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(CMP_MAX_BLOCKS):
                tbl[b, j] = b * CMP_MAX_BLOCKS + j
        return tbl

    def init_cmp_sparse_indices():
        # First WIN entries: window indices 0..WIN-1; keep only a short valid
        # compressed tail so the standalone harness exercises pad handling too.
        win_part = torch.arange(WIN, dtype=torch.int32).unsqueeze(0).expand(T, -1)
        cmp_part = torch.full((T, IDX_TOPK), -1, dtype=torch.int32)
        cmp_valid = STANDALONE_CMP_VALID
        cmp_part[:, :cmp_valid] = (torch.arange(cmp_valid, dtype=torch.int32) + WIN).unsqueeze(0).expand(T, -1)
        return torch.cat([win_part, cmp_part], dim=-1).contiguous()

    def init_seqused_kv():
        return torch.tensor([STANDALONE_SPARSE_K] * B, dtype=torch.int32)
    def init_cos():
        # The standalone contract uses split-half RoPE tables: low and high rope
        # lanes consume the same per-position phase values from separate halves.
        angles = torch.arange(T * HALF_ROPE).reshape(T, HALF_ROPE) * 1e-3
        cos_half = torch.cos(angles)
        return torch.cat([cos_half, cos_half], dim=-1)
    def init_sin():
        angles = torch.arange(T * HALF_ROPE).reshape(T, HALF_ROPE) * 1e-3
        sin_half = torch.sin(angles)
        return torch.cat([sin_half, sin_half], dim=-1)

    return [
        TensorSpec("q",                  [T, H, HEAD_DIM],                              torch.bfloat16, init_value=init_q),
        TensorSpec("ori_kv",             [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],      torch.bfloat16, init_value=init_ori_kv),
        TensorSpec("ori_block_table",    [B, ORI_MAX_BLOCKS],                           torch.int32,    init_value=init_ori_block_table),
        TensorSpec("cmp_kv",             [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],      torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table",    [B, CMP_MAX_BLOCKS],                           torch.int32,    init_value=init_cmp_block_table),
        TensorSpec("cmp_sparse_indices", [T, TOPK],                                     torch.int32,    init_value=init_cmp_sparse_indices),
        TensorSpec("attn_sink",          [H],                                           torch.float32,  init_value=init_attn_sink),
        TensorSpec("seqused_kv",         [B],                                           torch.int32,    init_value=init_seqused_kv),
        TensorSpec("freqs_cos",          [T, ROPE_DIM],                                 torch.bfloat16, init_value=init_cos),
        TensorSpec("freqs_sin",          [T, ROPE_DIM],                                 torch.bfloat16, init_value=init_sin),
        TensorSpec("o",                  [T, H, HEAD_DIM],                              torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import RunConfig, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        program=build_deepseek_v4_decode_sparse_attn_program(),
        specs=build_tensor_specs(),
        golden_fn=golden_deepseek_v4_decode_sparse_attn,
        config=RunConfig(
            rtol=1e-3,
            atol=1e-3,
            compile=dict(dump_passes=True),
            runtime=dict(
                platform=args.platform,
                device_id=args.device,
                runtime_profiling=args.runtime_profiling,
            ),
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
