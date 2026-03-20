# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
GLM-4.5 Attention Module (PTO 3.0)

This module implements the Attention mechanism for GLM-4.5 model, which uses
a paged memory management approach similar to operating systems to efficiently
handle variable-length sequences and dynamic batch sizes in attention computation.

Migrated from PTO 2.0 to PTO 3.0.
"""

from __future__ import annotations

import os
import math

import pypto.language as pl


B = 8
S1 = 1
S2 = 16384
Q_D = 128
NQ = 12
NKV = 1
BLOCK_SIZE = 128

G_TILE = 12
S2_TILE = 512
M_TILE = 128
CUBE_TILE = 128


def build_glm_attention_program(
    batch: int = B,
    s1: int = S1,
    s2: int = S2,
    q_d: int = Q_D,
    nq: int = NQ,
    nkv: int = NKV,
    block_size: int = BLOCK_SIZE,
    g_tile: int = G_TILE,
    s2_tile: int = S2_TILE,
    m_tile: int = M_TILE,
    cube_tile: int = CUBE_TILE,
):
    b_cfg = batch
    s1_cfg = s1
    s2_cfg = s2
    q_d_cfg = q_d
    nq_cfg = nq
    nkv_cfg = nkv
    block_size_cfg = block_size
    g_tile_cfg = g_tile
    s2_tile_cfg = s2_tile
    m_tile_cfg = m_tile
    cube_tile_cfg = cube_tile
    
    kv_num_blocks = b_cfg * ((s2_cfg + block_size_cfg - 1) // block_size_cfg)
    max_num_blocks_per_query = (s2_cfg + block_size_cfg - 1) // block_size_cfg
    softmax_scale = q_d_cfg ** -0.5
    group = nq_cfg // nkv_cfg
    g_loop = nq_cfg // nkv_cfg // g_tile_cfg
    block_num = s2_tile_cfg // block_size_cfg
    
    @pl.program
    class GLMAttention:
        @pl.function(type=pl.FunctionType.Opaque)
        def glm_flash_attention(
            self,
            q: pl.Tensor[[b_cfg * s1_cfg, nq_cfg, q_d_cfg], pl.BF16],
            k: pl.Tensor[[kv_num_blocks, block_size_cfg, nkv_cfg, q_d_cfg], pl.BF16],
            v: pl.Tensor[[kv_num_blocks, block_size_cfg, nkv_cfg, q_d_cfg], pl.BF16],
            block_table: pl.Tensor[[b_cfg, max_num_blocks_per_query], pl.INT32],
            kv_act_seqs: pl.Tensor[[b_cfg], pl.INT32],
            attn_out: pl.Tensor[[b_cfg * s1_cfg, nq_cfg, q_d_cfg], pl.BF16],
        ) -> None:
            with pl.auto_incore():
                for b_idx in pl.range(b_cfg):
                    for s1_idx in pl.range(s1_cfg):
                        cur_seq = pl.tensor.read(kv_act_seqs, [b_idx]) - (s1_cfg - 1 - s1_idx)
                        s2_loop = (cur_seq + s2_tile_cfg - 1) // s2_tile_cfg
                        
                        for n2_idx in pl.range(nkv_cfg):
                            for g_idx in pl.range(g_loop):
                                oi_update = pl.create_tensor([g_tile_cfg, q_d_cfg], dtype=pl.FP32)
                                oi_update = pl.mul(oi_update, 0.0)
                                sum_update = pl.create_tensor([g_tile_cfg, 1], dtype=pl.FP32)
                                sum_update = pl.mul(sum_update, 0.0)
                                max_update = pl.create_tensor([g_tile_cfg, 1], dtype=pl.FP32)
                                max_update = pl.mul(max_update, 0.0)
                                
                                for s2_idx in pl.range(s2_loop):
                                    idx = s2_idx * block_num
                                    bs_ofs = b_idx * s1_cfg + s1_idx
                                    n1g_ofs = n2_idx * group + g_idx * g_tile_cfg
                                    actual_s2_tile = pl.min(cur_seq - s2_idx * s2_tile_cfg, s2_tile_cfg)
                                    
                                    q_offset = bs_ofs * nq_cfg + n1g_ofs
                                    qi = pl.slice(q, [g_tile_cfg, q_d_cfg], [q_offset, 0])
                                    qi_fp32 = pl.cast(qi, target_type=pl.FP32)
                                    
                                    kj_assemble = pl.create_tensor([s2_tile_cfg, q_d_cfg], dtype=pl.FP32)
                                    kj_assemble = pl.mul(kj_assemble, 0.0)
                                    
                                    for i in range(block_num):
                                        block_idx = pl.tensor.read(block_table, [b_idx, idx + i])
                                        block_idx_valid = pl.max(block_idx, 0)
                                        k_offset = block_idx_valid * block_size_cfg
                                        kj_block = pl.slice(k, [block_size_cfg, 1, q_d_cfg], [k_offset, n2_idx, 0])
                                        kj_block_fp32 = pl.cast(kj_block, target_type=pl.FP32)
                                        kj_block_2d = pl.reshape(kj_block_fp32, [block_size_cfg, q_d_cfg])
                                        kj_assemble = pl.assemble(kj_assemble, kj_block_2d, [i * block_size_cfg, 0])
                                    
                                    kj_valid = pl.slice(kj_assemble, [actual_s2_tile, q_d_cfg], [0, 0])
                                    
                                    sij = pl.matmul(qi_fp32, kj_valid, transpose_b=True)
                                    sij_valid = pl.slice(sij, [g_tile_cfg, actual_s2_tile], [0, 0])
                                    
                                    if pl.is_loop_begin(s2_idx):
                                        sij_scale = pl.muls(sij_valid, softmax_scale)
                                        tilda_mij = pl.amax(sij_scale, axis=-1, keepdim=True)
                                        tsub = pl.sub(sij_scale, tilda_mij)
                                        tilda_pij = pl.exp(tsub)
                                        sum_local = pl.sum(tilda_pij, axis=-1, keepdim=True)
                                        
                                        sum_update = sum_local
                                        max_update = tilda_mij
                                        
                                        vj_assemble = pl.create_tensor([s2_tile_cfg, q_d_cfg], dtype=pl.FP32)
                                        vj_assemble = pl.mul(vj_assemble, 0.0)
                                        
                                        for i in range(block_num):
                                            block_idx = pl.tensor.read(block_table, [b_idx, idx + i])
                                            block_idx_valid = pl.max(block_idx, 0)
                                            v_offset = block_idx_valid * block_size_cfg
                                            vj_block = pl.slice(v, [block_size_cfg, 1, q_d_cfg], [v_offset, n2_idx, 0])
                                            vj_block_fp32 = pl.cast(vj_block, target_type=pl.FP32)
                                            vj_block_2d = pl.reshape(vj_block_fp32, [block_size_cfg, q_d_cfg])
                                            vj_assemble = pl.assemble(vj_assemble, vj_block_2d, [i * block_size_cfg, 0])
                                        
                                        vj_valid = pl.slice(vj_assemble, [actual_s2_tile, q_d_cfg], [0, 0])
                                        
                                        oi_tmp = pl.matmul(tilda_pij, vj_valid)
                                        oi_update = oi_tmp
                                    else:
                                        sij_scale = pl.muls(sij_valid, softmax_scale)
                                        tilda_mij = pl.amax(sij_scale, axis=-1, keepdim=True)
                                        max_new = pl.maximum(max_update, tilda_mij)
                                        tsub = pl.sub(sij_scale, max_new)
                                        tilda_pij = pl.exp(tsub)
                                        sum_local = pl.sum(tilda_pij, axis=-1, keepdim=True)
                                        
                                        tsub2 = pl.sub(max_update, max_new)
                                        update_mul = pl.exp(tsub2)
                                        sum_update = pl.add(pl.mul(sum_update, update_mul), sum_local)
                                        max_update = max_new
                                        
                                        vj_assemble = pl.create_tensor([s2_tile_cfg, q_d_cfg], dtype=pl.FP32)
                                        vj_assemble = pl.mul(vj_assemble, 0.0)
                                        
                                        for i in range(block_num):
                                            block_idx = pl.tensor.read(block_table, [b_idx, idx + i])
                                            block_idx_valid = pl.max(block_idx, 0)
                                            v_offset = block_idx_valid * block_size_cfg
                                            vj_block = pl.slice(v, [block_size_cfg, 1, q_d_cfg], [v_offset, n2_idx, 0])
                                            vj_block_fp32 = pl.cast(vj_block, target_type=pl.FP32)
                                            vj_block_2d = pl.reshape(vj_block_fp32, [block_size_cfg, q_d_cfg])
                                            vj_assemble = pl.assemble(vj_assemble, vj_block_2d, [i * block_size_cfg, 0])
                                        
                                        vj_valid = pl.slice(vj_assemble, [actual_s2_tile, q_d_cfg], [0, 0])
                                        
                                        oi_tmp = pl.matmul(tilda_pij, vj_valid)
                                        oi_update = pl.add(pl.mul(oi_update, update_mul), oi_tmp)
                                    
                                    if pl.is_loop_end(s2_idx):
                                        oi_final = pl.div(oi_update, sum_update)
                                        oi_final_bf16 = pl.cast(oi_final, target_type=pl.BF16)
                                        oi_final_3d = pl.reshape(oi_final_bf16, [1, g_tile_cfg, q_d_cfg])
                                        attn_out = pl.assemble(attn_out, oi_final_3d, [bs_ofs, n1g_ofs, 0])
    
    return GLMAttention