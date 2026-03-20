# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Gated Delta Rule Implementation Module (PTO 3.0)

Migrated from PTO 2.0 to PTO 3.0.
"""

from __future__ import annotations

import pypto.language as pl


L = 128
D = 128


def l2norm(query, key, eps=1e-6):
    sq_q = pl.mul(query, query)
    sum_q = pl.row_sum(sq_q)
    sum_q_eps = pl.adds(sum_q, eps)
    inv_q = pl.rsqrt(sum_q_eps)
    query_norm = pl.col_expand_mul(query, inv_q)
    
    sq_k = pl.mul(key, key)
    sum_k = pl.row_sum(sq_k)
    sum_k_eps = pl.adds(sum_k, eps)
    inv_k = pl.rsqrt(sum_k_eps)
    key_norm = pl.col_expand_mul(key, inv_k)
    
    return query_norm, key_norm


def pre_attn(gate_view, key_view_2d, beta_view, tril, mask, l):
    gate_cum = pl.matmul(tril, gate_view)
    
    gate_cum_t = pl.transpose(gate_cum, 0, 1)
    g_sub = pl.sub(gate_cum, gate_cum_t)
    decay_mask_raw = pl.exp(g_sub)
    decay_mask = pl.mul(pl.mul(decay_mask_raw, tril), tril)
    
    key_beta = pl.mul(key_view_2d, beta_view)
    
    key_t = pl.transpose(key_view_2d, 0, 1)
    kkt = pl.matmul(key_beta, key_t)
    a = pl.mul(pl.mul(kkt, decay_mask), mask)
    
    return gate_cum, decay_mask, a, key_beta


def cal_value_and_key_cumdecay(attn, value_view, beta_view, key_beta, gate_cum):
    value_beta_view = pl.mul(value_view, beta_view)
    value_out = pl.matmul(attn, value_beta_view)
    
    g_exp = pl.exp(gate_cum)
    weighted_k_beta_view = pl.mul(key_beta, g_exp)
    key_cum_out = pl.matmul(attn, weighted_k_beta_view)
    
    return value_out, key_cum_out


def inverse_pto_min_length(attn, eye, min_length):
    attn_t = pl.transpose(attn, 0, 1)
    
    attn_inv = pl.create_tensor([min_length, min_length], dtype=pl.FP32)
    attn_inv = pl.mul(attn_inv, 0.0)
    
    row_0 = pl.slice(attn, [1, min_length], [0, 0])
    attn_inv = pl.assemble(attn_inv, row_0, [0, 0])
    row_1 = pl.slice(attn, [1, min_length], [1, 0])
    attn_inv = pl.assemble(attn_inv, row_1, [1, 0])
    
    for i in range(2, min_length):
        attn_inv_prev = pl.slice(attn_inv, [i, min_length], [0, 0])
        
        row = pl.slice(attn, [1, min_length], [i, 0])
        
        col = pl.slice(attn_t, [i, 1], [0, i])
        prod = pl.matmul(attn_inv_prev, col)
        prod_2d = pl.reshape(prod, [i, 1])
        attn_update = pl.add(row, pl.transpose(prod_2d, 0, 1))
        
        attn_inv = pl.assemble(attn_inv, attn_update, [i, 0])
    
    result = pl.add(attn_inv, eye)
    return result


def inverse_pto(attn, eye, l):
    min_length = l // 8
    
    attn_8_8_blocks = []
    for i in range(8):
        block = pl.slice(attn, [min_length, min_length], [min_length * i, min_length * i])
        attn_8_8_blocks.append(block)
    
    attn_inv_8_blocks = []
    for i in range(8):
        block = attn_8_8_blocks[i]
        eye_block = pl.slice(eye, [min_length, min_length], [0, min_length * i])
        inv_block = inverse_pto_min_length(block, eye_block, min_length)
        attn_inv_8_blocks.append(inv_block)
    
    m_len = min_length
    
    zeros_16 = pl.create_tensor([16, 16], dtype=pl.FP32)
    zeros_16 = pl.mul(zeros_16, 0.0)
    
    attn_inv_4_blocks = []
    for i in range(4):
        inv_1 = attn_inv_8_blocks[i * 2]
        inv_2 = attn_inv_8_blocks[i * 2 + 1]
        
        a_21 = pl.slice(attn, [m_len, m_len], [m_len * (i * 2 + 1), m_len * (i * 2)])
        temp1 = pl.matmul(inv_2, a_21)
        inv_21 = pl.matmul(temp1, inv_1)
        
        inv_block = pl.create_tensor([m_len * 2, m_len * 2], dtype=pl.FP32)
        inv_block = pl.assemble(inv_block, inv_1, [0, 0])
        inv_block = pl.assemble(inv_block, zeros_16, [0, m_len])
        inv_block = pl.assemble(inv_block, inv_21, [m_len, 0])
        inv_block = pl.assemble(inv_block, inv_2, [m_len, m_len])
        attn_inv_4_blocks.append(inv_block)
    
    zeros_32 = pl.create_tensor([32, 32], dtype=pl.FP32)
    zeros_32 = pl.mul(zeros_32, 0.0)
    
    m_len = min_length * 2
    attn_inv_2_blocks = []
    for i in range(2):
        inv_1 = attn_inv_4_blocks[i * 2]
        inv_2 = attn_inv_4_blocks[i * 2 + 1]
        
        a_21 = pl.slice(attn, [m_len, m_len], [m_len * (i * 2 + 1), m_len * (i * 2)])
        temp1 = pl.matmul(inv_2, a_21)
        inv_21 = pl.matmul(temp1, inv_1)
        
        inv_block = pl.create_tensor([m_len * 2, m_len * 2], dtype=pl.FP32)
        inv_block = pl.assemble(inv_block, inv_1, [0, 0])
        inv_block = pl.assemble(inv_block, zeros_32, [0, m_len])
        inv_block = pl.assemble(inv_block, inv_21, [m_len, 0])
        inv_block = pl.assemble(inv_block, inv_2, [m_len, m_len])
        attn_inv_2_blocks.append(inv_block)
    
    zeros_64 = pl.create_tensor([64, 64], dtype=pl.FP32)
    zeros_64 = pl.mul(zeros_64, 0.0)
    
    m_len = min_length * 4
    inv_1 = attn_inv_2_blocks[0]
    inv_2 = attn_inv_2_blocks[1]
    
    a_21 = pl.slice(attn, [m_len, m_len], [m_len, 0])
    temp1 = pl.matmul(inv_2, a_21)
    inv_21 = pl.matmul(temp1, inv_1)
    
    attn_inv = pl.create_tensor([l, l], dtype=pl.FP32)
    attn_inv = pl.assemble(attn_inv, inv_1, [0, 0])
    attn_inv = pl.assemble(attn_inv, zeros_64, [0, m_len])
    attn_inv = pl.assemble(attn_inv, inv_21, [m_len, 0])
    attn_inv = pl.assemble(attn_inv, inv_2, [m_len, m_len])
    
    return attn_inv


def recurrent_state_attn_all(query, key, value, k_cumdecay, gate, state, decay_mask, tril, l):
    dv = value.shape[1]
    d = key.shape[1]
    
    gate_exp = pl.exp(gate)
    
    last_gate = pl.slice(gate, [1, 1], [l - 1, 0])
    gate_diff = pl.sub(last_gate, gate)
    kgexp = pl.mul(key, gate_diff)
    
    qgexp = pl.mul(query, gate_exp)
    
    state_t = pl.transpose(state, 0, 1)
    v_prime = pl.matmul(k_cumdecay, state_t)
    
    attn_inter = pl.matmul(qgexp, state_t)
    
    kgexp_t = pl.transpose(kgexp, 0, 1)
    temp_matmul_vprime = pl.matmul(v_prime, kgexp_t)
    temp_matmul_value = pl.matmul(value, kgexp_t)
    
    key_t = pl.transpose(key, 0, 1)
    attn = pl.matmul(query, key_t)
    
    last_gate_exp = pl.slice(gate_exp, [1, 1], [l - 1, 0])
    final_state_1 = pl.col_expand_mul(state, last_gate_exp)
    state_new = pl.add(pl.sub(final_state_1, temp_matmul_vprime), temp_matmul_value)
    
    attn_tmp = pl.mul(pl.mul(attn, decay_mask), tril)
    chunk_attn_value = pl.matmul(attn_tmp, value)
    
    chunk_attn_vprime = pl.matmul(attn_tmp, v_prime)
    chunk_attn_out = pl.add(pl.sub(attn_inter, chunk_attn_vprime), chunk_attn_value)
    
    return chunk_attn_out, state_new


def build_chunk_gated_delta_rule_program(b, nqk, nv, d, l):
    b_cfg = b
    nqk_cfg = nqk
    nv_cfg = nv
    d_cfg = d
    l_cfg = l
    group = nv_cfg // nqk_cfg
    
    @pl.program
    class ChunkGatedDeltaRule:
        @pl.function(type=pl.FunctionType.Opaque)
        def chunk_gated_delta_rule(
            self,
            query: pl.Tensor[["T", nqk_cfg, d_cfg], pl.FP32],
            key: pl.Tensor[["T", nqk_cfg, d_cfg], pl.FP32],
            value: pl.Tensor[["T", nv_cfg, d_cfg], pl.FP32],
            beta: pl.Tensor[["T", nv_cfg], pl.FP32],
            gate: pl.Tensor[["T", nv_cfg], pl.FP32],
            states: pl.Tensor[[b_cfg, nv_cfg, d_cfg, d_cfg], pl.FP32],
            mask: pl.Tensor[[l_cfg, l_cfg], pl.FP32],
            tril_mask: pl.Tensor[[l_cfg, l_cfg], pl.FP32],
            eye: pl.Tensor[[16, l_cfg], pl.FP32],
            act_seq_len: pl.Tensor[[b_cfg + 1], pl.INT32],
            core_attn_out: pl.Tensor[["T", nv_cfg, d_cfg], pl.FP32],
            last_state_data: pl.Tensor[[b_cfg, nv_cfg, d_cfg, d_cfg], pl.FP32],
        ) -> None:
            with pl.auto_incore():
                for b_idx in pl.range(b_cfg):
                    s_end = pl.tensor.read(act_seq_len, [b_idx + 1])
                    s_start = pl.tensor.read(act_seq_len, [b_idx])
                    s = s_end - s_start
                    b_ofs = s_start
                    
                    for nv_idx in pl.range(nv_cfg):
                        nqk_idx = nv_idx // group
                        last_state = pl.slice(states, [d_cfg, d_cfg], [b_idx, nv_idx, 0, 0])
                        
                        for s_idx in pl.range(0, s, l_cfg):
                            bs_ofs = b_ofs + s_idx
                            
                            query_view = pl.slice(query, [l_cfg, 1, d_cfg], [bs_ofs, nqk_idx, 0])
                            key_view = pl.slice(key, [l_cfg, 1, d_cfg], [bs_ofs, nqk_idx, 0])
                            value_view = pl.slice(value, [l_cfg, 1, d_cfg], [bs_ofs, nv_idx, 0])
                            beta_view = pl.slice(beta, [l_cfg, 1], [bs_ofs, nv_idx])
                            gate_view = pl.slice(gate, [l_cfg, 1], [bs_ofs, nv_idx])
                            
                            query_view_2d = pl.reshape(query_view, [l_cfg, d_cfg])
                            key_view_2d = pl.reshape(key_view, [l_cfg, d_cfg])
                            value_view_2d = pl.reshape(value_view, [l_cfg, d_cfg])
                            
                            query_norm, key_norm = l2norm(query_view_2d, key_view_2d)
                            scale = 1.0 / (d_cfg ** 0.5)
                            query_scale = pl.muls(query_norm, scale)
                            
                            gate_cum, decay_mask, a_block, key_beta = pre_attn(
                                gate_view, key_norm, beta_view, tril_mask, mask, l_cfg
                            )
                            
                            a_block_inverse = inverse_pto(a_block, eye, l_cfg)
                            
                            value_out, key_cum_out = cal_value_and_key_cumdecay(
                                a_block_inverse, value_view_2d, beta_view, key_beta, gate_cum
                            )
                            
                            chunk_attn_out_tile, cur_state = recurrent_state_attn_all(
                                query=query_scale,
                                key=key_norm,
                                value=value_out,
                                k_cumdecay=key_cum_out,
                                gate=gate_cum,
                                state=last_state,
                                decay_mask=decay_mask,
                                tril=tril_mask,
                                l=l_cfg
                            )
                            
                            last_state = cur_state
                            
                            chunk_attn_out_3d = pl.reshape(chunk_attn_out_tile, [l_cfg, 1, d_cfg])
                            core_attn_out = pl.assemble(core_attn_out, chunk_attn_out_3d, [bs_ofs, nv_idx, 0])
                            last_state_data = pl.assemble(last_state_data, last_state, [b_idx, nv_idx, 0, 0])
    
    return ChunkGatedDeltaRule