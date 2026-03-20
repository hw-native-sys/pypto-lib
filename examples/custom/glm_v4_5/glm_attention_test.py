# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
GLM-4.5 Attention Test Module (PTO 3.0)

Migrated from PTO 2.0 to PTO 3.0.
"""

from __future__ import annotations

import os
import math

import torch

from pypto.runtime import RunConfig, run, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy


B = 8
S1 = 1
S2 = 16384
Q_D = 128
NQ = 12
NKV = 1
BLOCK_SIZE = 128


def gen_block_table(actual_seq_len, block_size, block_table_shape):
    block_num_per_batch = []
    block_num = 0
    
    if isinstance(actual_seq_len, torch.Tensor):
        if actual_seq_len.device.type != 'cpu':
            actual_seq_len_cpu = actual_seq_len.cpu()
        else:
            actual_seq_len_cpu = actual_seq_len
        
        for actual_seq in actual_seq_len_cpu:
            block_num_per_batch.append(math.ceil(actual_seq.item() / block_size))
            block_num += math.ceil(actual_seq.item() / block_size)
    else:
        for actual_seq in actual_seq_len:
            block_num_per_batch.append(math.ceil(actual_seq / block_size))
            block_num += math.ceil(actual_seq / block_size)
    
    block_idx_list = torch.arange(0, block_num, dtype=torch.int32)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))]
    
    block_table = torch.full(block_table_shape, -1, dtype=torch.int32)
    block_idx = 0
    block_table_batch_idx = 0
    
    for idx in block_num_per_batch:
        for j in range(idx):
            block_table[block_table_batch_idx][j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_batch_idx += 1
    
    return block_table


def kv_cache_concat_bsnd(k_cache, v_cache, block_table, b, nkv, q_d, block_size, kv_cache_actual_seq):
    dtype = k_cache.dtype
    device = k_cache.device
    
    if isinstance(kv_cache_actual_seq, torch.Tensor):
        if kv_cache_actual_seq.device.type != 'cpu':
            kv_cache_actual_seq_cpu = kv_cache_actual_seq.cpu()
        else:
            kv_cache_actual_seq_cpu = kv_cache_actual_seq
        kv_max = (torch.max(kv_cache_actual_seq_cpu).item() + block_size - 1) // block_size * block_size
    else:
        kv_max = (max(kv_cache_actual_seq) + block_size - 1) // block_size * block_size
    
    k_cache_bsnd = torch.zeros([b, kv_max, nkv, q_d], dtype=dtype, device=device)
    v_cache_bsnd = torch.zeros([b, kv_max, nkv, q_d], dtype=dtype, device=device)
    
    for b_idx in range(b):
        block_list = block_table[b_idx]
        s_idx = 0
        
        for _, block_idx in enumerate(block_list):
            if block_idx == -1:
                break
            
            start_idx = s_idx * block_size
            end_idx = (s_idx + 1) * block_size
            
            k_cache_bsnd[b_idx:b_idx + 1, start_idx:end_idx, :, :] = k_cache[block_idx:block_idx + 1, :, :, :]
            v_cache_bsnd[b_idx:b_idx + 1, start_idx:end_idx, :, :] = v_cache[block_idx:block_idx + 1, :, :, :]
            s_idx += 1
    
    return k_cache_bsnd, v_cache_bsnd


def softmax(x, is_fp16=False):
    if is_fp16:
        original_dtype = x.dtype
        x = x.float()
    x_max = x.max(dim=-1, keepdim=True).values
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdim=True)
    ans = y / x_sum
    if is_fp16:
        ans = ans.to(original_dtype)
        x_max = x_max.to(original_dtype)
        x_sum = x_sum.to(original_dtype)
    return ans, x_max, x_sum


def golden_attention(q, k_cache_bsnd, v_cache_bsnd, kv_cache_actual_seq, softmax_scale):
    b = q.shape[0] // S1
    nq = q.shape[1]
    d = q.shape[2]
    nkv = k_cache_bsnd.shape[2]
    group = nq // nkv
    
    attention_output = torch.zeros_like(q)
    
    for i in range(b):
        for j in range(S1):
            for n2_idx in range(nkv):
                kv_seq_len = kv_cache_actual_seq[i].item()
                seq_len = kv_seq_len - S1 + 1 + j
                q_bs = q[i * S1 + j]
                k_bs = k_cache_bsnd[i, :seq_len, n2_idx:n2_idx + 1].reshape(seq_len, d)
                v_bs = v_cache_bsnd[i, :seq_len, n2_idx:n2_idx + 1].reshape(seq_len, d)
                
                qk_bmm_res = torch.matmul(q_bs, k_bs.transpose(1, 0))
                qk_ele_res = qk_bmm_res * softmax_scale
                softmax_res, _, _ = softmax(qk_ele_res, True)
                bmm2_res = torch.matmul(softmax_res, v_bs)
                
                attention_output[i * S1 + j] = bmm2_res
    
    return attention_output


def build_tensor_specs(b, s1, s2, q_d, nq, nkv, block_size):
    kv_num_blocks = b * ((s2 + block_size - 1) // block_size)
    max_num_blocks_per_query = (s2 + block_size - 1) // block_size
    
    actual_seq_values = [s2] * b
    actual_seq_tensor = torch.tensor(actual_seq_values, dtype=torch.int32)
    
    q_shape = [b * s1, nq, q_d]
    kv_shape = [kv_num_blocks, block_size, nkv, q_d]
    block_table_shape = [b, max_num_blocks_per_query]
    
    q = torch.empty(q_shape, dtype=torch.bfloat16).uniform_(-1, 1)
    k = torch.empty(kv_shape, dtype=torch.bfloat16).uniform_(-1, 1)
    v = torch.empty(kv_shape, dtype=torch.bfloat16).uniform_(-1, 1)
    
    block_table = gen_block_table(actual_seq_tensor, block_size, block_table_shape)
    
    k_cache_bsnd, v_cache_bsnd = kv_cache_concat_bsnd(
        k, v, block_table, b, nkv, q_d, block_size, actual_seq_tensor
    )
    
    softmax_scale = q_d ** -0.5
    golden_out = golden_attention(q, k_cache_bsnd, v_cache_bsnd, actual_seq_tensor, softmax_scale)
    
    attn_out = torch.zeros_like(q)
    
    return [
        TensorSpec("q", q_shape, torch.bfloat16, init_value=q),
        TensorSpec("k", kv_shape, torch.bfloat16, init_value=k),
        TensorSpec("v", kv_shape, torch.bfloat16, init_value=v),
        TensorSpec("block_table", block_table_shape, torch.int32, init_value=block_table),
        TensorSpec("kv_act_seqs", [b], torch.int32, init_value=actual_seq_tensor),
        TensorSpec("attn_out", q_shape, torch.bfloat16, is_output=True),
    ], golden_out


def compile_and_run(
    batch: int = B,
    s1: int = S1,
    s2: int = S2,
    q_d: int = Q_D,
    nq: int = NQ,
    nkv: int = NKV,
    block_size: int = BLOCK_SIZE,
    platform: str = "a2a3",
    device_id: int = 0,
    work_dir: str | None = None,
    dump_passes: bool = True,
):
    from glm_attention import build_glm_attention_program
    
    tensor_specs, golden_out = build_tensor_specs(
        batch, s1, s2, q_d, nq, nkv, block_size
    )
    
    program = build_glm_attention_program(
        batch=batch, s1=s1, s2=s2, q_d=q_d, nq=nq, nkv=nkv, block_size=block_size
    )
    
    if work_dir is None:
        work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "glm_attention_dump"))
    
    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=None,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=2e-2,
            atol=2e-2,
            strategy=OptimizationStrategy.Default,
            dump_passes=dump_passes,
            backend_type=BackendType.Ascend950,
        ),
    )
    
    if not result.passed and result.error and "code_runner" in result.error:
        print("Result: COMPILE OK — device run skipped (code_runner not found).")
        print(f"  Generated kernels/orchestration: {work_dir}")
        return result
    
    if not result.passed and result.error:
        print(f"Result: {result.error}")
        print(f"  Pass dumps may still have been written to: {work_dir}")
    else:
        print(f"  Generated kernels/orchestration: {work_dir}")
    
    return result


if __name__ == "__main__":
    compile_and_run()