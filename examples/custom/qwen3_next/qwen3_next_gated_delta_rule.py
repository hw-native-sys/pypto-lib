# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Qwen3-next Gated Delta Rule Test Module (PTO 3.0)

Migrated from PTO 2.0 to PTO 3.0.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

import pypto.language as pl
from pypto.runtime import RunConfig, run, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy


L = 128
D = 128


def gen_dims(params):
    dims = {}
    dims["T"] = params["T"]
    dims["B"] = params["B"]
    dims["Nqk"] = params["Nqk"]
    dims["Nv"] = params["Nv"]
    dims["D"] = D
    dims["L"] = L
    return dims


def gen_inputs(dims, dtype=torch.float32):
    t = dims["T"]
    b = dims["B"]
    nqk = dims["Nqk"]
    nv = dims["Nv"]
    d = dims["D"]
    l = dims["L"]
    
    query = torch.rand([t, nqk, d], dtype=dtype) * (1.3655 + 0.2785) - (1.3655 + 0.2785)
    key = torch.rand([t, nqk, d], dtype=dtype) * (1.4664 + 0.2785) - (1.4664 + 0.2785)
    value = torch.rand([t, nv, d], dtype=dtype) * (1.6488 + 0.2785) - (1.6488 + 0.2785)
    beta = torch.rand([t, nv], dtype=dtype) * (0.8927 - 0.0889) - (0.8927 - 0.0889)
    gate = torch.rand([t, nv], dtype=dtype) * (-0.1343 + 37.5452) - (-0.1343 + 37.5452)
    states = torch.zeros([b, nv, d, d], dtype=dtype)
    
    seq_len_per_batch = t // b
    act_seq_len = [i * seq_len_per_batch for i in range(b + 1)]
    act_seq_len = torch.tensor(act_seq_len, dtype=torch.int32)
    
    mask = torch.tril(-torch.ones([l, l], dtype=dtype), diagonal=-1)
    tril_mask = torch.ones([l, l], dtype=dtype).tril()
    eye_data = torch.eye(16, dtype=dtype).repeat(1, 8)
    eye_data_unaligned = torch.eye(16, dtype=dtype)
    
    return {
        "query": query,
        "key": key,
        "value": value,
        "beta": beta,
        "gate": gate,
        "states": states,
        "act_seq_len": act_seq_len,
        "mask": mask,
        "tril_mask": tril_mask,
        "eye_data": eye_data,
        "eye_data_unaligned": eye_data_unaligned,
    }


def golden_chunk_gated_delta_rule(inputs: dict, dims: dict):
    query = inputs["query"]
    key = inputs["key"]
    value = inputs["value"]
    gate = inputs["gate"]
    beta = inputs["beta"]
    states = inputs["states"]
    act_seq_len = inputs["act_seq_len"]
    
    core_attn_out, final_state = segs_chunk_gated_delta_rule(
        query=query.clone(),
        key=key.clone(),
        value=value.clone(),
        gate=gate.clone(),
        beta=beta.clone(),
        act_seq_len=act_seq_len.clone(),
        chunk_size=128,
        initial_state=states.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    
    return {
        "core_attn_out": core_attn_out,
        "final_state": final_state,
    }


def segs_chunk_gated_delta_rule(**kwargs):
    query = kwargs.get("query")
    key = kwargs.get("key")
    value = kwargs.get("value")
    g = kwargs.get("gate")
    beta = kwargs.get("beta")
    act_seq_len = kwargs.get("act_seq_len")
    chunk_size = kwargs.get("chunk_size")
    initial_state = kwargs.get("initial_state")
    output_final_state = kwargs.get("output_final_state")
    use_qk_l2norm_in_kernel = kwargs.get("use_qk_l2norm_in_kernel")
    
    t, n1, d = query.shape
    t, n, d = value.shape
    batch = act_seq_len.shape[0] - 1
    
    query = query.repeat_interleave(n // n1, dim=1)
    key = key.repeat_interleave(n // n1, dim=1)
    
    final_state = torch.zeros([batch, n, d, d], dtype=torch.float32, device=query.device)
    
    query, key, value, beta, g = \
        [x.transpose(0, 1).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]
    final_attn = torch.zeros([t, n, d], dtype=torch.float32, device=query.device)
    
    for b_idx in range(batch):
        s = act_seq_len[b_idx + 1] - act_seq_len[b_idx]
        b_ofs = act_seq_len[b_idx]
        seg_s = 128
        pad_size = (chunk_size - s % chunk_size) % chunk_size
        pad_seq_length = s + pad_size
        batch_query, batch_key, batch_value = \
            [F.pad(x[:, b_ofs:b_ofs + s], (0, 0, 0, pad_size)) for x in (query, key, value)]
        batch_beta, batch_g = [F.pad(x[:, b_ofs:b_ofs + s], (0, pad_size)) for x in (beta, g)]
        result_list = []
        recurrent_state = initial_state[b_idx:b_idx + 1, ...]
        for s_idx in range(0, pad_seq_length, seg_s):
            chunk_query, chunk_key, chunk_value = \
                [x[:, s_idx:s_idx + seg_s, :].reshape(1, n, seg_s, d) for x in (batch_query, batch_key, batch_value)]
            chunk_gate, chunk_beta = [x[:, s_idx:s_idx + seg_s].reshape(1, n, seg_s) for x in (batch_g, batch_beta)]
            cur_attn, cur_state = segs_chunk_gated_delta_rule_sub(
                query=chunk_query, key=chunk_key, value=chunk_value,
                g=chunk_gate, beta=chunk_beta, chunk_size=chunk_size, initial_state=recurrent_state,
                output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            result_list.append(cur_attn.squeeze(0))
            recurrent_state = cur_state
        batch_attn = torch.cat(result_list, dim=0)[:s]
        final_attn[b_ofs:b_ofs + s] = batch_attn
        final_state[b_idx:b_idx + 1, ...] = recurrent_state
    return final_attn, final_state


def segs_chunk_gated_delta_rule_sub_inverse(attn, chunk_size):
    for index in range(1, chunk_size):
        line = attn[..., index, :index].clone()
        sub = attn[..., :index, :index].clone()
        attn[..., index, :index] = line + (line.unsqueeze(-1) * sub).sum(-2)
    return attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)


def segs_chunk_gated_delta_rule_sub_cycle(**kwargs):
    query = kwargs.get("query")
    key = kwargs.get("key")
    value = kwargs.get("value")
    decay_mask = kwargs.get("decay_mask")
    k_cumdecay = kwargs.get("k_cumdecay")
    g = kwargs.get("g")
    last_recurrent_state = kwargs.get("last_recurrent_state")
    total_sequence_length = kwargs.get("total_sequence_length")
    chunk_size = kwargs.get("chunk_size")
    
    attn_out = torch.zeros_like(value).to(query.device)
    attn_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    
    for index in range(0, total_sequence_length // chunk_size):
        q_index, k_index, v_index = query[:, :, index], key[:, :, index], value[:, :, index]
        attn = (q_index @ k_index.transpose(-1, -2) * decay_mask[:, :, index]).masked_fill_(attn_mask, 0)
        v_new = v_index - (k_cumdecay[:, :, index]) @ last_recurrent_state
        attn_out[:, :, index] = (q_index * g[:, :, index, :, None].exp()) @ last_recurrent_state + attn @ v_new
        last_recurrent_state = last_recurrent_state * g[:, :, index, -1, None, None].exp() + \
            (k_index * (g[:, :, index, -1, None] - g[:, :, index]).exp()[..., None]).transpose(-1, -2) @ v_new
        
    return attn_out, last_recurrent_state


def segs_chunk_gated_delta_rule_sub(**kwargs):
    query = kwargs.get("query")
    key = kwargs.get("key")
    value = kwargs.get("value")
    g = kwargs.get("g")
    beta = kwargs.get("beta")
    chunk_size = kwargs.get("chunk_size")
    initial_state = kwargs.get("initial_state")
    output_final_state = kwargs.get("output_final_state")
    use_qk_l2norm_in_kernel = kwargs.get("use_qk_l2norm_in_kernel")
    
    b, n, s, d = value.shape
    
    initial_state = initial_state.transpose(3, 2)
    if use_qk_l2norm_in_kernel:
        query = query * torch.rsqrt((query * query).sum(dim=-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)
    
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = [F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value)]
    beta, g = [F.pad(x, (0, pad_size)) for x in (beta, g)]
    
    total_sequence_length = sequence_length + pad_size
    query = query * (1 / (query.shape[-1] ** 0.5))
    
    v_beta, k_beta = [x * beta.unsqueeze(-1) for x in (value, key)]
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    
    attn = segs_chunk_gated_delta_rule_sub_inverse(attn, chunk_size)
    
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    
    if initial_state is None:
        last_recurrent_state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=query.device).to(value)
    else:
        last_recurrent_state = initial_state.to(value)
    
    attn_out, last_recurrent_state = segs_chunk_gated_delta_rule_sub_cycle(
        query=query, key=key, value=value,
        decay_mask=decay_mask, k_cumdecay=k_cumdecay, g=g, last_recurrent_state=last_recurrent_state,
        total_sequence_length=total_sequence_length, chunk_size=chunk_size
    )
    
    if not output_final_state:
        last_recurrent_state = None
    attn_out = attn_out.reshape(attn_out.shape[0], attn_out.shape[1], -1, attn_out.shape[-1])
    attn_out = attn_out[:, :, :sequence_length].transpose(1, 2).contiguous()
    
    last_recurrent_state = last_recurrent_state.transpose(3, 2)
    
    return attn_out, last_recurrent_state


def gen_data(case_name):
    if case_name.startswith("ChunkGatedDeltaRuleSTest.b2_nqk2_nv4_s1k"):
        params = {"T": 1024 * 2, "B": 2, "Nqk": 2, "Nv": 4}
    elif case_name.startswith("ChunkGatedDeltaRuleSTest.b2_nqk4_nv8_s4k"):
        params = {"T": 1024 * 8, "B": 2, "Nqk": 4, "Nv": 8}
    elif case_name.startswith("ChunkGatedDeltaRuleSTest.b2_nqk2_nv4_s8k"):
        params = {"T": 1024 * 16, "B": 2, "Nqk": 2, "Nv": 4}
    elif case_name.startswith("ChunkGatedDeltaRuleSTest.b2_nqk4_nv8_s8k"):
        params = {"T": 1024 * 16, "B": 2, "Nqk": 4, "Nv": 8}
    elif case_name.startswith("ChunkGatedDeltaRuleSTest.b1_nqk16_nv32_s32k"):
        params = {"T": 1024 * 32, "B": 1, "Nqk": 16, "Nv": 32}
    else:
        params = {"T": 1024 * 2, "B": 2, "Nqk": 2, "Nv": 4}
    
    seed = 0
    torch.manual_seed(seed)
    dims = gen_dims(params)
    inputs = gen_inputs(dims, torch.float32)
    outputs = golden_chunk_gated_delta_rule(inputs, dims)
    return dims, inputs, outputs


def build_tensor_specs(dims, inputs, outputs):
    b = dims["B"]
    nqk = dims["Nqk"]
    nv = dims["Nv"]
    d = dims["D"]
    l = dims["L"]
    t = dims["T"]
    
    return [
        TensorSpec("query", [t, nqk, d], torch.float32, init_value=inputs["query"]),
        TensorSpec("key", [t, nqk, d], torch.float32, init_value=inputs["key"]),
        TensorSpec("value", [t, nv, d], torch.float32, init_value=inputs["value"]),
        TensorSpec("beta", [t, nv], torch.float32, init_value=inputs["beta"]),
        TensorSpec("gate", [t, nv], torch.float32, init_value=inputs["gate"]),
        TensorSpec("states", [b, nv, d, d], torch.float32, init_value=inputs["states"]),
        TensorSpec("mask", [l, l], torch.float32, init_value=inputs["mask"]),
        TensorSpec("tril_mask", [l, l], torch.float32, init_value=inputs["tril_mask"]),
        TensorSpec("eye", [16, l], torch.float32, init_value=inputs["eye_data"]),
        TensorSpec("act_seq_len", [b + 1], torch.int32, init_value=inputs["act_seq_len"]),
        TensorSpec("core_attn_out", [t, nv, d], torch.float32, is_output=True),
        TensorSpec("last_state_data", [b, nv, d, d], torch.float32, is_output=True),
    ]


def compile_and_run(
    case_name: str = "ChunkGatedDeltaRuleSTest.b2_nqk2_nv4_s1k",
    platform: str = "a2a3",
    device_id: int = 0,
    work_dir: str | None = None,
    dump_passes: bool = True,
):
    from gated_delta_rule_impl import build_chunk_gated_delta_rule_program
    
    dims, inputs, golden_data = gen_data(case_name)
    
    b = dims["B"]
    nqk = dims["Nqk"]
    nv = dims["Nv"]
    d = dims["D"]
    l = dims["L"]
    
    program = build_chunk_gated_delta_rule_program(b, nqk, nv, d, l)
    tensor_specs = build_tensor_specs(dims, inputs, golden_data)
    
    if work_dir is None:
        work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), f"{case_name}_dump"))
    
    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=None,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=1e-3,
            atol=1e-3,
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