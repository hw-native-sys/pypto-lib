# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Small shuffled-page mixed-pipe test, including empty queries and leaf tails."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import pypto.language as pl
from pypto.runtime import pto_isa_include_dir
from golden import TensorSpec, run

SOURCE_DIR = Path(__file__).resolve().parent
ROOT = Path("build_output/indexer_cube_score_smoke")
ROOT.mkdir(parents=True, exist_ok=True)
COUNTS = [0, 1, 2, 127, 128, 129, 257, 1025, 2047, 2048, 2049, 4096, 8191, 8192, 8193, 8257]
CANARY = -987654.0
WEIGHT = None
REFERENCE = []
CANCELLATION_QUERY = 1
INT8_BOUNDARY_QUERY = 2
FIXTURE = {}
FIXTURE_TENSORS = {}

@pl.jit.extern(core_type="mixed", aic_source=SOURCE_DIR / "score_fused.cpp",
               aiv_source=SOURCE_DIR / "score_fused.cpp", dual_aiv_dispatch=True,
               include_dirs=(pto_isa_include_dir(), SOURCE_DIR))
def fused_extern(pairs: pl.Out[pl.Tensor], pipe_workspace: pl.InOut[pl.Tensor],
                 query: pl.Tensor, coefficient_hi: pl.Tensor, coefficient_lo: pl.Tensor,
                 keys: pl.Tensor, scales: pl.Tensor, table: pl.Tensor,
                 positions: pl.Tensor, lengths: pl.Tensor, constants: pl.Tensor,
                 max_leaves: pl.Scalar[pl.INDEX], table_stride: pl.Scalar[pl.INDEX]) -> pl.Tensor:
    ...

@pl.jit
def fused_test(
    pairs: pl.InOut[pl.Tensor[[1024, 1024], pl.FP32]],
    query: pl.Tensor[[1024, 128], pl.INT8],
    coefficient_hi: pl.Tensor[[16, 64], pl.FP16],
    coefficient_lo: pl.Tensor[[16, 128], pl.FP16],
    keys: pl.Tensor[[8384, 128], pl.INT8],
    scales: pl.Tensor[[1, 8384], pl.FP32],
    table: pl.Tensor[[2, 8192], pl.INT32],
    positions: pl.Tensor[[16], pl.INT32],
    lengths: pl.Tensor[[2], pl.INT32],
    constants: pl.Tensor[[1, 5376], pl.FP32],
):
    pipe_workspace = pl.create_tensor([24, 2048], dtype=pl.FP32)
    with pl.spmd(24, name_hint="compact_score_pipe_topk"):
        pairs = fused_extern(pairs, pipe_workspace, query, coefficient_hi, coefficient_lo,
                             keys, scales, table, positions, lengths, constants, 2, 8192)
    return pairs

def specs():
    global WEIGHT
    rng = torch.Generator().manual_seed(20260919)
    query = torch.randint(-64,65,(1024,128),dtype=torch.int8,generator=rng)
    keys = torch.randint(-64,65,(8384,128),dtype=torch.int8,generator=rng)
    WEIGHT = (torch.rand((16,64),generator=rng) * 2 - 1) * .001
    scales = .01 + torch.rand((1,8384),generator=rng) * .02
    table = torch.zeros((2,8192),dtype=torch.int32)
    for batch in range(2):
        table[batch,:262] = torch.randperm(262,generator=rng).int()
    # Retain the original mixed-page fixture, using its one-candidate query
    # as a dedicated regression for the target-magnitude coefficient loss.
    # Every dot is 128*127^2, so this remains in the actual quantizer range.
    query[CANCELLATION_QUERY*64:(CANCELLATION_QUERY+1)*64].fill_(127)
    cancellation_row = int(table[0,0])*32
    keys[cancellation_row].fill_(127)
    scales[0,cancellation_row] = 0.00784325785934925
    WEIGHT[CANCELLATION_QUERY,:32] = -0.010233103297650814
    WEIGHT[CANCELLATION_QUERY,32:] = 0.010233104228973389
    # This two-candidate query covers both a large negative dot and the
    # true signed-INT8 positive endpoint. Keep first key row =127 so the
    # separate cancellation query is unchanged. Other long queries retain
    # random Q/weights; shared cache modifications are included in golden.
    query[INT8_BOUNDARY_QUERY*64:(INT8_BOUNDARY_QUERY+1)*64].fill_(-128)
    boundary_negative_row = int(table[0,0])*32
    boundary_positive_row = boundary_negative_row+1
    keys[boundary_negative_row].fill_(127)
    keys[boundary_positive_row].fill_(-128)
    WEIGHT[INT8_BOUNDARY_QUERY].fill_(2**-10)
    coefficient = WEIGHT * 16384
    hi = coefficient.half()
    residual = coefficient - hi.float()
    low = residual.half()
    third = (residual - low.float()).half()
    lo = torch.cat((low,third),dim=1).contiguous()
    reconstructed = (hi.double()+low.double()+third.double())/16384
    FIXTURE.update(seed=20260919,query_count=len(COUNTS),batch_count=2,coefficient_hi_shape=list(hi.shape),
                   coefficient_low_third_shape=list(lo.shape),
                   coefficient_reconstruction_maxabs=float((reconstructed-WEIGHT.double()).abs().max()),
                   cancellation_query=CANCELLATION_QUERY,cancellation_physical_row=cancellation_row,
                   cancellation_dot=128*127*127,cancellation_key_scale=float(scales[0,cancellation_row]),
                   cancellation_coefficients=[float(WEIGHT[CANCELLATION_QUERY,0]),float(WEIGHT[CANCELLATION_QUERY,-1])])
    FIXTURE["int8_boundary"] = dict(query=INT8_BOUNDARY_QUERY,
        logical_indices=[0,1],physical_rows=[boundary_negative_row,boundary_positive_row],
        dot_values=[-128*128*127,128*128*128],coefficient=2**-10)
    positions = torch.tensor([4*c-1 if c else 0 for c in COUNTS],dtype=torch.int32)
    lengths = torch.tensor([4*1026,4*8258],dtype=torch.int32)
    constants = torch.empty((1,5376),dtype=torch.float32)
    nm = torch.zeros((64,16),dtype=torch.float16); nm[:,0] = -.75
    ni = -torch.eye(64,dtype=torch.float16)
    ad = torch.zeros((64,16),dtype=torch.float16); ad[:,0] = 2**-14
    br = torch.zeros((16,256),dtype=torch.float16); br[0,:] = 1024
    bias = torch.full((1,256),1145043968,dtype=torch.int32)
    for first,last,value in [(0,512,nm),(512,2560,ni),(2560,3072,ad),(3072,5120,br),(5120,5376,bias)]:
        constants[0,first:last] = value.contiguous().view(torch.float32).flatten()
    values = dict(query=query,coefficient_hi=hi,coefficient_lo=lo,keys=keys,
                  scales=scales,table=table,positions=positions,lengths=lengths,constants=constants)
    FIXTURE_TENSORS.update(values)
    FIXTURE_TENSORS["pairs"] = torch.full((1024,1024),CANARY)
    result = [TensorSpec("pairs",[1024,1024],torch.float32,
                         init_value=lambda:torch.full((1024,1024),CANARY))]
    for name,value in values.items():
        result.append(TensorSpec(name,list(value.shape),value.dtype,init_value=lambda x=value:x))
    return result

def golden(v):
    REFERENCE.clear()
    out = v["pairs"]
    out.fill_(CANARY)
    for query,count in enumerate(COUNTS):
        for leaf in range(2):
            valid = max(0,min(8192,count-leaf*8192))
            if not valid:
                continue
            span = ((valid+255)//256)*128
            for lane in range(2):
                begin = leaf*8192+lane*span
                n = max(0,min(span,valid-lane*span))
                logical = torch.arange(begin,begin+n)
                physical = v["table"][query//8,logical//32].long()*32 + logical%32
                dot = v["query"][query*64:(query+1)*64].int() @ v["keys"][physical].int().T
                score = (dot.float().clamp_min(0)*WEIGHT[query,:,None]).sum(dim=0)*v["scales"][0,physical]
                selected_values, selected = score.topk(min(512,n))
                selected_indices = logical[selected].int()
                row = query*64+leaf*2+lane
                out[row].fill_(torch.finfo(torch.float32).min)
                out[row,:2*len(selected_values):2] = selected_values
                out[row,1:2*len(selected_values):2] = selected_indices.view(torch.float32)
                REFERENCE.append((row,begin,n,score,selected_indices))

def compare(actual,expected,**_):
    actual = actual.cpu()
    untouched = expected == CANARY
    canaries = bool((actual[untouched] == CANARY).all())
    bad_scores = missing = duplicate = invalid = total = 0
    max_abs = 0.0
    for row,begin,n,scores,selected_indices in REFERENCE:
        take = min(n,512)
        values = actual[row,0:take*2:2].contiguous()
        indices = actual[row,1:take*2:2].contiguous().view(torch.int32)
        expected_values = expected[row,0:take*2:2]
        error = (values-expected_values).abs()
        bad_scores += int((~torch.isfinite(values) | (error > 1e-4+expected_values.abs()/128)).sum())
        max_abs = max(max_abs,float(error.max()) if take else 0)
        total += take
        missing += len(set(selected_indices.tolist())-set(indices.tolist()))
        duplicate += take-len(set(indices.tolist()))
        invalid += int(((indices<begin)|(indices>=begin+n)).sum())
        if take < 512:
            invalid += int((actual[row,take*2::2] != torch.finfo(torch.float32).min).sum())
    report = dict(canaries_preserved=canaries,compared_scores=total,score_outliers=bad_scores,
                  selected_set_missing=missing,duplicate_indices=duplicate,invalid_indices=invalid,
                  max_abs=max_abs,visible_counts=COUNTS,fixture=FIXTURE)
    # Apply the identical atol/rtol/0.1%-ratio rule to this one-score probe
    # independently; the other thousands of scores must not hide its failure.
    cancel_row = CANCELLATION_QUERY*64
    cancel_actual = actual[cancel_row,0]
    cancel_expected = expected[cancel_row,0]
    cancel_error = float((cancel_actual-cancel_expected).abs())
    cancel_tolerance = float(1e-4+cancel_expected.abs()/128)
    cancel_ok = bool(torch.isfinite(cancel_actual)) and cancel_error <= cancel_tolerance
    report["target_magnitude_cancellation"] = dict(actual=float(cancel_actual),expected=float(cancel_expected),
        abs_error=cancel_error,tolerance=cancel_tolerance,score_outliers=int(not cancel_ok),
        compared_scores=1,allowed_error_ratio=.001,passed=cancel_ok)
    boundary_row = INT8_BOUNDARY_QUERY*64
    boundary_indices = actual[boundary_row,1:4:2].contiguous().view(torch.int32)
    boundary_values = actual[boundary_row,0:4:2]
    expected_indices = expected[boundary_row,1:4:2].contiguous().view(torch.int32)
    expected_values = expected[boundary_row,0:4:2]
    boundary_ok = True
    boundary_records = []
    for logical_index, dot in zip((0,1),(-128*128*127,128*128*128)):
        actual_mask = boundary_indices == logical_index
        expected_mask = expected_indices == logical_index
        present = int(actual_mask.sum()) == 1 and int(expected_mask.sum()) == 1
        av = float(boundary_values[actual_mask][0]) if present else None
        ev = float(expected_values[expected_mask][0]) if present else None
        error = abs(av-ev) if present else None
        tol = 1e-4+abs(ev)/128 if present else None
        okay = present and bool(torch.isfinite(boundary_values[actual_mask]).all()) and error <= tol
        if logical_index == 0:
            okay = okay and av == ev == 0.0  # ReLU of a negative dot must be zero.
        boundary_ok = boundary_ok and okay
        boundary_records.append(dict(logical_index=logical_index,dot=dot,actual=av,expected=ev,
                                     abs_error=error,tolerance=tol,passed=okay))
    report["int8_boundary"] = dict(query=INT8_BOUNDARY_QUERY,passed=boundary_ok,candidates=boundary_records)
    (ROOT/"test_fused_result.json").write_text(json.dumps(report,indent=2)+"\n")
    torch.save(actual,ROOT/"test_fused_actual.pt")
    print(json.dumps(report),flush=True)
    return canaries and bad_scores/max(total,1)<=.001 and missing==duplicate==invalid==0 and cancel_ok and boundary_ok,json.dumps(report)

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("-d","--device",type=int,default=13);p.add_argument("--compile-only",action="store_true")
    p.add_argument("--runtime-dir")
    p.add_argument("--cpu-fixture-only",action="store_true")
    a=p.parse_args();torch.set_num_threads(4)
    if a.cpu_fixture_only:
        specs()
        golden(FIXTURE_TENSORS)
        row = INT8_BOUNDARY_QUERY*64
        values = FIXTURE_TENSORS["pairs"][row,0:4:2]
        indices = FIXTURE_TENSORS["pairs"][row,1:4:2].contiguous().view(torch.int32)
        expected_by_id = {int(i):float(v) for i,v in zip(indices,values)}
        assert expected_by_id[0] == 0.0 and expected_by_id[1] > 0.0
        q = FIXTURE_TENSORS["query"][INT8_BOUNDARY_QUERY*64:(INT8_BOUNDARY_QUERY+1)*64]
        k = FIXTURE_TENSORS["keys"][FIXTURE["int8_boundary"]["physical_rows"]]
        dots = q.int()@k.int().T
        assert bool((dots[:,0] == -128*128*127).all())
        assert bool((dots[:,1] == 128*128*128).all())
        payload = dict(cpu_only=True,device_execution=False,passed=True,fixture=FIXTURE,
                       boundary_expected_by_id=expected_by_id,
                       cancellation_golden=float(FIXTURE_TENSORS["pairs"][CANCELLATION_QUERY*64,0]),
                       selected_scores=sum(min(n,512) for _,_,n,_,_ in REFERENCE),
                       reference_rows=len(REFERENCE))
        (ROOT/"test_fused_cpu_fixture.json").write_text(json.dumps(payload,indent=2)+"\n")
        print(json.dumps(payload,indent=2))
        raise SystemExit(0)
    source_hash = lambda: {name:hashlib.sha256((SOURCE_DIR/name).read_bytes()).hexdigest()
                          for name in ("score_fused.cpp","cube_score_exact.hpp","vector_topk.hpp")}
    before = source_hash()
    result=run(fn=fused_test,specs=specs(),golden_fn=golden,compare_fn={"pairs":compare},
               compile_only=a.compile_only,runtime_dir=a.runtime_dir,save_data=True,config={"platform":"a2a3","device_id":a.device})
    payload = dict(passed=result.passed,error=result.error,work_dir=str(result.work_dir),
                   compile_only=a.compile_only,device=a.device,runtime_dir=a.runtime_dir,
                   fixture=FIXTURE,source_sha256_before=before,source_sha256_after=source_hash(),
                   harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    payload["source_unchanged_during_run"] = payload["source_sha256_before"] == payload["source_sha256_after"]
    record = ROOT/("test_fused_compile.json" if a.compile_only else "test_fused_run.json")
    record.write_text(json.dumps(payload,indent=2)+"\n")
    print(result,flush=True)
    if not result.passed: raise SystemExit(1)
