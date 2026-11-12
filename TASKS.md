# DeepSeek V4.1 Flash Decode acceptance task

This file is the independent A3 Decode session record. It starts at
`not_run`; no Prefill goal, session, verifier, or passing state is reused.
CPU precision is outside this acceptance. Formal numerical comparisons are
NPU PyPTO versus an independently launched vLLM-Ascend reference process.

## Baseline and provenance

The product checkout is currently detached at the Decode session's appended
commit (the exact hash is recorded by `decode_acceptance.py`). Its parent is
the requested `addb655f7b4f4a4ce56af875724a04d79d3fe325` target. The four
source commits remain unchanged and are not squashed, amended, reset, or
replaced:

| order | commit | author | author time |
| --- | --- | --- | --- |
| 1 | `500858a9abd6ce1d3796a5e26a4bb597cdf622e4` | Adelaide999 &lt;73272686+Adelaide999@users.noreply.github.com&gt; | `2026-10-31T10:13:30+08:00` |
| 2 | `d401f4e82e6348aee1566118c0c768b66e70504d` | sunhaolun &lt;sunhaolun3@huawei.com&gt; | `2026-10-31T17:51:06+08:00` |
| 3 | `c0f23d56800fbd34d4b3442d5df9565be387a955` | Lun7777 &lt;1055318345@qq.com&gt; | `2026-11-02T10:00:49+08:00` |
| 4 | `addb655f7b4f4a4ce56af875724a04d79d3fe325` | Lun7777 &lt;1055318345@qq.com&gt; | `2026-11-02T14:36:58+08:00` |

`eed3280972f6743fc46d68219bcb108596e64578` is the common base. The supplied
source checkout and `cherry-pick-receipt.json` were not available in this
workspace. A temporary clean clone executed the four ordered `git
cherry-pick -x` commands and its generated hashes/tree are recorded in
`decode_cherry_pick_receipt.json`; the product checkout retains the original
source object IDs.

The worktree already contained user changes when this session started:
Decode entry points, documentation, `source_manifest.json`, runtime topology
configuration, `TASKS.md`, and `.pypto-a3-state` build outputs. They were read
and retained. No Prefill files were reset.

## Product boundary

Product compute is PyPTO only: QKV, RoPE, cache read/write, compressed/index
attention, projection, quantization, mHC, Engram, shared/routed experts,
dispatch/combine and output restoration must be PyPTO before a formal gate can
pass. vLLM/vLLM-Ascend and Torch-NPU are reference/diagnostic processes only.
The product import closure scan is:

```bash
PYTHONPATH=$PWD python -m models.deepseek_v4_1_flash_a2.decode_acceptance --json
```

It must report `product_dependency_scan.status=pass`. `_runtime.py` modules
that launch native reference code are reference-only and are not a product
dependency. Torch imports reachable from the current golden/reference helpers
are recorded separately and do not constitute product computation.

## Fixed contract

The machine-readable contract is emitted by
`models/deepseek_v4_1_flash_a2/decode_acceptance.py` and includes:

- `hidden_size=5120`, `num_attention_heads=64`, `num_key_value_heads=1`,
  `head_dim=512`, `q_lora_rank=1280`, `o_groups=8`, `o_lora_rank=1024`;
- MoE intermediate `2304`, `384` routed experts, top-6 routing, one shared
  expert, and `hc_mult=4`;
- W8A8 dynamic channel-scaled projection, INT8 Engram, 128-row pages,
  `enable_expert_parallel=true`, fused MC2, DSA-CP, FlashComm1 and shared
  expert DP;
- decode token, history length, position IDs, initial cache, block/page table,
  slot mapping, valid lengths, mask, active/finished requests, Engram rows,
  cache source IDs and ratio-1/ratio-2 state as required inputs;
- layer output/logits or hidden, cache delta, index/router/dispatch state,
  untouched cache rows and inactive rows as required outputs;
- floating `rtol=0.05`, `atol=0.05`; discrete routes, lengths, masks, source
  IDs and state regions must match exactly.
- TP1/TP2/TP4 and EP2 local shapes, KV-head replication, page layout and
  dispatch/combine ownership are recorded in `decode_shape_contract.py`; the
  runtime-resolved fields remain `not_run` until a capture proves them.
- The constructed runtime contract records `use_sequence_parallel_moe`, cache
  block size, request scheduler, communication groups and fused communication
  switches. Single-machine capacity and per-rank peak memory remain `not_run`
  until real checkpoint buffers are allocated.

The fixed topologies are DP2/TP1/EP2 for the MoE gate, DP8/TP2/EP16 as the
main 16-device gate, and DP16/TP1/EP16 plus DP4/TP4/EP16 as compatibility
checks. `EP=DP*TP`; EP16 never means TP16.

## Search K=2

`decode_candidates.py` registers two executable PyPTO bounded routes with the
same baseline, input boundary, page size, scale dtype, topology and tolerance:

| candidate | executable entry | explored ABI |
| --- | --- | --- |
| A | `decode_fwd.make_decode_step` | canonical `[block,row,kv_head,head_dim]` cache and canonical KN INT8 weights |
| B | `decode_replay.make_decode_replay(steps=1)` | the same cache page with a padded persistent-state envelope |

Candidate promotion is disabled until both routes run on the same NPU inputs
and Main independently reproduces the selected route at the same topology.
The independent reference process contract is emitted by
`decode_reference.py`; it records vLLM-Ascend paths and mapped libraries
without importing that stack in the PyPTO process.

## Current implementation and gates

The current PyPTO code includes the bounded L0 SWA ABI, a 1/2/8-step state
replay, an independent EP2 routed/shared MoE product driver, and a complete
L0 composition entry. The EP2 driver keeps all 384 routed experts, owns 192
experts per rank, and compiles its dispatch/combine windows with PyPTO. The
real two-card MoE boundary has executed with the checkpoint and the
independent vLLM-Ascend capture input; route IDs, route weights, logits, and
the merged MoE output pass the fixed floating gate. The evidence is
`.pypto-a3-state/product/decode-ep2-l0-v25e.json`.

The composition entry has now consumed a real layer input, historical cache,
request state, and cache write boundary in a queued two-card A3 run. The
representative L0 result compares the complete PyPTO output, router input,
logits, routes, MoE output, FFN state, and the populated cache pages against
the same vLLM-Ascend boundary. The latest joint reference contract is v29 and
the product evidence is `.pypto-a3-state/product/decode-layer-ep2-l0-v29.json`.

The cache replay path was revalidated on A3 after correcting the physical
cache prefix read to the product's two-block ABI and binding runtime assembly
to the writable pinned PTO-ISA checkout. The v30 fixture and report
`.pypto-a3-state/product/decode-layer-ep2-l0-v30.json` pass the same cache,
FFN state, route, MoE and output comparisons on two ranks. This is still a
single-request L0 representative case.

The formal EP2 gate remains `not_run`: this run is one active request and does
not yet cover the required no-local-input collective, finished/inactive rows,
unequal-length batch, or slot-reuse cases. Complete C1A/C2A
index/compression, Engram, CP/SP and the requested 16-device gates remain
separate blockers.

The latest queued two-card component replays extend the independent
vLLM-Ascend comparison to two request schedules for each of L1, L2, L3, L14,
L15, L19, L20, L21, L24 and L25. Each report compares PyPTO logits, exact
top-six route IDs, route weights and the merged routed/shared output at
`rtol=0.05`, `atol=0.05`:

| layers | active batch | unequal lengths | scope |
| --- | --- | --- | --- |
| L1/L2/L3 | `decode-ep2-l{1,2,3}-active-batch-v1.json` | `decode-ep2-l{1,2,3}-unequal-v1.json` | MoE boundary only |
| L14/L15 | `decode-ep2-l{14,15}-active-batch-v1.json` | `decode-ep2-l{14,15}-unequal-v1.json` | MoE boundary only |
| L19/L20/L21 | `decode-ep2-l{19,20,21}-active-batch-v1.json` | `decode-ep2-l{19,20,21}-unequal-v1.json` | MoE boundary only |
| L24/L25 | `decode-ep2-l{24,25}-active-batch-v1.json` | `decode-ep2-l{24,25}-unequal-v1.json` | MoE boundary only |

These 20 reports have `status=pass` and `formal_accuracy=not_run` by design;
their cache/state field is `not_applicable_at_moe_boundary`. They do not
establish a complete physical layer result. The complete-layer runner now has
layer-specific Engram, C2A, C1A, index/cache, mHC and MoE compositions for the
ten target layers. The current paired complete-layer evidence is: L1, L2, L3,
L14, L19, L20, L21, L24 and L25 pass; L15 reaches the complete replay but
remains a precision failure at the fixed gate because its observable
`moe_output` does not match the fused vLLM boundary.

The supplied Aurora W8A8 Dynamic export is registered as an external
checkpoint asset. Its index references 270 quantized shard files using the
`of-00272` naming convention, and four separate Engram safetensors are present
for L1/L14. `decode_checkpoint_manifest.py` records metadata, shard presence,
target-layer mapping, and safetensors headers without hashing tensor payloads.

The independent layer setup is recorded by
`tools/deepseek_v41_layer_contract.py` and
`tools/deepseek_v41_layer_stage_inventory.py`. The generated reports under
`.pypto-a3-state/decode-assets/` distinguish representative product evidence
from formal accuracy, and keep all reference, NPU, and accuracy fields at
`not_run` until the matching layer has its own capture and PyPTO replay.
Capacity-16 stage construction is setup evidence only; it is not a layer
precision result.

The no-device per-layer preflight is recorded by
`tools/deepseek_v41_layer_preflight.py`. It creates one plan directory for each
of L0, L1, L2, L3, L14, L15, L19, L20, L21, L24 and L25 under
`.pypto-a3-state/decode-assets/layers/`. Each plan fixes the layer mode,
compression/cache source, weight shards, Engram assets, EP2 topology, input
state boundary, reference command, product command status and recovery rule.
It never creates placeholder token/cache/capture files. The product runner
exposes all retained EP2 layer compositions; reuse and reindex commands remain
blocked until source state and target request metadata match exactly.
The verifier also rejects generic layer/MoE captures for non-SWA layers: C2A
and C1A require semantic compressed-KV, index-cache/scale and (for C1A)
candidate-pool state, while L1/L14 require captured Engram rows.

| gate | status | recovery condition |
| --- | --- | --- |
| product import/dependency scan | `pass` | Keep vLLM/Torch-NPU out of the product closure |
| Aurora checkpoint metadata/header audit | `pass` | Keep the external manifest and set `DEEPSEEK_V41_CHECKPOINT` for jobs |
| bounded L0 PyPTO compile/ABI | `bounded_only` | Does not establish checkpoint accuracy |
| EP2 PyPTO MoE code generation | `pass` | `decode_moe_compile`; compiler artifact has physical NZ lowering |
| complete L0 EP2 PyPTO composition code generation | `pass` | `decode_layer_l0_compile`; mHC/attention/cache/MoE composition compiles |
| DP2/TP1/EP2 MoE boundary Decode | `pass` | `.pypto-a3-state/product/decode-ep2-l0-v25e.json`; real two-card A3 route/output gate passed |
| DP2/TP1/EP2 complete physical layer representative Decode | `pass` | `task_20261106_042451_245850019523`; `.pypto-a3-state/product/decode-layer-ep2-l0-v29.json`; output, router input, routes, MoE, FFN state, and cache pages compared on A3 |
| L1 EP2 PyPTO/vLLM-Ascend MoE boundary replay | `pass` | `task_20261106_182455_412234226294`; `.pypto-a3-state/product/decode-ep2-l1-moe-v2.json`; logits, exact route IDs, route weights, and output compared on devices 2,3 |
| L2 EP2 PyPTO/vLLM-Ascend MoE boundary replay | `pass` | `task_20261106_182455_412234319554`; `.pypto-a3-state/product/decode-ep2-l2-moe-v2.json`; logits, exact route IDs, route weights, and output compared on devices 4,5 |
| L3 EP2 PyPTO/vLLM-Ascend MoE boundary replay | `pass` | `task_20261106_182455_412234130828`; `.pypto-a3-state/product/decode-ep2-l3-moe-v2.json`; logits, exact route IDs, route weights, and output compared on devices 0,1 |
| L14 EP2 PyPTO/vLLM-Ascend MoE boundary replay | `pass` | `task_20261107_005915_175293029255`; `.pypto-a3-state/product/decode-ep2-l14-moe-v2.json`; logits, exact route IDs, route weights, and output compared on devices 6,7 |
| L15 EP2 PyPTO/vLLM-Ascend MoE boundary replay | `pass` | `task_20261107_005915_175292910575`; `.pypto-a3-state/product/decode-ep2-l15-moe-v2.json`; logits, exact route IDs, route weights, and output compared on devices 4,5 |
| L19 EP2 PyPTO/vLLM-Ascend MoE boundary replay | `pass` | `task_20261107_010231_177414730073`; `.pypto-a3-state/product/decode-ep2-l19-moe-v1.json`; logits, exact route IDs, route weights, and output compared on devices 4,5 |
| L20 EP2 PyPTO/vLLM-Ascend MoE boundary replay | `pass` | `task_20261107_023110_224905226856`; `.pypto-a3-state/product/decode-ep2-l20-moe-v1.json`; logits, exact route IDs, route weights, and output compared on devices 0,1 |
| L21 EP2 PyPTO/vLLM-Ascend MoE boundary replay | `pass` | `task_20261107_023110_224905330745`; `.pypto-a3-state/product/decode-ep2-l21-moe-v1.json`; logits, exact route IDs, route weights, and output compared on devices 2,3 |
| L24 EP2 PyPTO/vLLM-Ascend MoE boundary replay | `pass` | `task_20261107_023325_226337011669`; `.pypto-a3-state/product/decode-ep2-l24-moe-v1.json`; logits, exact route IDs, route weights, and output compared on devices 2,3 |
| L25 EP2 PyPTO/vLLM-Ascend MoE boundary replay | `pass` | `task_20261107_023325_22633697697`; `.pypto-a3-state/product/decode-ep2-l25-moe-v1.json`; logits, exact route IDs, route weights, and output compared on devices 0,1 |
| L1 complete physical Decode | `pass` | `.pypto-a3-state/product/decode-layer-ep2-l1-current.json`; Engram/SWA, mHC, MoE and output boundaries pass on the paired A3 replay |
| L2 complete physical Decode | `pass` | `.pypto-a3-state/product/decode-layer-ep2-l2-v26.json`; C2A cache/index, mHC, MoE and output boundaries pass on the paired A3 replay |
| L3 complete physical Decode | `pass` | `.pypto-a3-state/product/decode-layer-ep2-l3-final.json`; current C2A consumer, cache/index, mHC, MoE and output boundaries pass on the paired A3 replay |
| L14 complete physical Decode | `pass` | `.pypto-a3-state/product/decode-layer-ep2-l14-v4.json`; Engram/C2A, mHC, MoE and output boundaries pass on the paired A3 replay |
| L15 complete physical Decode | `fail` | `.pypto-a3-state/product/decode-layer-ep2-l15-final.json`; attention/cache/index/HC/router/routes/output pass, while observable `moe_output` remains outside `atol=0.05, rtol=0.05` (`max_abs=0.132812`, `mean_abs=0.0017201`) because the reference uses a fused routed/shared boundary |
| L19 complete physical Decode | `pass` | `.pypto-a3-state/product/decode-layer-ep2-l19-v2.json`; C2A reuse, mHC, MoE and output boundaries pass on the paired A3 replay |
| L20 complete physical Decode | `pass` | `.pypto-a3-state/product/decode-layer-ep2-l20-final.json`; current C1A source, cache/index, mHC, MoE and output boundaries pass on the paired A3 replay |
| L21 complete physical Decode | `pass` | `.pypto-a3-state/product/decode-layer-ep2-l21-v35.json`; C1A reuse with source candidate/cache state passes on the paired A3 replay |
| L24 complete physical Decode | `pass` | `.pypto-a3-state/product/decode-layer-ep2-l24-v35.json`; C1A reindex/cache, mHC, MoE and output boundaries pass on the paired A3 replay |
| L25 complete physical Decode | `pass` | `.pypto-a3-state/product/decode-layer-ep2-l25-final.json`; source candidate/cache, attention, HC, MoE and final output pass on the paired A3 replay |
| L14/L15 vLLM-Ascend reference capture | `not_run` | Layer-root construction reaches 60.43 GiB per rank before the next 2.11 GiB W2 allocation; CPU offload 16/32/96 GiB does not apply until after FusedMoEFactory construction |
| DP2/TP1/EP2 formal Decode gate | `not_run` | Add no-local-input, finished/inactive, unequal-length, slot-reuse and additional request-state cases |
| DP8/TP2/EP16 complete single layer | `not_run` | Allocate 16 queued cards and run the complete composition |
| DP16/TP1/EP16 compatibility | `not_run` | Same complete layer at TP1 |
| DP4/TP4/EP16 compatibility | `not_run` | Same complete layer at TP4 |
| independent vLLM-Ascend MoE capture | `pass` | `.pypto-a3-state/reference/decode_ep2_l0_20261106_v25`; reference-only vLLM-Ascend process captured the same MoE input, routes and output |
| independent vLLM-Ascend full-state capture | `pass` | `.pypto-a3-state/reference/decode_ep2_l0_20261106_v29/reference_capture.json`; v29 joins layer, MoE, and cache state; v24/v26 remain historical split captures |
| same-input NPU PyPTO/reference representative comparison | `pass` | `.pypto-a3-state/product/decode-layer-ep2-l0-v29.json`; fixed `rtol=0.05`, `atol=0.05` and exact route IDs |
| target layers 0,1,2,3,14,15,19,20,21,24,25 | `not_run` | Each layer must pass its own complete DP8/TP2/EP16 gate |
| independent L1 reference capture | `not_run` | Earlier tasks failed before a complete capture: `task_20261106_113416_60083921106` could not import CANN `acl`, `task_20261106_142835_270996310588` expanded an unset `TASK_ID`, and `task_20261106_144922_29541712834` launched one process (`WORLD_SIZE=1`). The refreshed plan uses `torch.distributed.run --standalone --nproc_per_node=2`, absolute `vllm-python.sh`, cleared inherited paths, and `${TASK_ID:-not_recorded}`. v6 has rank-local arrays, but no assembled top-level contract yet. |
| independent layer contract matrix | `pass` | `.pypto-a3-state/decode-assets/layer-contracts.json`; metadata/setup only, formal status remains `not_run` |
| capacity-16 stage inventory | `pass` | `.pypto-a3-state/decode-assets/layer-stage-inventory.json`; stage construction only, no NPU or accuracy claim |
| no-device per-layer preflight | `pass` | `.pypto-a3-state/decode-assets/layer-preflight.json` and `layers/l*/plan.json`; 11 isolated plans, no synthetic input/cache assets |
| per-layer capture/replay execution plan | `pass` | `.pypto-a3-state/decode-assets/layer-contracts.json`; fixed EP2 vLLM-Ascend commands and product runner paths are recorded for all 11 layers; source/consumer pairs remain fail-closed until request metadata matches |

No status in this table may be promoted from the historical Prefill evidence,
CPU golden, simulator replay, TP1 diagnosis, single expert, or native
reference-only run.

## Executed checks

The following checks were run in the pinned shared environment on 2026-11-05:

| check | result | evidence |
| --- | --- | --- |
| fixed A3 environment activation | `pass` | PyPTO `72a62d13efabbb25d240e35944f68ba5bcc22b7f`, Python 3.10.22, `SOC_VERSION=ascend910_9382`, workspace state directory |
| queued EP2 A3 device preflight | `pass` | `task_20261105_101415_246219319021`; `TASK_DEVICE=0,1`, `ASCEND_RT_VISIBLE_DEVICES=0,1`, Ascend910 health OK; no model compute |
| product import/dependency scan | `pass` | `python -m models.deepseek_v4_1_flash_a2.decode_acceptance --static-scan` |
| PyPTO single-step compile | `pass` | `decode_fwd -p a2a3 --sim --compile-only` |
| PyPTO replay compile | `pass` | `decode_replay --steps 1/2/8 -p a2a3 --sim --compile-only` |
| EP2 PyPTO MoE code generation | `pass` | `decode_moe_compile`; `.pypto-a3-state/decode-ep2-compile.json`; A3 artifact generated with 384 routed experts and 192 local experts per rank |
| real Aurora EP2 weight ABI probe | `pass` | `decode_weight_abi.py`; one real layer-0 expert, router, and shared expert packed in the vLLM environment; NZ round-trip and W2 zero-tail checks passed |
| Decode asset structure verifier | `pass` | `decode_asset_contract.py`; current report rejects the absent input, history-cache, and reference-capture files before queue submission |
| Aurora checkpoint manifest | `pass` | `.pypto-a3-state/deepseek-v41-checkpoint-manifest.json`; metadata/header audit only |
| simulator numerical replay | `not_run` | CPU/simulator precision is outside this session; no NPU result inferred |
| independent reference manifest | `pass` | `.pypto-a3-state/reference/decode_ep2_l0_20261106_v29/reference_capture.json`; vLLM-Ascend paths, native hashes, input, layer, MoE, and cache capture are recorded |
| queued two-card PyPTO EP2 MoE run | `pass` | `task_20261106_024728_200089531660`; A3 `TASK_DEVICE=0,1`, routes and output pass in `.pypto-a3-state/product/decode-ep2-l0-v25e.json` |
| EP2 complete physical-layer representative run | `pass` | `task_20261106_042451_245850019523`; `.pypto-a3-state/product/decode-layer-ep2-l0-v29.json` |
| L0 cache ABI and pinned PTO-ISA revalidation | `pass` | `task_20261107_211521_159540026017`; `.pypto-a3-state/product/decode-layer-ep2-l0-v30.json`; cache `[2,2,128,1,512]`, state, routes and output passed on A3 |
| queued L14/L15/L19 PyPTO EP2 MoE boundary runs | `pass` | `task_20261107_005915_175293029255`, `task_20261107_005915_175292910575`, `task_20261107_010231_177414730073`; user-state PyPTO runtime, devices 6,7 / 4,5 / 4,5; each report has logits, exact routes, route weights and output PASS |
| queued L20/L21/L24/L25 PyPTO EP2 MoE boundary runs | `pass` | `task_20261107_023110_224905226856`, `task_20261107_023110_224905330745`, `task_20261107_023325_226337011669`, `task_20261107_023325_22633697697`; user-state PyPTO runtime; each report has logits, exact routes, route weights and output PASS |
| initial L14/L15 retry with shared PyPTO path | `blocked_environment` | `task_20261107_005442_173030223340`, `task_20261107_005442_173030330312`; `PYTHONPATH=$PWD` selected the read-only shared `simpler_setup` and failed on `.pto-isa.lock`; reruns preserved `$PYPTO_A3_STATE_DIR/python` and passed |
| EP2 formal multi-case gate | `not_run` | The representative case lacks no-local-input, finished/inactive, unequal-length, and slot-reuse coverage |
| DP8/TP2/EP16 gate | `not_run` | `decode_gate.py --topology main`; complete layer input/cache/reference assets are missing |

The Aurora loader was also exercised against the supplied checkpoint without
materializing the full model. Layer 0, 1 and 14 index coverage was verified,
with samples from routed experts 0/191/192/383 and the shared expert. The
observed routed W1/W3 layout is INT8 `[2304, 5120]`, W2 is INT8
`[5120, 2304]`, scales are FP32 `[out, 1]`, and sampled offsets are zero.
This is weight-layout evidence only; it does not supply the required decode
token, history cache, request state, or reference output.

`decode_weight_abi.py` is a host preprocessing boundary and is not imported
by the PyPTO product closure. It reads the real Aurora tensors lazily, rejects
non-zero affine offsets, packs routed W1/W3/W2 to FRACTAL_NZ, pads only the
W2 reduction tail from 2304 to 2560, and transposes/concatenates the shared
expert operands for the logical `moe_shared` ABI. The probe covered one real
expert and does not constitute an EP2 NPU or accuracy gate.

The gate entry points are:

```bash
PYTHONPATH=$PWD python -m models.deepseek_v4_1_flash_a2.decode_gate \
  --topology ep2_gate --record .pypto-a3-state/decode-ep2-gate.json
PYTHONPATH=$PWD python -m models.deepseek_v4_1_flash_a2.decode_gate \
  --topology main --record .pypto-a3-state/decode-main-gate.json
```

They only emit the public queue command and a missing-asset report. They do not
load or launch vLLM-Ascend and do not claim NPU accuracy.

For the supplied checkpoint, set `DEEPSEEK_V41_CHECKPOINT` to
`/mnt/data/models/DeepSeek-V4.1-Flash-w8a8` and point
`DEEPSEEK_V41_DECODE_ASSETS` at the private Decode asset directory. The latest
queued no-device EP2 report records checkpoint `metadata_verified`; the v29
joint runtime assets are now present under the private reference directory.

The reference candidate environment exists at a nobody-owned shared path that
is now readable through the supplied shared release. The independent capture
uses its Python with vLLM `0.31.0+empty`, vLLM-Ascend
`0.19.1rc2.dev2832+g23d1d9055`, the pinned source/native identities, and
`torch_npu` only in the reference process. The v29 capture records the same
layer-0 input, route tensors, layer output, and cache before/after state in one
joint contract consumed by the PyPTO run. v24/v26 remain historical split
captures and are not used for this comparison. No vLLM serve process was
started. The bundled extra
`npu_rms_norm_cast` check remains a diagnostic failure because the current
`libopapi.so` lacks `aclnnRmsNormCast`.

## Environment and queue

Use the supplied activation script with a writable session state directory:

```bash
export PYPTO_A3_STATE_DIR="$PWD/.pypto-a3-state"
source /mnt/data/pyptouser/liufei-shared-envs/pypto-a3-72a62d13-20261031/activate-shared-pypto.sh
```

The pinned PyPTO revision is
`72a62d13efabbb25d240e35944f68ba5bcc22b7f`, platform `a2a3`,
`SOC_VERSION=ascend910_9382`. Every NPU command must use the public queue:

```bash
task-submit --device auto --device-num 2 --run '... EP2 command ...'
task-submit --device auto --device-num 16 --run '... main Decode command ...'
```

Use `$TASK_DEVICE` inside the job and record `ASCEND_RT_VISIBLE_DEVICES`, the
physical-to-logical rank map, CANN/PyPTO runtime paths, loaded `.so` files,
source hashes, and child-process status. Direct device probing, persistent
holders, device reset, and single-card substitutes are prohibited.

## Recovery conditions and non-goals

The representative full physical-layer task is recorded as `pass` for the
single active request case. The formal EP2 task remains `not_run` until the
additional request-state and collective cases are run and the joint
vLLM-Ascend layer/MoE/cache capture is recorded. This session does not run
Prefill, a continuous full network, MTP, DSpark, datasets, HTTP serving, or
`vllm serve`; none of those statuses are inferred here.
