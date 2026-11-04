# DeepSeek V4.1 Flash Decode acceptance task

This file is the independent A3 Decode session record. It starts at
`not_run`; no Prefill goal, session, verifier, or passing state is reused.
CPU precision is outside this acceptance. Formal numerical comparisons are
NPU PyPTO versus an independently launched vLLM-Ascend reference process.

## Baseline and provenance

The checkout is detached at `addb655f7b4f4a4ce56af875724a04d79d3fe325`.
The requested linear history is present and has not been squashed or amended:

| order | commit | author | author time |
| --- | --- | --- | --- |
| 1 | `500858a9abd6ce1d3796a5e26a4bb597cdf622e4` | Adelaide999 &lt;73272686+Adelaide999@users.noreply.github.com&gt; | `2026-10-31T10:13:30+08:00` |
| 2 | `d401f4e82e6348aee1566118c0c768b66e70504d` | sunhaolun &lt;sunhaolun3@huawei.com&gt; | `2026-10-31T17:51:06+08:00` |
| 3 | `c0f23d56800fbd34d4b3442d5df9565be387a955` | Lun7777 &lt;1055318345@qq.com&gt; | `2026-11-02T10:00:49+08:00` |
| 4 | `addb655f7b4f4a4ce56af875724a04d79d3fe325` | Lun7777 &lt;1055318345@qq.com&gt; | `2026-11-02T14:36:58+08:00` |

`eed3280972f6743fc46d68219bcb108596e64578` is the common base. The supplied
source checkout and `cherry-pick-receipt.json` were not available in this
workspace, so the commit graph is the receipt evidence and the missing paths
remain a recovery condition.

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

## Current implementation and gates

The current PyPTO code is a bounded L0 SWA ABI and a 1/2/8-step state replay.
It does not yet compose C1A/C2A index/compression, mHC/residual, Engram,
shared/routed MoE, distributed dispatch/combine, CP/SP, or a full checkpoint
driver. The kernel accepts position/request/source/finished controls but does
not consume those controls in its device schedule and advances only
`cache_lens[0]`; these are explicit formal blockers.

| gate | status | recovery condition |
| --- | --- | --- |
| product import/dependency scan | `pass` | Keep vLLM/Torch-NPU out of the product closure |
| bounded L0 PyPTO compile/ABI | `bounded_only` | Does not establish checkpoint accuracy |
| DP2/TP1/EP2 full MoE Decode | `not_run` | Implement 384-expert top-6 dispatch/combine and allocate two queued cards |
| DP8/TP2/EP16 complete single layer | `not_run` | Implement the full layer and allocate 16 queued cards |
| DP16/TP1/EP16 compatibility | `not_run` | Same complete layer at TP1 |
| DP4/TP4/EP16 compatibility | `not_run` | Same complete layer at TP4 |
| independent vLLM-Ascend capture | `not_run` | Supply matching checkpoint, reference environment and capture manifest |
| same-input NPU PyPTO/reference comparison | `not_run` | Supply all contract inputs and state outputs |
| target layers 0,1,2,3,14,15,19,20,21,24,25 | `not_run` | Each layer must pass its own complete DP8/TP2/EP16 gate |

No status in this table may be promoted from the historical Prefill evidence,
CPU golden, simulator replay, TP1 diagnosis, single expert, or native
reference-only run.

## Executed checks

The following checks were run in the pinned shared environment on 2026-11-04:

| check | result | evidence |
| --- | --- | --- |
| product import/dependency scan | `pass` | `python -m models.deepseek_v4_1_flash_a2.decode_acceptance --static-scan` |
| PyPTO single-step compile | `pass` | `decode_fwd -p a2a3 --sim --compile-only` |
| PyPTO replay compile | `pass` | `decode_replay --steps 1/2/8 -p a2a3 --sim --compile-only` |
| simulator state replay | `blocked` | runtime assemble could not create the shared environment PTO-ISA lock (`Errno 30`, read-only path); no numerical status inferred |
| EP2 MoE gate | `not_run` | `decode_gate.py --topology ep2_gate`; checkpoint/cache/reference assets missing |
| DP8/TP2/EP16 gate | `not_run` | `decode_gate.py --topology main`; checkpoint/cache/reference assets missing |

The gate entry points are:

```bash
PYTHONPATH=$PWD python -m models.deepseek_v4_1_flash_a2.decode_gate \
  --topology ep2_gate --record .pypto-a3-state/decode-ep2-gate.json
PYTHONPATH=$PWD python -m models.deepseek_v4_1_flash_a2.decode_gate \
  --topology main --record .pypto-a3-state/decode-main-gate.json
```

They only emit the public queue command and a missing-asset report. They do not
load or launch vLLM-Ascend and do not claim NPU accuracy.

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

The formal task is blocked at `not_run` until the checkpoint/config conversion,
historical cache and block-table assets, same-input vLLM-Ascend reference
capture, distributed driver, and per-rank memory/communication reports are
available. This session does not run Prefill, a continuous full network, MTP,
DSpark, datasets, HTTP serving, or `vllm serve`; none of those statuses are
inferred here.
