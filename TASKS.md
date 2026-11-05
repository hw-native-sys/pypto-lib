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
replay, and an independent EP2 routed/shared MoE product driver. The EP2
driver keeps all 384 routed experts, owns 192 experts per rank, and compiles
its dispatch/combine windows with PyPTO. It has not yet consumed the real
checkpoint, historical cache, request state, or an independent reference
capture; complete C1A/C2A index/compression, mHC/residual, Engram, CP/SP and
full-layer composition remain formal blockers.

The bounded SWA entry and the EP2 MoE entry are intentionally separate
operators. No current entry claims a complete physical layer that composes
attention/cache, mHC residual streams, Engram, and routed/shared MoE; that
composition must be added and independently gated before DP8/TP2/EP16 can
advance.

The supplied Aurora W8A8 Dynamic export is registered as an external
checkpoint asset. Its index references 270 quantized shard files using the
`of-00272` naming convention, and four separate Engram safetensors are present
for L1/L14. `decode_checkpoint_manifest.py` records metadata, shard presence,
target-layer mapping, and safetensors headers without hashing tensor payloads.

| gate | status | recovery condition |
| --- | --- | --- |
| product import/dependency scan | `pass` | Keep vLLM/Torch-NPU out of the product closure |
| Aurora checkpoint metadata/header audit | `pass` | Keep the external manifest and set `DEEPSEEK_V41_CHECKPOINT` for jobs |
| bounded L0 PyPTO compile/ABI | `bounded_only` | Does not establish checkpoint accuracy |
| EP2 PyPTO MoE code generation | `pass` | `decode_ep2_compile`; compiler artifact has physical NZ lowering, execution remains not_run |
| DP2/TP1/EP2 full MoE Decode | `not_run` | Submit the real checkpoint/input/cache through the two-card queue and compare to the same-input reference |
| DP8/TP2/EP16 complete single layer | `not_run` | Implement the full layer and allocate 16 queued cards |
| DP16/TP1/EP16 compatibility | `not_run` | Same complete layer at TP1 |
| DP4/TP4/EP16 compatibility | `not_run` | Same complete layer at TP4 |
| independent vLLM-Ascend capture | `not_run` | Supply matching Decode input/cache, reference environment and capture manifest |
| same-input NPU PyPTO/reference comparison | `not_run` | Supply all contract inputs and state outputs |
| target layers 0,1,2,3,14,15,19,20,21,24,25 | `not_run` | Each layer must pass its own complete DP8/TP2/EP16 gate |

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
| EP2 PyPTO MoE code generation | `pass` | `decode_ep2_compile`; `.pypto-a3-state/decode-ep2-compile.json`; A3 artifact generated with 384 routed experts and 192 local experts per rank |
| real Aurora EP2 weight ABI probe | `pass` | `decode_weight_abi.py`; one real layer-0 expert, router, and shared expert packed in the vLLM environment; NZ round-trip and W2 zero-tail checks passed |
| Decode asset structure verifier | `pass` | `decode_asset_contract.py`; current report rejects the absent input, history-cache, and reference-capture files before queue submission |
| Aurora checkpoint manifest | `pass` | `.pypto-a3-state/deepseek-v41-checkpoint-manifest.json`; metadata/header audit only |
| simulator numerical replay | `not_run` | CPU/simulator precision is outside this session; no NPU result inferred |
| independent reference manifest | `not_run` | vLLM-Ascend path, Decode input/cache and capture are not configured |
| EP2 MoE gate | `not_run` | `decode_gate.py --topology ep2_gate`; checkpoint metadata is verified, but `decode_input.json`, `history_cache.npz`, and `reference_capture.json` are missing |
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
queued no-device EP2 report records checkpoint `metadata_verified`, with the
three runtime assets above still absent.

The reference candidate environment exists at a nobody-owned shared path that
is now readable through the supplied shared release. The independent probe
`tools/deepseek_v41_vllm_reference.py` was run with the release's Python and
recorded at `.pypto-a3-state/vllm-reference-config.json` and
`.pypto-a3-state/vllm-reference-npu-probe.json`. It confirmed vLLM
`0.31.0+empty`, vLLM-Ascend
`0.19.1rc2.dev2832+g23d1d9055`, the pinned source/native identities, and
`torch_npu.npu_rms_norm` on queued device 4. This is runtime/config evidence
only: no weights were loaded, no vLLM serve process was started, and no
same-input Decode output or cache capture exists yet. The bundled extra
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

The formal task is blocked at `not_run` until the Decode input, historical
cache and block-table assets, same-input vLLM-Ascend reference capture, a
queued NPU execution, and per-rank memory/communication reports are available.
The checkpoint metadata is present, but it does not supply those runtime
states. This session does not run Prefill, a continuous full network, MTP,
DSpark, datasets, HTTP serving, or `vllm serve`; none of those statuses are
inferred here.
