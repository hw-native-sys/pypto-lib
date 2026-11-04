# DeepSeek V4.1 Flash on A2

`models/deepseek_v4_1_flash_a2/` packages the A2 single-layer Prefill
operators separately from the original TP4/DP2/EP8 model scaffold. It includes
W8A8 and BF16 projections, SWA and compressed-cache stages, mHC, Engram,
384-expert top-six MoE, and the native operator boundaries used during
independent layer alignment.

The file-level ownership and migration map against the A5 implementation is
maintained in [`MODULE_ALIGNMENT.md`](../../../models/deepseek_v4_1_flash_a2/MODULE_ALIGNMENT.md).
Use it when adding a Decode or Prefill stage: it records the corresponding A5
semantic entry, the A2 owner, the dtype translation, and whether the stage is
currently partial or accepted.

## Development priorities

| Field | Current priority |
| --- | --- |
| Performance target | Correct independent single-layer Prefill and A2/A3 decoder paths with the original full-width operands and numerical boundaries. |
| Deferred performance work | Whole-network execution, serving, multi-card execution, and performance tuning. Decoder operators are included for independent layer validation; whole-network serving remains outside this directory. |
| Compatibility requirements | Preserve the frozen operators' dtypes, quantization boundaries, page layouts, request controls, source ownership, readonly inputs, inactive rows, and untouched cache regions. |

## Construction and ownership

The directory exposes operators and bounded stage factories. Callers supply
actual checkpoint operands, scheduler controls, caches, and output buffers.
It does not contain a checkpoint downloader, a full-block replay driver, or
the private orchestration used by any earlier work; those results are not part
of this Decode session.

The supplied DeepSeek V4.1 Flash W8A8 Dynamic Aurora export is kept outside
the repository and selected with `DEEPSEEK_V41_CHECKPOINT`. Run
`decode_checkpoint_manifest.py` to verify its JSON metadata, indexed shard
presence, target-layer-to-shard mapping, Engram files, and selected
safetensors headers. The manifest does not hash or copy tensor payloads.

| Files | Responsibility |
| --- | --- |
| `ops.py`, `qkv_w8a8.py`, `ecotech_linear*.py` | Per-token INT8 quantization, KN weight packing, full INT32 accumulation, explicit FP32 scaling and BF16 narrowing. |
| `ecotech_swa_prefill.py`, `swa.py`, `grouped_output.py` | Full 64-head SWA, inverse partial RoPE, eight grouped Wo-A products and BF16 Wo-B. The raw operand adapter in `ecotech_swa_prefill.py` uses layer-zero checkpoint key names. |
| `layer2_c1c2.py`, `c2_prefill_pool.py`, `ecotech_full2_*.py` | Ratio-two projection, ring state, pooling, cache publication and index observation. The width-128 index publisher includes the accepted 16-row quantization and reciprocal correction. |
| `ratio1_prefill.py` | Ratio-one publication starting at an actual BF16 compressor projection. This is a distinct input boundary from the FP32 ratio-two compressor. |
| `ecotech_prefill_consumer.py`, `indexer.py`, `sparse_consumer_prefill.py` | Consumer projections, causal controls and sparse attention stages. |
| `decode_c2a_full.py`, `decode_c2a_reuse.py`, `decode_c1a_full.py`, `decode_c1a_reuse.py`, `decode_c1a_reindex.py` | A2/A3 stage owners for L14/L15/L19/L20/L21/L24/L25. `layer14_15.py` and `layers19_25.py` remain aggregate compatibility providers. |
| `decode_fwd.py`, `decode_replay.py` | Bounded PyPTO L0 SWA Decode ABI, single-step entry, and fixed 1/2/8-step state replay. These entries do not claim checkpoint accuracy, full-model, or distributed Decode coverage. |
| `decode_candidates.py`, `decode_acceptance.py`, `decode_gate.py` | Search K=2 candidate registry, independent Decode contract, product import closure scan, runtime identity capture, and fail-closed layer/topology gate reporting. |
| `decode_checkpoint_manifest.py`, `decode_weights.py` | Read-only Aurora metadata/header audit and lazy tensor resolver; neither performs product computation. |
| `candidate_pool.py`, `raw_checkpoint.py` | A2/A3 candidate metadata and raw checkpoint loading. |
| `swa_scheduler.py`, `smla_scheduler.py` | Integer-only request, page and causal-selection scheduling. Original and compressed lanes retain separate page tables. |
| `swa_native.py`, `smla_native.py`, `smla_staging.py` | Registered BF16 SWA/ratio-one SMLA leaves, fresh metadata, and PyPTO dtype boundaries. |
| `engram_ecotech*.py`, `engram_hash.py`, `engram_offload.py` | INT8 row lookup, BF16 projection, original-basis gating with probability observation, and host row preparation. |
| `hc.py`, `ecotech_mhc_*.py`, `moe_*.py` | Four-stream mHC, complete shared expert, routing, native dispatch, routed expert and route combination. |
| `*_runtime.py`, `bf16_linear_native.py` | Actual-operand native boundaries and input/output/library identity checks. |

Construct the principal stages from the repository root in a Python 3.10+
environment containing Torch and the selected PyPTO checkout:

```python
from models.deepseek_v4_1_flash_a2.ecotech_swa_prefill import GEOMETRY, make_builders
from models.deepseek_v4_1_flash_a2.ecotech_full2_prefill import make_entries
from models.deepseek_v4_1_flash_a2.ratio1_prefill import make_ratio1_compress_publish

swa = make_builders(capacity=16)
full2 = make_entries(active=7)
full2_projected = make_entries(active=7, compressor_projected=True)
ratio1 = make_ratio1_compress_publish(16, dict(GEOMETRY, index_head_dim=128))
```

These calls construct DSL entries without executing a device workload. The
Full2 observation entries retain the bounded active counts 1, 7 and 16.
`compressor_projected=True` consumes supplied FP32 projection buffers rather
than computing them in DSL. Preserve the selected projection boundary when
comparing with an external runtime.

Quantized checkpoint preprocessing remains on the host; forward numerical
operations execute in PyPTO or the explicitly named NPU operators. INT8
weights retain their channel scales, full reduction widths and INT32
accumulation. BF16 rounding and the order of scale multiplication are part
of the operator contract.

SWA uses BF16 `PA_BBND` pages with 128 rows. Ratio-two compressed/index
storage also uses lossless physical views of native 64-row pages; retain
the original slot mapping when using these views. Index payloads are INT8
with FP16 row scales; compressor state and projections are FP32. Consumers
must receive their declared source layer's cache and index state. Independent
layer acceptance does not validate chaining these buffers across layers.

## Native runtime dependencies

The aligned path includes native NPU dispatch. Install a compatible CANN,
Torch-NPU and vLLM-Ascend runtime exposing:

- `_C_ascend.npu_hc_pre_v3` for the HC-pre boundary;
- `_C_ascend.npu_quant_lightning_indexer_v2_metadata` and
  `_C_ascend.npu_quant_lightning_indexer_v3` for QLI;
- `_C_ascend.npu_sparse_flash_mla_metadata` and
  `_C_ascend.npu_sparse_flash_mla` for BF16 SWA and ratio-one SMLA;
- `torch_npu.npu_moe_init_routing_v2` for dispatch;
- `torch_npu.npu_rms_norm` and Ascend `aten.linear` dispatch for the explicit
  normalization and BF16/FP32 projection boundaries.

The BF16 linear registration preserves BF16 operands and dispatch; it does
not widen the projection to FP32. Keep its returned library handles alive.
The SWA and SMLA registrations retain their library handles internally and
must each be initialized once per interpreter. Their public namespaces are
`pypto_dsv41_swa_bf16_v1`, `pypto_dsv41_ratio1_smla_bf16_v1`, and
`pypto_dsv41_bf16_linear_v1`.

Fill in `runtime.example.json` with your installed runtime paths. Before
calling a subprocess host boundary, set `ASCEND_RT_VISIBLE_DEVICES` to the
single physical device specified in that configuration. The subprocess uses
logical device zero. Device scheduling belongs to the caller; these modules
do not reserve cards or start persistent holders.

HC additionally requires an installed OPP vendor and a manifest whose
`files` mapping pins these relative library names to SHA256 digests:

```text
op_api/lib/libcust_opapi.so
op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so
```

The HC boundary checks both the digests and the libraries actually mapped in
the process. QLI callers similarly supply the expected native library
identities. Runtime manifests and compiled libraries are external inputs;
they are not bundled in this model directory. A different native build
requires its own precision validation.

## Acceptance and source provenance

This directory is the independent A3 Decode session. No historical Prefill,
CPU, simulator, or private capture result is carried into its status. The
formal status for layers **0, 1, 2, 3, 14, 15, 19, 20, 21, 24 and 25** starts at
`not_run` and can only change after a same-input NPU PyPTO versus vLLM-Ascend
reference comparison at the requested topology. The floating policy is
`atol=0.05, rtol=0.05`; routes, lengths, masks, source IDs, readonly inputs,
inactive rows and untouched cache regions are exact checks.

`source_manifest.json` maps each ported file to its original source digest.
It also records unchanged definition AST digests, the native-call identities,
and host changes needed for relocation. Its `decode_files` records identify the
new PyPTO Decode entry points and explicitly keep their accuracy evidence
separate from source migration. Dependencies extracted from the original
parent model retain their compute definitions in this flat directory.
The port includes the L14 Engram observation interface, L1 native boundaries,
and the final L20 index-publication candidate.

The local tree contains bounded PyPTO L0 single-step and fixed-step SWA Decode
entries. They make no claim of validated C1A/C2A chaining, mHC/residual,
Engram residency, shared/routed MoE, distributed dispatch/combine, CP/SP,
stacked layers, whole-network score, multi-card execution, or serving support.
The independent vLLM-Ascend capture is described by
[`decode_reference.py`](../../../models/deepseek_v4_1_flash_a2/decode_reference.py)
and is not imported by the product path.

## Independent Decode acceptance

The Decode session starts with every formal gate at `not_run`. Search K=2 is
registered in [`decode_candidates.py`](../../../models/deepseek_v4_1_flash_a2/decode_candidates.py);
both candidates use PyPTO entry points and share the same page size, scale
contract, input boundary, topology and tolerance. The acceptance contract and
product dependency scan are executable without importing a model kernel:
The four-commit ancestry and tree IDs are recorded in
[`decode_cherry_pick_receipt.json`](../../../models/deepseek_v4_1_flash_a2/decode_cherry_pick_receipt.json).
The generated `git cherry-pick -x` proof is kept in a temporary clean clone;
the product checkout still retains the requested source commit IDs unchanged.
[`decode_shape_contract.py`](../../../models/deepseek_v4_1_flash_a2/decode_shape_contract.py)
records the global dimensions, the TP1/TP2/TP4 matrix, EP2 expert ownership,
the replicated single KV head, page layout, and dispatch/combine contract.
`runtime_config.py` records the constructed-runtime fields for sequence
parallel MoE, cache block size, request scheduling and communication groups;
unresolved fields stay `not_run`. Capacity and per-rank peak memory are also
left `not_run` until real checkpoint buffers are allocated.

```bash
PYTHONPATH=$PWD python -m models.deepseek_v4_1_flash_a2.decode_acceptance --json
PYTHONPATH=$PWD python -m models.deepseek_v4_1_flash_a2.decode_acceptance \
  --record .pypto-a3-state/decode-acceptance.json
PYTHONPATH=$PWD python -m models.deepseek_v4_1_flash_a2.decode_gate \
  --topology ep2_gate --record .pypto-a3-state/decode-ep2-gate.json
PYTHONPATH=$PWD python -m models.deepseek_v4_1_flash_a2.decode_gate \
  --topology main --record .pypto-a3-state/decode-main-gate.json
```

The scan rejects vLLM, vLLM-Ascend and Torch-NPU imports in the product import
closure. Files ending in `_runtime.py` are reference-only and cannot be loaded
by a product entry. The scan records Torch imports used by the golden/reference
side separately; they do not establish product accuracy.

`decode_gate.py` is intentionally fail-closed. It emits the required
`task-submit --device auto --device-num 2/16` command and records missing
checkpoint, cache, input, or reference assets; it never launches vLLM or
claims a gate without a same-input capture. `decode_reference.py` records the
separate reference process's Python path, source digest, mapped libraries,
visible devices and runtime library path; it never imports vLLM in the product
process.

The current bounded entries accept the full state ABI, but do not yet consume
per-request position/source/finished controls in the device schedule and only
advance the first length slot. Consequently the EP2 MoE gate, DP8/TP2/EP16
complete layer gates, cache/state comparison, vLLM-Ascend same-input capture,
and all eleven target-layer gates remain `not_run` until a checkpoint conversion,
distributed driver, and independent reference capture are available.

## Checks without a device

Compile the bounded Decode ABI without allocating a device:

```bash
PYTHONPATH=$PWD python -m models.deepseek_v4_1_flash_a2.decode_fwd -p a2a3 --sim --compile-only
```

Compile the deterministic single-context state replay for each supported step
count. These commands exercise the bounded PyPTO ABI only; they are not NPU
checkpoint accuracy evidence:

```bash
PYTHONPATH=$PWD python -m models.deepseek_v4_1_flash_a2.decode_replay --steps 1 -p a2a3 --sim
PYTHONPATH=$PWD python -m models.deepseek_v4_1_flash_a2.decode_replay --steps 2 -p a2a3 --sim
PYTHONPATH=$PWD python -m models.deepseek_v4_1_flash_a2.decode_replay --steps 8 -p a2a3 --sim
```

The external A5-shaped validation tree is not part of this Decode session and
must not be used as an acceptance shortcut. Actual operator validation should
use the existing [Golden Harness](../../run-and-validate/golden-harness.md)
only to compile the PyPTO entry, then compare NPU outputs and state against the
independent reference manifest under the same checkpoint, layer, cache and
topology. Native boundary reports verify identity and interface integrity;
they do not replace numerical comparison against an independent reference.

## Decoder operators

The A2/A3 decoder adaptation is integrated into this same flat package. The
decoder contract is shared through `config.py`, `operator_config.py`,
`engram.py`, and `sparse_attention.py`; W8A8 quantization and GEMM live in
`quant.py` and `gemm.py`. The decoder entry points use the A5 file names:
`decode_c2a_full.py`, `decode_c2a_reuse.py`, `decode_c1a_full.py`,
`decode_c1a_reuse.py`, and `decode_c1a_reindex.py`. The CPU references and
bounded PyPTO validators are direct modules in this directory, with no nested
`verifier` package. The original `layer14_15.py` and `layers19_25.py` imports
remain available during migration.

The old external validator suite is intentionally excluded from this session.
The retained A2/A3 ABI is BF16 hidden and cache rows, INT8 weights and index
payloads, FP32 W8A8 scales, FP16 index-cache scales, and INT32 GEMM
accumulation. Prefill callers continue to use the existing `prefill.py`,
`attention.py`, and cache modules unchanged; no Prefill result changes Decode
status.
