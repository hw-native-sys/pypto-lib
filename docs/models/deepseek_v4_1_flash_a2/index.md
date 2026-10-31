# DeepSeek V4.1 Flash on A2

`models/deepseek_v4_1_flash_a2/` packages the A2 single-layer Prefill
operators separately from the original TP4/DP2/EP8 model scaffold. It includes
W8A8 and BF16 projections, SWA and compressed-cache stages, mHC, Engram,
384-expert top-six MoE, and the native operator boundaries used during
independent layer alignment.

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
the private orchestration used to obtain the historical acceptance results.

| Files | Responsibility |
| --- | --- |
| `ops.py`, `qkv_w8a8.py`, `ecotech_linear*.py` | Per-token INT8 quantization, KN weight packing, full INT32 accumulation, explicit FP32 scaling and BF16 narrowing. |
| `ecotech_swa_prefill.py`, `swa.py`, `grouped_output.py` | Full 64-head SWA, inverse partial RoPE, eight grouped Wo-A products and BF16 Wo-B. The raw operand adapter in `ecotech_swa_prefill.py` uses layer-zero checkpoint key names. |
| `layer2_c1c2.py`, `c2_prefill_pool.py`, `ecotech_full2_*.py` | Ratio-two projection, ring state, pooling, cache publication and index observation. The width-128 index publisher includes the accepted 16-row quantization and reciprocal correction. |
| `ratio1_prefill.py` | Ratio-one publication starting at an actual BF16 compressor projection. This is a distinct input boundary from the FP32 ratio-two compressor. |
| `ecotech_prefill_consumer.py`, `indexer.py`, `sparse_consumer_prefill.py` | Consumer projections, causal controls and sparse attention stages. |
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

The historical frozen snapshots passed independent same-input complete-block
checks for layers **0, 1, 2, 3, 14, 15, 19, 20, 21, 24 and 25**. Layer zero
has six complete calls; each of the other selected layers has twelve.
`acceptance_manifest.json` records the original evidence digests and limits.
The most recent L20 gate uses six cases and twelve calls with all 384 experts,
top-six routing and 64 attention heads. Its floating policy is
`atol=0.05, rtol=0.05`, with zero violations and nonfinite values; discrete
controls, routes, readonly inputs, inactive rows and untouched regions are
checked exactly. Historical acceptance belongs to those frozen snapshots.

`source_manifest.json` maps each ported file to its original source digest.
It also records unchanged definition AST digests, the native-call identities,
and host changes needed for relocation. Dependencies extracted from the
original parent model retain their compute definitions in this flat directory.
The port includes the L14 Engram observation interface, L1 native boundaries,
and the final L20 index-publication candidate.

The relocated directory has host validation and factory-construction coverage;
it has not been replayed on an NPU. It makes no claim of a newly validated
full-block runner, stacked layers, whole-network dataset score, Decode,
full-table Engram residency, multi-card execution or serving support. The
original L25 evidence's missing large-readonly and dispatch-input artifacts
remain a limitation of that historical record.

## Checks without a device

Run the source receipt, dependency-closure and prefill-control checks:

```bash
python -m models.deepseek_v4_1_flash_a2.validate
```

With Torch and PyPTO installed, also import every module and construct the
bounded SWA, Full2, Ratio1 and Engram entries:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m models.deepseek_v4_1_flash_a2.validate --factories
```

These checks do not allocate NPU buffers. Actual operator validation should
use the existing [Golden Harness](../../run-and-validate/golden-harness.md)
with the same compute implementation, checkpoint operands and comparison
policy. Native boundary reports verify identity and interface integrity;
they do not replace numerical comparison against an independent reference.

## Decoder operators

The A2/A3 decoder adaptation is integrated into this same flat package. The
decoder contract is shared through `config.py`, `engram.py`, and
`sparse_attention.py`; W8A8 quantization and GEMM live in `quant.py` and
`gemm.py`. Layer-14 source and layer-15 reuse stages are in `layer14_15.py`.
The CPU references and bounded PyPTO validators are direct modules in this
directory, with no nested `verifier` package.

The decoder validator suite can be run without a device:

```bash
python -m pytest models/deepseek_v4_1_flash_a2/test_operators.py -q
python -m models.deepseek_v4_1_flash_a2.validate_layer14 --sim
python -m models.deepseek_v4_1_flash_a2.validate_layer15 --sim
```

The validators retain the A2/A3 ABI: BF16 hidden and cache rows, INT8 weights
and index payloads, FP32 W8A8 scales, FP16 index-cache scales, and INT32 GEMM
accumulation. Prefill callers continue to use the existing `prefill.py`,
`attention.py`, and cache modules unchanged.
