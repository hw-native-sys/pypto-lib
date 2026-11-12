# DeepSeek V4.1 Flash A5-to-A2 Module Alignment

This document is the ownership map between
`models/deepseek_v4_1_flash/` (the A5 implementation) and this A2/A3
implementation. It is a planning and review contract. It does not claim that
an A5 entry point can be copied unchanged: A5 uses FP8/FP4 and MX scale
layouts that are not part of the A2/A3 ABI.

## ABI translation

The semantic stages are aligned, while the tensor boundary is translated at
the owning operator:

| A5 boundary | A2/A3 boundary |
| --- | --- |
| FP8 projection weights and MX scales | INT8 W8A8 weights with FP32 channel scales |
| FP8/FP4 cache payloads | BF16 window and compressed KV rows |
| FP8 index payload and E8M0 scale | INT8 index payload with FP16 row scales |
| mixed precision dot products | INT32 accumulation, FP32 dequantization, BF16 narrowing |
| A5 distributed forward entry | bounded A2/A3 stage factory or single-layer entry |

Do not add `pl.FP8*`, `pl.FP4*`, or `pl.MX_B_NN` parameters to an A2 module
just to match an A5 signature. The A2 contract is defined in `config.py` and
must remain visible in the stage owner and its reference implementation.

## A5 module map

Status values mean:

- **aligned**: one A2 owner has the same semantic responsibility;
- **split**: the A2 implementation intentionally separates kernel, runtime,
  or validation responsibilities;
- **partial**: the stage exists, but a complete A5-style forward is not part of
  the accepted A2 scope;
- **out of scope**: no A2 implementation is planned for the current target.

### Shared operators

| A5 module | A2 owner | Status | Notes |
| --- | --- | --- | --- |
| `config.py` | `config.py` | aligned | Model dimensions and layer ownership contract. |
| A2/A3 operator ABI | `operator_config.py` | split | Dynamic shapes, dtype contract, and operator tiling defaults. |
| Runtime parallel sizing | `runtime_config.py` | split | TP sizing is read at runtime and kept outside model config. |
| `gemm.py` | `gemm.py`, `ops.py` | split | INT8 GEMM, packing, and dequantization are separate leaves. |
| `quant.py` | `quant.py`, `index_cache.py` | split | W8A8 and index-cache quantization use different scales. |
| `rmsnorm.py` | `rmsnorm.py`, `norm_rope.py` | split | Kernel and reference helpers are kept together by boundary. |
| `qkv_proj_rope.py` | `qkv_w8a8.py`, `qkv_proj_rope.py`, `qkv.py` | split | W8A8 QKV, staged BF16 projection, and reference are separate. |
| `o_proj.py` | `o_proj.py`, `output_proj.py`, `grouped_output.py` | split | A2 grouped Wo-A/Wo-B projection has a BF16 reference boundary. |
| `rope_tables.py` | `norm_rope.py`, `staged.py` | split | RoPE tables and the tiled RoPE leaf share the BF16 boundary. |
| `sparse_attention.py` | `sparse_attention.py`, `swa.py`, `swa_attention.py`, `swa_cache.py` | split | Cache publication, gather, attention, and reference are explicit. |
| `compressor.py` | `compressor.py`, `compressor_core.py` | split | Ratio-one and ratio-two state paths have separate ownership. |
| `hc.py` | `hc.py`, `hc_pre_runtime.py`, `hc_pre_full2_runtime.py` | split | PyPTO HC leaves are separate from native HC-pre boundaries. |
| `engram.py` | `engram.py`, `engram_ecotech.py`, `engram_hash.py`, `engram_offload.py`, `engram_w8a8.py` | split | Lookup, hash, residency, and projection are independently testable. |
| `expert_routed.py`, `expert_shared.py`, `gate.py` | `moe.py`, `moe_router.py`, `moe_shared.py`, `moe_routed.py` | split | A2 W8A8 expert and gate paths are split by routing and dispatch boundary. |

### Decode path

| A5 module | A2 owner | Status | Layer or responsibility |
| --- | --- | --- | --- |
| `decode_c2a_full.py` | `decode_c2a_full.py`, `decode_c2a_common.py` | partial | L14 C2A source; the shared provider remains behind the canonical entry. |
| `decode_c2a_reuse.py` | `decode_c2a_reuse.py`, `decode_c2a_common.py`, `decode_layer_stages.py` | aligned | L15 and L19 C2A reuse contracts. |
| `decode_c1a_full.py` | `decode_c1a_full.py`, `decode_layer_stages.py`, `candidate_pool.py` | aligned | L20 C1A source, compressor, QLI, and candidate blocks. |
| `decode_c1a_reuse.py` | `decode_c1a_reuse.py`, `decode_layer_stages.py` | aligned | L21 and L25 C1A cache reuse contracts. |
| `decode_c1a_reindex.py` | `decode_c1a_reindex.py`, `decode_layer_stages.py`, `indexer_select.py`, `indexer_prepare.py` | aligned | L24 index refresh followed by sparse attention. |
| `decode_indexer.py` | `indexer.py`, `indexer_query.py`, `indexer_select.py`, `indexer_prepare.py`, `index_cache.py`, `candidate_pool.py`, `qli_*` | split | QLI and candidate selection are separated at A2 native boundaries. |
| `decode_indexer.py` | `decode_indexer.py` | aligned | Canonical A2 entry re-exports query, selection, preparation, cache, and candidate stages. |
| `decode_swa.py` | `swa_attention.py`, `swa_native.py`, `swa_scheduler.py` | split | Decoder name remains a compatibility export; implementation is shared. |
| `decode_moe.py` | `moe.py`, `moe_router.py`, `moe_dispatch.py`, `moe_combine.py`, `moe_*_runtime.py` | split | W8A8 expert leaves, metadata, and dispatch are separate. |
| `decode_moe.py` | `decode_moe.py` | aligned | Canonical EP2 orchestration entry; expert and runtime leaves remain separate. |
| `decode_layer.py` | `decode_c2a_*.py`, `decode_c1a_*.py`, `decode_c2a_common.py`, `decode_layer_stages.py` | partial | Layer contracts and stage factories; complete A5 forward is deferred. |
| `decode_layer.py` | `decode_layer.py` | aligned | Canonical layer selector for the supported EP2 physical-layer entries. |
| `decode_fwd.py` | `layer0.py`, `prefill.py`, `decode_c2a_*.py`, `decode_c1a_*.py` | partial | Independent layer composition only; no serving or whole-network runner. |

The migrated L19-L25 schedule is fixed in the canonical decode files:

```text
L19: C2A reuse <- L14
L20: C1A full/source
L21: C1A reuse <- L20
L24: C1A reindex <- L20 cache
L25: C1A reuse <- L20 cache and L24 index selection
```

### Prefill path

| A5 module | A2 owner | Status | Responsibility |
| --- | --- | --- | --- |
| `prefill_c2a_full.py` | `layer2_prefill.py`, `layer2_c1c2.py`, `c2_prefill_pool.py`, `ecotech_full2_*` | split | Ratio-two projection, pooling, state publication, and readback. |
| `prefill_c2a_reuse.py` | `ecotech_prefill_consumer.py`, `sparse_consumer_prefill.py` | partial | Consumer stages are available; a distributed full forward is deferred. |
| `prefill_c1a_full.py` | `ratio1_prefill.py`, `ecotech_full2_prefill.py`, `ecotech_full2_observed.py` | split | Ratio-one publication and observed index stages. |
| `prefill_c1a_reindex.py` | `indexer_query.py`, `indexer_select.py`, `indexer_prepare.py`, `ecotech_prefill_consumer.py` | partial | Reindex leaves are available as independent stages. |
| `prefill_c1a_reuse.py` | `ecotech_prefill_consumer.py`, `sparse_consumer_prefill.py` | partial | Reuse consumer stages are available. |
| `prefill_swa.py` | `ecotech_swa_prefill.py`, `swa.py`, `swa_cache.py` | aligned | BF16 SWA publication and attention. |
| `prefill_indexer.py` | `indexer*.py`, `index_cache.py`, `c1c2.py` | split | Index preparation and cache publication are independent. |
| `prefill_moe.py` | `moe.py`, `moe_router.py`, `moe_dispatch.py`, `moe_combine.py` | split | A2 W8A8 shared/routed expert path. |
| `prefill_layer.py` | `layer0.py`, `layer0_single_card.py`, `layer2_prefill.py`, `decode_c2a_common.py` | partial | Single-layer composition and references. |
| `prefill_fwd.py` | `prefill.py`, `attention.py`, `attention_block.py` | partial | Existing A2 prefill entries; no A5 distributed runner. |
| `prefill_fwd.py` | `prefill_fwd.py` | aligned | Canonical A2 Prefill composition entry. |
| `prefill_cp_allgather.py` | none | out of scope | A2 single-card target does not accept the A5 collective ABI. |

## A2 current ownership groups

The physical package remains flat for import compatibility, but new changes
should follow these logical groups:

| Logical group | Current files | Owns |
| --- | --- | --- |
| Shared leaves | `config.py`, `operator_config.py`, `runtime_config.py`, `ops.py`, `gemm.py`, `quant.py`, `norm_rope.py`, `rmsnorm.py`, `staged.py`, `qkv.py`, `qkv_w8a8.py`, `qkv_proj_rope.py`, `o_proj.py`, `output_proj.py`, `grouped_output.py`, `sparse_attention.py`, `swa.py`, `swa_attention.py`, `swa_cache.py`, `decode_swa.py` | A2/A3 tensor ABI and reusable computation. |
| Cache and index stages | `compressor*.py`, `c1c2.py`, `c2_prefill_pool.py`, `ratio1_prefill.py`, `index_cache.py`, `indexer*.py`, `candidate_pool.py`, `sparse_consumer_prefill.py` | C1A/C2A publication, QLI, candidate selection, and cache ownership. |
| Layer composition | `attention*.py`, `prefill*.py`, `rmsnorm.py`, `layer0*.py`, `layer2_*.py`, `decode_layer.py`, `decode_c2a_*.py`, `decode_c1a_*.py`, `decode_c2a_common.py`, `decode_layer_stages.py`, `ecotech_*.py` | Bounded layer and path composition. Canonical entries select the existing stage owners. |
| HC, Engram, and MoE | `hc.py`, `engram*.py`, `moe.py`, `moe_*.py` | Model-specific side paths and W8A8 expert boundaries. |
| Runtime boundaries | `*_native.py`, `*_runtime.py`, `*_boundary.py`, `*_scheduler.py`, `smla_staging.py`, `projection_runtime.py`, `runtime_utils.py`, `scheduler_utils.py`, `bf16_linear_native.py` | Native calls, descriptors, scheduling metadata, and identity checks. |
| Validation and data | `raw_checkpoint.py`, `weights.py`, `source_manifest.json` | Checkpoint loading and implementation source receipts. Precision, vLLM capture, and test entry points live under `/data/disk/sunhaolun/captrue_and_vllm/models/deepseek_v4_1_flash_a2/`. |

### Runtime and validation support

| A5 responsibility | A2 owner | Status |
| --- | --- | --- |
| `input_pack.py`, decode metadata | `utils.py`, `swa_scheduler.py`, `smla_scheduler.py`, `smla_staging.py` | split |
| native operator registration | `swa_native.py`, `smla_native.py`, `bf16_linear_native.py` | aligned |
| native subprocess boundaries | `*_runtime.py`, `*_boundary.py`, `runtime_utils.py`, `projection_runtime.py` | split |
| model weights and checkpoint loading | `raw_checkpoint.py`, `weights.py` | split |
| Torch/vLLM capture and comparison | `/data/disk/sunhaolun/captrue_and_vllm/models/deepseek_v4_1_flash_a2/capture_vllm_*.py`, `compare_vllm_dump.py`, `precision.py` | validation-only |
| executable validation entries | `/data/disk/sunhaolun/captrue_and_vllm/models/deepseek_v4_1_flash_a2/validate*.py`, `test_operators.py`, `probe_swiglu.py` | validation-only |

## A5 modules intentionally not mirrored

`dspark_attention.py`, `dspark_context_kv.py`, `dspark_drafter.py`,
`dspark_markov.py`, `vision.py`, and `lm_head.py` are not part of the current
A2/A3 decoder and single-layer prefill target. They require separate runtime,
collective, or dtype contracts. They should not be copied into this package
until an A2 consumer, input contract, and independent validation case exist.

The A5 C++/header files under `kernels/` are likewise not portable source
receipts for A2. A2 native implementations live behind the registered runtime
modules and must be added only with an A2-compatible ABI and validation case.

## Dependency direction

The intended dependency graph is:

```text
config -> shared leaves -> path stages -> layer factories -> runtime boundary
                                      \-> Torch/vLLM validation
```

Decode and prefill stages may share leaves, but they must not import each
other. Validation imports the implementation it measures; implementation code
does not import capture or comparison scripts. The current flat layout is kept
to preserve existing imports. A future physical `common/`, `decode/`,
`prefill/`, `runtime/`, and `validation/` split is safe only after all callers
are migrated and the source/acceptance manifests are regenerated.

## Migration order

1. Keep the A2/A3 dtype contract and the L19-L25 ownership table frozen.
2. Add or complete one path-specific stage only when it has a real caller and
   a vLLM comparison case.
3. Move validation scripts behind compatibility imports only after the stage
   import graph is acyclic.
4. Add a complete `decode_fwd` or `prefill_fwd` entry only with a bounded
   single-layer runner, checkpoint input contract, and device evidence.
5. Remove compatibility imports and update `source_manifest.json` only after
   all repository callers use the aligned owner module.
