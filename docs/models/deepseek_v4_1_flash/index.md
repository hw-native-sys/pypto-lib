# DeepSeek V4.1 Flash

`models/deepseek_v4_1_flash/` is the implementation staging area for the
DeepSeek-V4.1-Flash checkpoint. The first milestone establishes the text-model
configuration, layer schedule, cache ownership, inference metadata, Torch
goldens, and prefill/decode kernel contracts. Checkpoint loading and optimized
PyPTO leaf kernels remain follow-up work.

## Checkpoint shape

The `FLASH` preset in
[config.py](../../../models/deepseek_v4_1_flash/config.py) mirrors the released
checkpoint's 40-layer text backbone and quantization metadata. Auxiliary
drafting, n-gram, and multimodal components are intentionally out of scope.

| Property | Value |
| --- | ---: |
| Hidden size | 5,120 |
| Backbone layers | 40 |
| Attention heads | 64 |
| Head dimension | 512 |
| Routed experts / active experts | 384 / 6 |
| Shared experts | 1 |
| Hyper-connection width | 4 |
| Vocabulary | 129,280 |
| Maximum position | 1,048,576 |

The attention schedule is:

- Layers 0-1 use the 128-token sliding window only.
- Layers 2-19 use ratio-2 compressed sparse attention. KV sources are layers
  2, 8, and 14; those layers are also index sources.
- Layers 20-39 use ratio-1 compressed sparse attention. Layer 20 owns the KV
  cache, while layers 20, 24, 28, 32, and 36 refresh the index selection.

## Parallel-development structure

Each attention mode and execution phase has one ownership file. Every file
contains a Torch golden and an explicit `@pl.jit.inline` ABI; kernel bodies are
the remaining parallel work.

Run an operator file directly to execute its deterministic CPU golden:

```bash
source .venv/bin/activate-pypto
python models/deepseek_v4_1_flash/decode_c1a_reindex.py
```

The command prints `[GOLDEN] PASS` and exits nonzero when the reference fails.
Once a kernel body lands, its owner can extend the same file with the thin
`@pl.jit` entry, `build_tensor_specs()`, and device `run(...)` block.

| Workstream | Files |
| --- | --- |
| Encoder SWA | `prefill_attn_swa.py` (leaf), `prefill_swa.py` (HC orchestration), `decode_swa.py` |
| Encoder C2A Full | `prefill_c2a_full.py`, `decode_c2a_full.py` |
| Encoder C2A Reuse | `prefill_c2a_reuse.py`, `decode_c2a_reuse.py` |
| Decoder C1A Full | `prefill_c1a_full.py`, `decode_c1a_full.py` |
| Decoder C1A Reindex | `prefill_c1a_reindex.py`, `decode_c1a_reindex.py` |
| Decoder C1A Reuse | `prefill_c1a_reuse.py`, `decode_c1a_reuse.py` |
| Hierarchical indexer | `hierarchical_sparse_indexer.py` |
| Hyper-connections | `hc_mixes.py`, `hc_pre.py`, `hc_post.py` |
| Attention TP transports | `attention_tp.py` |
| Expert parallelism | `moe.py` |
| Shared configuration and goldens | `config.py`, `metadata.py`, `golden.py`, `attention_common.py` |
| Quantization and RoPE tables | `quantization.py`, `rope_tables.py` |

## Decode composition

[decode_layer.py](../../../models/deepseek_v4_1_flash/decode_layer.py) resolves
all six modes and source ownership from `FLASH.layer_config`. Its complete
Block skeleton preserves delayed pre-mix ordering: Attention consumes the
incoming mix, FFN consumes the Attention pre-mix, and the Block returns the
FFN pre-mix for the next layer. Run the six small CPU Block references with:

```bash
python models/deepseek_v4_1_flash/decode_layer.py --stage block --cpu-golden
```

The full Block device path awaits C1A decode integration, cache ABI agreement,
and MoE integration. `decode_layer_kernel_skip_reason` lists those dependencies.
Both the Block factory and `--stage block` device command enforce readiness before JIT
construction. Block CPU references currently require all capacity rows active;
the full Block hardware fixture remains pending integration.

The same file provides `--stage attention` (the default) for implemented SWA and
C2A Full/Reuse paths. It selects the leaf adapter before JIT dependency
discovery and does not require MoE. `attention_half_skip_reason` gates
unintegrated C1A paths. `make_decode_layer_program` selects the stage in Python;
both stages use `make_attention_rank` for the same Attention orchestration.
For an allocated TP4 group:

```bash
python models/deepseek_v4_1_flash/decode_layer.py --stage attention -p a5 -d 0,1,2,3 \
  --tp 4 --layer-id 3 --tokens 33 --active-tokens 31 --requests 6 \
  --epochs 2 --save-data
```

Use representative layer IDs 0, 2, and 3 for SWA, C2A Full, and C2A Reuse.

Q/KV preprocessing lives in `qkv_proj_rope.py`. `q_proj_qr` writes the
normalized Q latent, `q_proj_rope` expands and rotates it, and
`kv_proj_rope` projects, normalizes, and rotates window KV. SWA and C2A
prefill/decode use these stages; C2A Full also passes the same normalized
latent to its indexer. Scratch remains caller-owned to preserve chunk
reuse and task ordering. Prefill SWA uses the `prefill_*` stages to retain
its group-32 scale-corrected BF16 Q projection and fixed-worker scheduling.
Cache publication and TP communication stay in the Attention caller.
C1A uses its existing preprocessing until its independent golden baseline
and migration are accepted.

`o_proj.py` combines inverse RoPE, grouped Wo-A, and MXFP8 Wo-B into
an FP32 TP-local partial output. SWA/C2A call `o_proj`; prefill SWA calls
`prefill_o_proj` to preserve its fixed-worker RoPE schedule. The caller
owns the unrotated/latent scratch and runs the original prefill/decode
TP all-reduce. C1A retains its existing output path pending migration.

TP1 and TP4 are supported by the half-layer validation entry. Each dispatch
computes mHC mixes/pre and input RMSNorm, invokes Attention with consecutive
communication epochs, then computes mHC post. Validation reuses each leaf's
fixture, reference, and precision checks and exposes the intermediate
boundaries. It checks updated caches and exact non-owner storage.

mHC boundaries cover the full token capacity. RMSNorm and Attention write
the active prefix; their visible buffers are `InOut` so inactive rows retain
the caller's values. The Reuse case validates those rows with a nonzero
sentinel. `attention_hidden` and `attention_pre_mix` are fully written `Out`
boundaries with shapes `[tokens, 4, 5120]` and `[tokens, 4]` per rank.
Normalized and hidden precision statistics cover active rows only; the inactive
suffix is checked independently so it cannot dilute the active error budget.

Full attention owns compressed KV and index-key publication. Reindex consumes
the C1A cache and the layer-20 candidate mask but computes a new index query.
Reuse consumes the source layer's physical Top-K rows and has no compressor or
indexer weights. The hierarchical indexer first selects 2,048 blocks of eight
compressed positions at layer 20; later reindex layers select their final 512
positions only inside that candidate mask.

The final HC collapse has no learned head parameters: it applies the last
layer's delayed `pre_mix` directly to the four residual streams. HC mixes are
depth-local values and are not persisted as sequence state.

The production cache ABI uses a 128-token scheduler block and keeps payloads
quantized in HBM:

- Window KV payload: `[blocks, 128, 1, 512]`, MXFP8 E4M3, with
  `[blocks, 128, 1, 16]` E8M0 group-of-32 scales.
- Ratio-2 compressed KV payload: logical `[blocks, 128, 1, 512]`, packed
  MXFP4 E2M1, with `[blocks, 128, 1, 32]` E4M3 group-of-16 scales. Layers 2,
  8, and 14 own these pools; one physical row represents two original tokens.
- Ratio-1 compressed KV uses the same packed FP4 payload and scale ABI and is
  owned by layer 20.
- Index-key payload: logical `[blocks, 128, 1, 128]`, packed MXFP4 E2M1, with
  `[blocks, 128, 1, 4]` E8M0 group-of-32 scales.
- Ratio-2 recurrent state: `[32, 2, 512]`, FP32 and request-scoped. Head zero
  stores one pending KV row and head one stores its gate scores. Ratio 1 has no
  recurrent compressor state.

Compressed KV and index-key tensors for a source share the same
`c{ratio}a_cmp_kv` block table. Recurrent state is indexed by the DP-local
request row and does not grow with context length. Torch goldens quantize on
cache publication and dequantize on cache reads; BF16 cache values are only an
intermediate reference representation, not the kernel ABI.
Indexer workspaces store physical flattened cache-row ids, padded with `-1`,
so separately scheduled window and compressed sparse-attention kernels do not
depend on a concatenated-cache offset. Candidate masks remain in request-local
compressed-position space.

`ForwardMetadata` lowers packed query starts, request ids, absolute positions,
previous/new KV lengths, cache slot mappings, sliding-window indices,
per-token causal compressed lengths, per-request compressed lengths and
remainders, ragged compressor output starts, source-token rows, and compressed
RoPE positions. The same lowering serves prefill and continuous-batch decode.

The target deployment is one eight-card A5 node with TP4 attention, two DP
groups, and EP8 routed experts. TP1/2/4/8 and compatible EP2/4/8 shapes remain
available for bring-up. The EP world is reinterpreted as `DP = EP / TP`
contiguous attention groups; `tp_rank = rank % TP` and
`group_base = rank - tp_rank`.

The first implementation targets pure head tensor parallelism. Every TP rank
sees the same token batch. `wq_a`, `wkv`, compressor, indexer, and the
single-head KV caches are replicated. `wq_b`, query heads, attention sinks,
and output groups are sharded across TP ranks. Each rank computes
16 query heads and two output groups; one FP32 TP all-reduce reconstructs the
complete hidden output. Before EP8 dispatch, token-row ownership is assigned
round-robin across the four TP ranks. This prevents replicated attention rows
from being dispatched four times; MoE combine returns the rows to the TP
layout. DSA context parallelism is intentionally out of scope.

The service capacity contract is 32 active sequences and 4,096 scheduled
prefill token rows per DP group. With five reserved DSpark draft rows plus one
target row, the decode ABI reserves 192 token rows per DP group. DP2 therefore
supports up to 64 active sequences globally. DSpark execution itself remains
follow-up work.

Query/output low-rank projections and shared experts use MXFP8 payloads.
Routed expert weights remain output-major packed MXFP4 with E8M0 group-of-32
scales in HBM. The planned kernel loads one FP4 tile, casts that tile to FP8 in
on-chip memory, and uses the supported MXFP8 Cube path with dynamically
quantized activations; it does not expand the complete expert tensor.

At the maximum 1,048,576-token context, the low-bit attention cache is about
0.94 GB per request per card, compared with about 3.37 GB for BF16. At 32
requests this is about 30.2 GB/card instead of 107.8 GB/card. Ideal per-card
weight payload is about 66 GB before alignment and runtime workspaces: 36.1 GB
of EP8 routed experts, 25.3 GB of sharded Engram, and 4.7 GB of other TP4
weights. These figures make eight-card A5 deployment a plausible target, but
the issue remains open until a real-device run records peak HBM and generates
reference-matching text.

## Path to token generation

The implementation milestones are ordered by dependency:

1. Implement and compile the attention TP all-reduce, mHC, and SWA.
   Packed prefill SWA is wired through mHC: `mhc_mixes` → `mhc_pre` →
   `prefill_attn_swa` → `mhc_post`. Run `python models/deepseek_v4_1_flash/prefill_swa.py`.
2. Implement C2A Full, then validate Full-to-Reuse cache and Top-K replay.
3. Implement C1A Full and the level-one candidate selector, then Reindex and Reuse.
4. Implement the three-phase EP-MoE dispatch/local-expert/combine body.
5. Compose the operators into the 40-layer prefill/decode token loop.

Until the leaf kernels and weight loader land, this directory is not a runnable
model and is not exposed to `pypto-serving`.
