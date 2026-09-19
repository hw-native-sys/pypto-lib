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

## Reusable RoPE tables

Serving callers can generate each RoPE profile once with
`precompute_rope_tables(capacity, compressed_attention=...)`, move the returned
FP32 cosine/sine tables to the execution device once, and retain them across
requests. Use `materialize_token_rope_tables(cos, sin, position_ids)` to gather
rows without recomputing frequencies or trigonometric functions. Positions
must be INT32/INT64 tensors on the same device as the tables. Outputs retain
the table dtype and device, with shape `[*position_ids.shape, rope_dim // 2]`.
Negative positions yield identity rotation; nonnegative positions must be
below the allocated capacity. Unlike V4's duplicated full-width tables, V4.1
uses half-width tables for adjacent-pair rotation.

```python
from models.deepseek_v4_1_flash.rope_tables import (
    materialize_token_rope_tables,
    precompute_rope_tables,
)

# Initialize once per profile and device; retain these tensors in the caller.
cos, sin = precompute_rope_tables(capacity, compressed_attention=False)
cos, sin = cos.to(device), sin.to(device)

# Each forward: position_ids is already on device.
rope_cos, rope_sin = materialize_token_rope_tables(cos, sin, position_ids)
```

The caller owns table lifetime, capacity, and matching the model configuration
and attention profile. This library API does not allocate a serving cache or
change kernel arguments. `select_rope_rows` remains available for callers that
compute rows on demand without retaining a full table.

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
| Encoder C2A Full | `prefill_attn_c2a_full.py` (leaf), `prefill_c2a_full.py` (HC orchestration), `decode_c2a_full.py` |
| Encoder C2A Reuse | `prefill_attn_c2a_reuse.py` (leaf), `prefill_c2a_reuse.py` (HC orchestration), `decode_c2a_reuse.py` |
| Decoder C1A Full | `prefill_c1a_full.py`, `decode_c1a_full.py` |
| Decoder C1A Reindex | `prefill_c1a_reindex.py`, `decode_c1a_reindex.py` |
| Decoder C1A Reuse | `prefill_c1a_reuse.py`, `decode_c1a_reuse.py` |
| Hierarchical indexer | `hierarchical_sparse_indexer.py` |
| Hyper-connections | `hc_mixes.py`, `hc_pre.py`, `hc_post.py` |
| Attention TP transports | `attention_tp.py` |
| Expert parallelism | `moe.py` |
| Shared configuration and goldens | `config.py`, `metadata.py`, `golden.py`, `attention_common.py` |
| Quantization and RoPE tables | `quantization.py`, `rope_tables.py` |

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

For active ratio-2 requests, `build_forward_metadata` requires the keyword
`compressor_state_slots`: a mapping from each KV source layer to an INT32/INT64
`[B]` vector of engine-owned stable state rows. Allocated rows must be distinct
across active and paused requests and lie in `[0, MAX_BATCH_PER_DP)`. Reorder these vectors with
the batch while leaving persistent state in its original slots. Inactive
requests without a retained slot may use `-1`; an empty batch needs no state mapping. The caller owns
reset before slot reassignment and pending-pair restoration on resume. This
preserves the existing bounded state-pool ABI; it is not a ring-cache or
speculative rollback implementation.

`compressor_state.CompressorStateCache` supplies this lifecycle for engines
using Torch buffers. Create one pool per TP rank and use request keys containing
their generation. `allocate` initializes all source buffers, `slots` resolves
the current batch order without moving state, and `release` clears the slot
before reuse. Paused requests keep their slots until explicitly released.
Each `buffers[source]` has the existing FP32 `[32, 2, 512]` kernel layout;
the two rows hold the pending KV projection and gate score, respectively.

```python
from models.deepseek_v4_1_flash.compressor_state import CompressorStateCache
from models.deepseek_v4_1_flash.metadata import build_forward_metadata

state = CompressorStateCache(device=query_start_loc.device)
request_keys = [("request-a", 0), ("request-b", 0)]
for key in request_keys:
    state.allocate(key)
metadata = build_forward_metadata(
    query_start_loc, kv_seq_lens, window_block_table, compressed_block_tables,
    compressor_state_slots=state.slots(request_keys),
)
# For each source, pass metadata.compressor_state_rows[source] and
# state.buffers[source] to its C2A kernel. Keep this pool across dispatches.
# After all source kernels finish at the committed prefix boundary:
saved = state.snapshot(request_keys[0], prefix_length=committed_length)
state.release(request_keys[0])
# Restore the matching KV prefix separately, then restore pending state:
resumed_key = ("request-a", 1)
state.allocate(resumed_key, prefix_length=saved.prefix_length, snapshot=saved)
```

An odd prefix cannot resume without a matching pending-state snapshot; the
allocator rejects it instead of silently consuming zeros. An even prefix may
start with cleared state because its next token begins a new compression pair.
Snapshots clone only one pending pair per source, remain on the pool device,
and do not alias live storage. The engine supplies the committed prefix length
and must save the corresponding KV prefix and model/rank identity with it.
Lifecycle operations must run after kernel completion on the same stream, or
after an explicit wait across streams. Release must also wait for commands
already holding old slot mappings; generation keys do not revoke device work.
This helper does not replace an engine's existing allocator or transaction
manager, and does not implement ring history or speculative rollback.

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
   Packed prefill C2A Full and Reuse are wired through mHC the same way,
   with the attention RMSNorm the block runs between `mhc_pre` and the
   leaf: `python models/deepseek_v4_1_flash/prefill_c2a_full.py`.
3. Implement C1A Full and the level-one candidate selector, then Reindex and Reuse.
4. Implement the three-phase EP-MoE dispatch/local-expert/combine body.
5. Compose the operators into the 40-layer prefill/decode token loop.

Until the leaf kernels and weight loader land, this directory is not a runnable
model and is not exposed to `pypto-serving`.
