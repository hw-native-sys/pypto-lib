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

## Serving-owned full RoPE tables

V4.1 provides a PyPTO inline stage that reads caller-owned full cosine/sine
tables and writes only the active token rows inside the compiled graph. It does
not recompute frequencies, copy the full table per call, or invoke Torch during
dispatch. As in V4 MTP's decode forward, prepare rows once before the attention
layers; DSpark's decode forward likewise passes prepared token-major rows to
its layers.

`rope_tables.materialize_rope_rows` accepts:

- `freqs_cos`, `freqs_sin`: read-only FP32 `[table_capacity, rope_dim // 2]`
  tables for one profile, retained by serving across calls.
- `position_ids`: INT32 `[T]` absolute positions.
- `num_tokens`: the active prefix length, in `[0, T]`.
- `rope_cos`, `rope_sin`: caller-provided FP32 `[T, rope_dim // 2]` outputs.

Nonnegative positions must be below table capacity. Negative positions write
identity rotation, used for unpublished compressed tokens. Padding output
rows remain untouched; zero active tokens launch no row work. V4.1 retains
FP32 half-width tables for adjacent-pair rotation, rather than adopting V4's
BF16 full-width representation.

The exported `decode_attn_c2a_full`, `decode_c2a_full`, `prefill_c2a_full` and
`c2a_full_partial` retain
their token-major `rope_cos`/`rope_sin` and `compressed_rope_cos`/
`compressed_rope_sin` inputs. They do not allocate RoPE row buffers or gather
full tables on each layer invocation.

The `decode_attn_c2a_full.make_program` and `prefill_c2a_full.make_hc_program`
forward entries accept
`freqs_cos`/`freqs_sin`, `compressed_freqs_cos`/`compressed_freqs_sin`,
`position_ids` and `compressed_rope_positions`. All four tables share one position
capacity in these entries. Each rank allocates four row buffers and gathers the
query and compressed rows once, before the repeated-attention loop. Every epoch
in that invocation reuses those read-only rows. Table generation in the driver
is fixture initialization, not work performed during dispatch.

For a multilayer forward, prepare one pair of row buffers for each distinct
RoPE profile, position vector and active token count. Reuse those buffers for
all layers with that same combination. A different compression source may have
different positions even when it shares a profile; do not reuse rows across
such combinations. Recompute rows when the positions or active token count
change. This is per-forward reuse, not a persistent cache of selected rows.

Compose the preparation before the layers:

```python
materialize_rope_rows(
    freqs_cos, freqs_sin, position_ids, num_tokens, rope_cos, rope_sin,
)
materialize_rope_rows(
    compressed_freqs_cos, compressed_freqs_sin,
    compressed_rope_positions, num_tokens, compressed_rope_cos, compressed_rope_sin,
)
# Pass these same row tensors to each matching row-based attention entry.
```

Supply `metadata.compressed_rope_position_ids[source]` as
`compressed_rope_positions`; it contains compression-group start positions or
`-1`. Pass the gathered tensors directly to their RoPE consumers so the normal
tensor dependencies are visible. In an explicitly manual dependency region,
preserve both producer TaskIds alongside the existing cache-publication and
previous-epoch dependencies; source order alone does not establish an edge.
Keep the row buffers alive and unchanged until all consuming layers complete.
The external serving adapter and other attention modes are not migrated here.

`precompute_rope_tables` remains a CPU initialization utility when the caller
needs to generate a complete profile. It is not needed in the compiled dispatch
path when serving already supplies full tables.

## Parallel-development structure

Each attention mode and execution phase has one ownership file. Every file
contains a Torch golden and an explicit `@pl.jit.inline` ABI; kernel bodies are
the remaining parallel work.

Run an operator file directly to execute its deterministic CPU golden:

```bash
source .venv/bin/activate-pypto
python models/deepseek_v4_1_flash/decode_attn_c1a_reindex.py
```

The command prints `[GOLDEN] PASS` and exits nonzero when the reference fails.
Once a kernel body lands, its owner can extend the same file with the thin
`@pl.jit` entry, `build_tensor_specs()`, and device `run(...)` block.

| Workstream | Files |
| --- | --- |
| Encoder SWA | `prefill_attn_swa.py` (leaf), `prefill_swa.py` (HC orchestration), `decode_attn_swa.py` (decode leaf), `decode_swa.py` (decode HC orchestration) |
| Encoder C2A Full | `prefill_attn_c2a_full.py` (leaf), `prefill_c2a_full.py` (HC orchestration), `decode_attn_c2a_full.py` (decode leaf), `decode_c2a_full.py` (decode HC orchestration) |
| Encoder C2A Reuse | `prefill_attn_c2a_reuse.py` (leaf), `prefill_c2a_reuse.py` (HC orchestration), `decode_attn_c2a_reuse.py` (decode leaf), `decode_c2a_reuse.py` (decode HC orchestration) |
| Decoder C1A Full | `prefill_attn_c1a_full.py`, `decode_attn_c1a_full.py` (leaf), `prefill_c1a_full.py`, `decode_c1a_full.py` (HC orchestration) |
| Decoder C1A Reindex | `prefill_attn_c1a_reindex.py`, `decode_attn_c1a_reindex.py` (leaf), `prefill_c1a_reindex.py`, `decode_c1a_reindex.py` (HC orchestration) |
| Decoder C1A Reuse | `prefill_attn_c1a_reuse.py`, `decode_attn_c1a_reuse.py` (leaf), `prefill_c1a_reuse.py`, `decode_c1a_reuse.py` (HC orchestration) |
| Hierarchical indexer | `hierarchical_sparse_indexer.py` |
| Hyper-connections | `hc_mixes.py`, `hc_pre.py`, `hc_post.py` |
| Attention TP transports | `attention_tp.py` |
| Shared Attention primitives | `attention_ops.py` (`make_mx_projection`, BF16 projection, RMSNorm, RoPE, and dependency-aware variants) |
| Shared Q/KV preprocessing | `qkv_proj_rope.py` (`q_proj_qr`, `q_proj_rope`, `kv_proj_rope`, `qkv_proj_rope` and Prefill variants) |
| Shared output projection | `o_proj.py` (`grouped_output`, `o_proj`, `prefill_o_proj`) |
| Expert parallelism | `moe.py` |
| Layer composition | `prefill_layer.py` |
| Shared configuration and goldens | `config.py`, `metadata.py`, `golden.py`, `attention_common.py` |
| Quantization and RoPE tables | `quantization.py`, `rope_tables.py` |

SWA, C2A, and C1A use the shared Attention primitives and stage compositions
above. Decode C1A selects the dependency-aware variants because their explicit
TaskId edges are part of the hardware schedule, while retaining its distinct MX
projection implementation. It also uses `qkv_proj_rope_with_deps` and
`o_proj_with_deps` as the shared composition boundaries. The specialized
Prefill SWA Q-A projection also remains in
`qkv_proj_rope.py` because it preserves that path's group-32 scale decoding.

## Decode composition

[decode_layer_plan.py](../../../models/deepseek_v4_1_flash/decode_layer_plan.py)
resolves all six modes and source ownership from `FLASH.layer_config`. Each
mode has an independently executable Attention half-layer entry:

| Mode | Attention kernel | Composition entry | Representative layer |
| --- | --- | --- | ---: |
| SWA | `decode_attn_swa.py` | `decode_swa.py` | 0 |
| C2A Full | `decode_attn_c2a_full.py` | `decode_c2a_full.py` | 2 |
| C2A Reuse | `decode_attn_c2a_reuse.py` | `decode_c2a_reuse.py` | 3 |
| C1A Full | `decode_attn_c1a_full.py` | `decode_c1a_full.py` | 20 |
| C1A Reindex | `decode_attn_c1a_reindex.py` | `decode_c1a_reindex.py` | 24 |
| C1A Reuse | `decode_attn_c1a_reuse.py` | `decode_c1a_reuse.py` | 21 |

Decode Attention is sequence parallel over the TP group: each rank owns a
contiguous slab of the batch's token rows (`T_local = ceil(T / TP)`), the
normalized Attention input crosses the group once at the block head
(`decode_tp_input_all_gather`, `[T_local, D] -> [T, D]`), and the row-parallel
output projection leaves through ReduceScatter(SUM)
(`decode_tp_output_reduce_scatter`) rather than an all-reduce every rank would
slice afterwards. The residual stream itself never crosses the group; each rank
expands only its own rows in `mhc_post`. Every decode Attention mode keeps both
wirings — the historical all-reduce entries and the `*_sharded` sugar — and each
harness validates both by default; `OUTPUT_T_DYN` names the local row extent next
to `T_DYN` for the full range.

`--wiring {both,replicated,sharded}` selects which of the two runs a harness
executes (default `both`, what CI uses). The wirings are separate ABIs — the
sharded side shards the token extent and adds `gathered` — so a replay tree
belongs to exactly one of them: with `--golden-data`/`--runtime-dir`, the
directory for wiring `W` is always `<dir>/W`, e.g. `--golden-data out/data` reads
`out/data/replicated` and `out/data/sharded`. A tree saved from one wiring is
therefore never offered to the other.

Row ownership is the fixed physical slab `ceil(capacity / TP)`, not
`ceil(active / TP)`: a padded batch, `T < TP` and a rank whose slab starts past
the active range all keep the same row mapping the collectives publish, and the
active count only masks the slab suffix. Padding rows are therefore part of the
published slab but never part of the gather or the summed rows; every mode
materializes them as zeros (`zero_bf16_padding` for SWA/C2A, the same call after
`rms_norm` for C1A) and the ReduceScatter clears its inactive output rows, so
no stale transport window or harness sentinel reaches the next stage. The
comparators cover the active slabs and assert that the inactive ReduceScatter
rows are zero.

The device kernels already take the active count as their `num_tokens` scalar and
derive the slab width from the tensor shapes, but only C2A Reuse exposes it:
`validate_args` rejects `active_tokens < tokens` for SWA and C2A Full, and the
three C1A entries keep the token count in the host signature, so a caller cannot
express an inactive suffix at all and every capacity row runs as a real query.
The decode caller contract is therefore a compact batch, `tokens ==
active_tokens`: a caller holding a reserved-capacity batch compacts it before the
half-layer, or uses C2A Reuse, the mode that publishes only the active prefix. Its
`tokens = 32`, `active_tokens = 24` TP4 case is part of the validation matrix.

Computed per-rank traffic of the boundary (elements, not measured elapsed
time): AllGather publishes `T_local * D * 2` bytes of BF16 and reads
`T * D * 2`, ReduceScatter publishes and reads `T * D * 4` bytes of FP32
partials, i.e. `D * (2 * T_local + 10 * T)` bytes per rank per half-layer,
against `D * 20 * T` for the replicated all-reduce at TP4. At `T = 192`,
`D = 5120`, TP4 that is about 10.3 MB versus 19.7 MB, and the `4 * T * D * 4`
byte residual stream never crosses the group. End-to-end latency and bus
bandwidth still need a measurement.

The mode files own their entry contract and readiness state. SWA and C2A use
the spec-driven boundary helpers in `decode_common.py`; C1A keeps its native
static token/page ABI and validation harness. `decode_layer.py` is the thin
complete Block integration entry. Its Torch golden preserves delayed pre-mix
ordering: Attention consumes the incoming mix, FFN consumes the Attention
pre-mix, and the Block returns the FFN pre-mix for the next layer. Run the six
small CPU Block references with:

```bash
python models/deepseek_v4_1_flash/decode_layer.py --stage block --cpu-golden
```

All six Attention half-layers are implemented. The full Block device path still
awaits EP8 MoE integration and its hardware fixture;
`decode_layer_kernel_skip_reason` reports that dependency. Both the Block
factory and `--stage block` device command enforce readiness before JIT
construction. Block CPU references currently require all capacity rows active.

Every mode file provides hardware validation without requiring MoE. The
`decode_layer.py --stage attention` compatibility dispatcher accepts the
spec-driven SWA/C2A ABI; C1A validation uses each mode's native entry directly.
For an allocated TP4 group:

```bash
python models/deepseek_v4_1_flash/decode_c2a_reuse.py -p a5 -d 0,1,2,3 \
  --tp 4 --tokens 33 --active-tokens 31 --requests 6 \
  --epochs 2 --save-data
```

Each validation epoch invokes the complete production composition: mHC
mixes/pre, input RMSNorm, one Attention call, and mHC post. The MoE half follows
the same official Block ordering: its mHC mixes are computed first, the incoming
Attention pre-mix collapses the residual streams, RMSNorm runs once, then the
router and experts execute before mHC post. Epochs repeat the
same fixture inputs for validation and benchmarking; they do not feed one
epoch's hidden or pre-mix output into the next. Validation reuses each leaf's
fixture, reference, and precision checks and checks updated caches and exact
non-owner storage.

mHC boundaries cover the full token capacity. Composition owns the temporary
collapsed and normalized Attention inputs. `attention_output` remains a
caller-initialized `InOut` because Reuse writes only the active prefix while
mHC post consumes the full capacity. The Reuse case validates inactive rows
with a nonzero sentinel. `attention_hidden` and `attention_pre_mix` are fully
written `Out` boundaries. Hidden precision statistics cover active rows only;
the inactive suffix is checked independently so it cannot dilute the active
error budget.

Full attention owns compressed KV and index-key publication. Reindex consumes
the C1A cache and the layer-20 candidate mask but computes a new index query.
Reuse consumes the source layer's physical Top-K rows and has no compressor or
indexer weights. The hierarchical indexer first selects 2,048 blocks of eight
compressed positions at layer 20; later reindex layers select their final 512
positions only inside that candidate mask.
The prefill paged indexer reads packed FP4 keys and decodes each 64-key
tile inside the scoring task, without a decoded-key GM arena or separate
decode-wave dispatches. Queries are split by a 512 MiB
FP32 score budget, including Top-K row padding. Each chunk completes Top-K
and, for Full, candidate-mask selection before the next chunk overwrites the
shared score buffer. Reindex reads the supplied candidate-mask slice. The
factory's `max_logits_bytes` argument can specialize the budget; at least one
padded score row is retained. The budget covers scores, not persistent caches,
the output candidate mask, or selection scratch, and does not guarantee that
all capacity combinations fit in memory.

Each decoder C1A mode keeps its attention operator in `decode_attn_c1a_*.py` and
adds an mHC-wired `decode_c1a_*.py` entry. V4.1 staggers the coefficients:
the entry collapses with the `pre_mix` the previous sub-layer produced (one-hot
lane zero at the very first site), applies this site's `post_mix` and
`residual_mix` immediately, and hands its own computed `pre_mix` to the next
sub-layer, so it returns the new streams and that coefficient. The full entry
also hosts the shared HC fixture, goldens, and validation harness the other two
entries reuse.

The final HC collapse has no learned head parameters: it applies the last
layer's delayed `pre_mix` directly to the four residual streams. HC mixes are
depth-local values and are not persisted as sequence state.

The CPU boundary helper
[`decode_sp_integration.py`](../../../models/deepseek_v4_1_flash/decode_sp_integration.py)
models the same physical-slab owner mapping (`ceil(capacity / TP)`, padding-only
ranks included). Two *separate* checks cover the sequence-parallel hand-off:

- **Owner-slab metadata check** (CPU, `validate_two_layer_metadata`, exercised by
  `test_decode_sequence_parallel_metadata_covers_padded_and_short_batches`). It
  splits the first layer's residual output and delayed `next_pre_mix` into owner
  slabs, verifies each slab against `sequence_parallel_bounds`, asserts the
  padding contract (zero payload rows, `token_id=-1`, mask false) on *both*
  layers before anything is gathered, gathers the slabs back byte-exactly, and
  lets the second layer consume those gathered rows, so a rank-order, padding or
  delayed-coefficient slip cannot hide behind an identity round trip. This
  validates the Decode-side ABI on CPU only.
- **Block golden chain** (device path, `decode_layer.py --stage block` and
  `test_decode_sequence_parallel_two_layer_chain`). It currently **skips**, so it
  proves nothing yet: the full Block device path awaits EP8 MoE integration and
  the Block golden awaits the upstream `golden_moe` ABI (#1308). A passing
  owner-slab metadata check does not mean the Block chain passed.

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
- Ratio-2 recurrent state: `[num_state_blocks, STATE_CAPACITY, 1024]`, FP32.
  Each row stores the token's 512-channel KV projection followed by its
  512-channel gate score. `STATE_CAPACITY` is a model configuration constant
  (currently 4), independent of context length and batch size. Ratio 1 has no
  recurrent compressor state.

Compressed KV and index-key tensors for a source share the same
`c{ratio}a_cmp_kv` block table. Compressor state uses a separate engine-owned
`state_block_table` and must never use the current batch row as its physical
address. Torch goldens quantize on cache publication and dequantize on cache
reads; BF16 cache values are only an intermediate reference representation,
not the kernel ABI.

### Compressor state ownership

C2A Full prefill and decode take the same state inputs:

| Input | Contract |
| --- | --- |
| `query_start_loc` | INT32 `[B + 1]`, starts at zero; nondecreasing packed query boundaries. Equal adjacent entries represent an empty request. |
| `position_ids` | INT32 `[T]`, nonnegative absolute positions, consecutive within each valid request chunk. |
| `token_to_req_indices` | INT32 `[T]`, the current batch row `r` for every token in `[query_start_loc[r], query_start_loc[r + 1])`. This is not a persistent request ID. |
| `state_block_table` | INT32 `[B, 1]`, a stable physical state block per request; `-1` disables that request's compressor. |
| `state_cache` | FP32 `[num_state_blocks, STATE_CAPACITY, 2 * HEAD_DIM]`, an inout ring owned by the source layer. |

`num_tokens` counts the valid packed prefix and equals `query_start_loc[-1]`;
`T` may include trailing storage padding. The standalone compressor accepts
`num_tokens=0` with a positive padded tensor extent: it performs no projection,
state or output accesses. Empty requests perform no state accesses. Invalid
request/block indices are checked before table/cache indexing; negative
positions suppress publication and state writes. A live request must have a
valid allocation; an invalid block does not provide meaningful compression.
Callers disable the corresponding compressed/index cache slots for inactive
tokens as well. The metadata builder produces only the packed valid prefix. The C2A rank
drivers skip the entire sublayer when `num_tokens=0`; TP peers within a DP
group must agree on whether that group is idle.

For each valid token at absolute position `p`, the compressor computes
`block = state_block_table[token_to_req_indices[t], 0]` and ring slot
`block * STATE_CAPACITY + p % STATE_CAPACITY`. An odd-position token closes a
pair. At the start of a chunk it reads the predecessor at
`(p - 1) % STATE_CAPACITY`; within a chunk it reads the preceding projection
directly. All historical reads complete before state writes. Only the last
`min(chunk_length, STATE_CAPACITY)` rows of a chunk are written, giving each
ring slot at most one writer even when the chunk is longer than the ring.
The pooling softmax, BF16 rounding before RMSNorm, and publication parity are
unchanged.

The engine retains the same physical block while a request changes batch row.
It may release/reassign a block only after that request's outstanding state
users complete; distinct live requests cannot write the same block. Layers
2, 8 and 14 own independent state caches. `ForwardMetadata.state_block_tables`
contains a separate engine-supplied table for each source, allowing either
shared block numbering across those pools or independent allocations. No
payload is shared between source layers; reuse layers use their source's
compressed outputs and do not update compressor state.

A new request starts at position zero with no pending pair. Old bytes in a
reused block are harmless only under this contract: its own even-position
projection is written before a later call consumes it. Resuming at an odd
position requires restoring or recomputing that request/source's matching
predecessor KV and gate state. Restoring only compressed KV, or zeroing state,
is insufficient. Ring storage does not implement speculative acceptance,
rollback, or prefix-state restoration; those remain engine responsibilities.

Indexer workspaces store physical flattened cache-row ids, padded with `-1`,
so separately scheduled window and compressed sparse-attention kernels do not
depend on a concatenated-cache offset. Candidate masks remain in request-local
compressed-position space.

`ForwardMetadata` lowers packed query starts, token-to-request indices, absolute positions,
previous/new KV lengths, cache slot mappings, sliding-window indices,
per-token causal compressed lengths, per-request compressed lengths and
remainders, ragged compressor output starts, source-token rows, and compressed
RoPE positions. The same lowering serves prefill and continuous-batch decode.

The target deployment is one eight-card A5 node with TP4 attention, two DP
groups, and EP8 routed experts. TP1/2/4/8 and compatible EP2/4/8 shapes remain
available for bring-up. The EP world is reinterpreted as `DP = EP / TP`
contiguous attention groups; `tp_rank = rank % TP` and
`group_base = rank - tp_rank`.

This PR completes the decode Attention sequence-parallel boundary only. Each TP
rank owns a contiguous physical slab (`T_local = ceil(T / TP)`), including
inactive padding rows; AllGather and ReduceScatter operate on the active prefix
and preserve the padding contract. A reserved-capacity batch whose active prefix
stops short of the capacity (`active_tokens < tokens`) is follow-up work for C1A
Full/Reindex/Reuse, SWA and C2A Full; C2A Reuse is the mode that validates that
shape. The complete EP8 MoE dispatch/combine path, cross-layer token layout, and
TP4/EP8 end-to-end model run remain uncovered follow-up work.

The first implementation targets pure head tensor parallelism. Every TP rank
sees the same token batch. `wq_a`, `wkv`, compressor, indexer, and the
single-head KV caches are replicated. `wq_b`, query heads, attention sinks,
and output groups are sharded across TP ranks. Each rank computes
16 query heads and two output groups; one FP32 TP all-reduce reconstructs the
complete hidden output for standalone Attention callers. The sequence-parallel
prefill compositions below instead reduce-scatter those FP32 partials directly.
The existing owner-filtered MoE entry and the decode integration have separate
ABIs; adapting them to the standard EP boundary is tracked in
[issue #1275](https://github.com/hw-native-sys/pypto-lib/issues/1275). DSA context
parallelism is intentionally out of scope.

`prefill_swa.py` keeps mHC residuals token-sharded across TP ranks. It applies
mHC-pre and input RMSNorm locally, AllGathers the attention input, and
ReduceScatters FP32 output-projection partials before local mHC-post.
`incoming_pre_mix` and `next_pre_mix` carry the staggered mHC state between
sublayers. The standalone `prefill_attn_swa.py` retains replicated output.

The C2A compositions (`prefill_c2a_full.py`, `prefill_c2a_reuse.py`) and
`prefill_c1a_sp.py --mode {full,reindex,reuse}` use the same local residual
boundary. `attn_input` holds the gathered `[T,D]` input; `attn_output`,
`next_pre_mix`, and `x_hc_out` hold only the local contiguous token shard.
C1A exposes `make_rank(mode, epochs)` for a caller supplying real weights,
state and communication windows, and `make_program(mode, world_size, epochs)`
for a complete L3 launch. C2A exposes `prefill_c2a_full_sp`,
`prefill_c2a_reuse_sp`, and `make_hc_program`; its original inline sublayers
retain the replicated ABI used by `prefill_layer.py`.
The standalone `prefill_attn_c1a_*` and `prefill_attn_c2a_*` entries retain
replicated output. The compatibility mHC wrappers in `prefill_c1a_full.py`,
`prefill_c1a_reindex.py`, and `prefill_c1a_reuse.py` also use replicated residuals;
SP callers use `prefill_c1a_sp.py` and its local-state ABI instead.
All SP paths write active mHC-post values and inactive zeros in one producer.
Prefill stages BF16 compressor, index-weight, index-key and grouped output
projections through FP32 GM scratch before RNE narrowing. Their mixed
producers, C1A index scoring and C2A attention require Vector input before
the first Cube-to-Vector result transfer. This startup dependency addresses
the pending-producer overlap described in
[pypto #2829](https://github.com/hw-native-sys/pypto/issues/2829).
The staged producers use FP32 Vector stores; direct Acc-to-GM stores were
unstable in the concurrent A5 prefill workload.
C2A prefill uses the group-32 Q-A projection and three BF16 weight terms
for its FP32 compressor weights. A Vector task materializes those terms
in GM; at most 16 synchronously started mixed workers accumulate high
and residual products separately in 320-column K chunks. A separate
Vector task uses compensated summation across the 16 chunks to reduce
error near pooled BF16 rounding boundaries. At 4,096 tokens the staged
products occupy 256 MiB per projection, in addition to weight scratch.
These schedules cost scratch storage and extra tasks; decode keeps its
original operators. References retain FP32 head partials until the final
TP sum. Bounded head-RoPE and index permutation workers keep the
4,096-token capacity within the runtime's logical block limit.
C2A Full provisions 2 GiB per runtime heap ring for its retained compressor
products; C2A Reuse and SWA provision 1 GiB per ring for
that capacity; callers composing larger programs must size the retained live set.

All compositions require equal physical shard capacity `ceil(T/TP)` within a
TP group, including ranks with no active rows. The L3 `num_tokens` input is
`[world_size,1]`, with one group-global valid count repeated across its TP
ranks. Positions, request IDs, RoPE rows, slots, Top-K rows and cache metadata
remain in the full gathered token order. The C2A Full L3 entry gathers
RoPE rows from serving-owned full tables before the epoch loop. The caller
must pass that same mapping to the next local sublayer and give it only
`max(0, min(T_local, num_tokens - tp_rank*T_local))` active rows. Padding never
becomes a valid token. There is no residual-stream AllGather or additional TP
sum at these exits. Embedding, EP dispatch/combine, and final serving output
redistribution belong to their respective callers.

Each gather and reduction has separate retained windows and consecutive
one-based epochs. Their publish/consume handshakes include empty ranks and
empty DP groups. The current independent entry fixtures repeat a sublayer
with the same inputs while reusing windows; they are not a multi-layer model
or a real-checkpoint validation. CPU regressions separately check that the
local residual and delayed pre-mix can pass through two consecutive boundaries.

Example eight-card prefill checks (use the environment's device allocator):

```bash
python models/deepseek_v4_1_flash/prefill_c2a_full.py --tp 4 --dp 2 --tokens 7 --requests 3 --dp-tokens 7,1 --epochs 3 -d 0,1,2,3,4,5,6,7
python models/deepseek_v4_1_flash/prefill_c1a_sp.py --tp 4 --dp 2 --mode full --case mixed --dp-tokens 8,5 --epochs 3 -d 0,1,2,3,4,5,6,7
```

The C2A checks retain the existing Attention budget (relative L2 <= 1%, each
row <= 2%, cosine >= 0.9999), replay mHC-post on the actual device Attention
output, and require exact zero inactive shards and unchanged inactive cache
storage. C1A retains its existing per-row RMS and peak error bounds. Both
check every local shard independently. These layout boundaries follow the
[reference decoder](https://github.com/vllm-project/vllm-ascend/blob/36b7582431326361dd4dae79a78a555e76e715f9/vllm_ascend/models/deepseek_v41/model.py)
and [SP helpers](https://github.com/vllm-project/vllm-ascend/blob/36b7582431326361dd4dae79a78a555e76e715f9/vllm_ascend/models/common/ops/sequence_parallel.py)
at revision `36b7582431326361dd4dae79a78a555e76e715f9`.

On the decode side both wirings coexist: the historical entry keeps the
head-parallel all-reduce, and the `*_sharded` entry uses the sequence-parallel
boundary above. Weights and KV state stay replicated and query heads and output
groups stay sharded across TP ranks in both; the sharded one keeps the residual
stream rank-local. MoE token-owner de-duplication is not part of this boundary:
an EP integration on the decode side must consume the already sequence-parallel
local rows directly. Prefill keeps its all-reduce path; this PR is decode-only.

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

The paged-attention Torch reference accumulates the BF16 compressor, index-key,
index-weight, and grouped output projections in FP32. Compressor, index-key,
and grouped output results are rounded back to the activation dtype before
the next stage. The prefill C1A indexer rounds projected and scaled index
weights, QK dot products, weighted scores, and the head-reduction result to
BF16. Top-K scratch stores those rounded scores in FP32. The Torch reference
uses the same rounding boundaries.
This makes accumulation explicit rather than depending on the CPU backend's
native BF16 matrix multiplication.

C1A Full and Reindex, in both prefill and decode, temporarily order index-key
decoding after the index-weight projection completes
([pypto#2829](https://github.com/hw-native-sys/pypto/issues/2829)). On the pinned A5 stack,
a mixed projection's Cube producer can start while its paired Vector core
still executes a decoder, overwriting the decoder's UB through the local C2V
pipe. The explicit task dependency avoids this overlap at the cost of some
parallelism; it does not change the arithmetic or precision thresholds.
Reuse has no index-weight projection or index-key decoder.

For C1A prefill, the attention reference follows the kernel's 32-key online
softmax tiles, BF16 probability operand for PV, and FP32 correction of the
first 16 columns of the first head in each 16-head group. Each rank's final
projection remains FP32 through the rank-ordered TP reduction, with one BF16
cast after the sum.

C1A Full and Reindex validate Top-K eligibility, uniqueness, logical-position
ordering, trailing `-1` padding, and cutoff score quality before checking
the output. If an accepted selection differs from the nominal golden, the
output reference is recomputed for that selection using the **reference
cache values and original weights**. Device cache contents and output values
do not define this reference. Reuse uses its supplied selection directly.

Every output row (one token on one rank) must satisfy both bounds:

```text
RMS(actual - reference) <= 1e-6 + 0.01 * RMS(reference)
max(abs(actual - reference)) <= 1e-5 + 0.05 * RMS(reference)
```

Non-finite values fail. There is no global outlier quota: a bad row cannot
be diluted by other tokens or ranks, and a small number of large finite
errors cannot bypass the peak bound. The absolute floors cover near-zero
rows. Cache comparisons retain their separate quantization and ownership
checks. Saved `data/out` snapshots encode the reference arithmetic and must
be regenerated after these rounding rules change.

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

1. Keep the existing prefill Attention TP all-reduce and validate the
   decode-only sequence-parallel boundary described above. Packed prefill SWA
   remains wired through mHC: `mhc_mixes` → `mhc_pre` → attention RMSNorm →
   `prefill_attn_swa` → `mhc_post`. Run
   `python models/deepseek_v4_1_flash/prefill_swa.py`.
2. Implement C2A Full, then validate Full-to-Reuse cache and Top-K replay.
   Packed prefill C2A Full and Reuse are wired through mHC the same way,
   with the attention RMSNorm the block runs between `mhc_pre` and the
   leaf: `python models/deepseek_v4_1_flash/prefill_c2a_full.py`.
3. Implement C1A Full and the level-one candidate selector, then Reindex and Reuse.
   Run their sequence-parallel mHC composition with `prefill_c1a_sp.py --mode full`,
   `--mode reindex`, or `--mode reuse`.
4. Implement the three-phase EP-MoE dispatch/local-expert/combine body.
5. Compose the operators into one layer, then into the 40-layer prefill/decode
   token loop. `prefill_layer.py` is the prefill half of the first step: one
   complete layer, the C2A Reuse attention sublayer followed by the EP MoE
   sublayer, both through mHC and with the delayed pre-mix passed between them.
   Run `python models/deepseek_v4_1_flash/prefill_layer.py`, which validates it
   on two cards at TP1/DP2/EP2; the A5 job also runs TP2/DP2 on four cards.

Until the leaf kernels and weight loader land, this directory is not a runnable
model and is not exposed to `pypto-serving`.

The compressor ownership regression can run without devices:

```bash
python -m pytest tests/contract/test_v41_compressor_state.py -q
python tests/contract/test_v41_compressor_state.py -p a5sim
```

The second command runs two consecutive kernel calls with reordering, block
reuse, a chunk longer than the ring, inactive requests, trailing padding, and
an empty-work case. Use `-p a5 -d <allocated_device>` for the same test on a
real A5 device. Full attention validation additionally uses
`decode_c2a_full.py` and `prefill_c2a_full.py`; their fixtures use nonidentity
state block mappings.

## Pytest precision coverage

Each implemented A5 entry contains its own `test_precision` function. Declare
CI coverage with ordinary pytest parameters in that file, for example:

```python
@pytest.mark.parametrize("tp,dp", [(1, 1), (2, 2), (4, 1)])
def test_precision(tp, dp, a5_args):
    result = validate(a5_args(tp=tp, dp=dp))
    assert result.passed, result.error
```

`validate` runs the existing golden harness and returns its result, including
output and cache precision comparisons. The decode Attention entries validate
the replicated and the sequence-parallel wiring in the same call: `validate`
runs the historical all-reduce path first and the `*_sharded` path second, and
returns the failing result when either one misses. `main` calls the same
validation path for local execution. CLI choices remain independent of the
cases selected for PR CI. The standard combinations are `(tp, dp) = (1, 1),
(2, 2), (4, 1)`.
A decode entry and the `_attn_` leaf it composes are selected together, so they
split the combinations instead of repeating them: the composing entry keeps
`(1, 1)` and `(4, 1)`, the leaf keeps `(2, 2)` — or `(1, 1)` when the leaf
exposes DP1 only. Other entries supporting only DP1 or exposing only TP keep TP1
and TP4; single-card entries run once. MoE uses EP4 with TP2 or TP4,
corresponding to DP2 or DP1.
Daily CI retains its existing CLI-based Cartesian-product sweep.

The CI runner collects the tests in each selected `# ci: a5` file using pytest,
then submits each node to `task-submit` with `tp * dp` cards, or `ep` cards
for expert-parallel entries with implicit `dp = ep / tp`. The queued command
runs **pytest on that node**, passing declared `--tp`/`--dp`/`--ep` options and allocated `--device`
IDs. Each node runs in a fresh process because model shapes depend on CLI
arguments at import time. PRs retain selection by the reverse-import graph.
Changes to the pytest queue runner or this directory's `conftest.py` select
all marked V4.1 implementations. Validation entry points reject explicit TP
arguments that disagree with their import-time TP size.
Daily CI continues to invoke the local main entry points with its existing matrix.

With the development environment activated, inspect a file's tests:

```bash
python -m pytest models/deepseek_v4_1_flash/decode_attn_swa.py --collect-only -q
```

Run an individual precision case through the device queue:

```bash
task-submit --device auto --device-num 4 --run \
  'python -m pytest "models/deepseek_v4_1_flash/decode_attn_swa.py::test_precision[2-2]" --tp 2 --dp 2 --device $TASK_DEVICE -v -s'
```

The fixture checks that test parameters match the process's TP/DP/EP options and
that the allocated device count matches. For all declared cases, the CI runner
handles the separate processes and allocations (requires the CI `activate.sh`):

```bash
python .github/scripts/run_a5_pytest.py models/deepseek_v4_1_flash/decode_attn_swa.py
```

Local script execution is still supported with all original CLI options:

```bash
task-submit --device auto --device-num 2 --run \
  'python models/deepseek_v4_1_flash/decode_attn_swa.py -p a5 --tp 1 --dp 2 -d $TASK_DEVICE'
```
