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
python models/deepseek_v4_1_flash/decode_attn_c1a_reindex.py
```

The command prints `[GOLDEN] PASS` and exits nonzero when the reference fails.
Once a kernel body lands, its owner can extend the same file with the thin
`@pl.jit` entry, `build_tensor_specs()`, and device `run(...)` block.

| Workstream | Files |
| --- | --- |
| Encoder SWA | `prefill_attn_swa.py` (leaf), `prefill_swa.py` (HC orchestration), `decode_swa.py` |
| Encoder C2A Full | `prefill_attn_c2a_full.py` (leaf), `prefill_c2a_full.py` (HC orchestration), `decode_c2a_full.py` |
| Encoder C2A Reuse | `prefill_attn_c2a_reuse.py` (leaf), `prefill_c2a_reuse.py` (HC orchestration), `decode_c2a_reuse.py` |
| Decoder C1A Full | `prefill_c1a_full.py`, `decode_attn_c1a_full.py` (leaf), `decode_c1a_full.py` (HC orchestration) |
| Decoder C1A Reindex | `prefill_c1a_reindex.py`, `decode_attn_c1a_reindex.py` (leaf), `decode_c1a_reindex.py` (HC orchestration) |
| Decoder C1A Reuse | `prefill_c1a_reuse.py`, `decode_attn_c1a_reuse.py` (leaf), `decode_c1a_reuse.py` (HC orchestration) |
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
16 query heads and two output groups. Standalone Attention entries retain
all-reduce. Layer integration uses Attention leaves with `reduce_scatter=True`, followed
by [tp_ep_layer.py](../../../models/deepseek_v4_1_flash/tp_ep_layer.py):

1. Run Attention mHC pre on the replicated residual stream.
2. Call the mode-specific Attention leaf with `reduce_scatter=True`. It sums FP32
   projection partials and writes this rank's contiguous token range, with
   one BF16 cast after accumulation.
3. Call `tp_ep_layer_tail` with that shard, the original residual, and the
   Attention post/residual mix coefficients. It selects matching rows without
   SUM, runs Attention mHC post, MoE mHC pre, EP MoE, and MoE mHC post, then
   gathers the FP32 residual streams within the TP group.

Prefill SWA and C2A leaves live in `prefill_attn_*`; their `prefill_*`
mHC wrappers retain the standalone replicated-output path. Decode C1A
leaves live in `decode_attn_c1a_*`; the `decode_c1a_*` mHC wrappers retain
the replicated-output path. The layout flag is `pl.constexpr`: it is removed
from the device ABI. Direct Python compilation defaults to `False`; DSL
callers must explicitly pass `False` (AllReduce) or `True` (ReduceScatter).
Do not feed an
already all-reduced output to ReduceScatter, or apply Attention mHC post twice.

For a DP group's `T` active rows and local TP rank `r`, the range is
`width = ceil(T / TP)`, `first = min(r * width, T)`, and
`count = min(width, T - first)`. Attention, residual, and mHC coefficients
share this mapping. TP4 with eight rows gives `[0:2]`, `[2:4]`, `[4:6]`,
`[6:8]`. AllGather restores the token order and all four residual streams.

All EP ranks must allocate the same positive shard capacity
`S >= max_DP(ceil(T / TP))`, at most `ceil(PREFILL_MAX_TOKENS / TP)`.
`T` can differ across DP groups but must agree within a TP group and fit the
input capacity and the chosen Attention phase limit. Every EP rank executes
`ceil(S / MOE_TOKENS)` rounds, including empty shards. Padding does not route
or contribute to returned results; the expert kernels retain fixed block
padding. Only the first `count` shard rows and first `T` gathered rows are
defined.

Empty blocks initialize gate outputs and skip normalization and route-selection
grids, whose device launches require a positive block count. They still
participate in every EP communication round.

MoE consumes distinct rank-local tokens and no longer accepts `token_owners`
or a TP rank. `ForwardMetadata` no longer constructs `moe_token_owners`.
Combine returns routes to their source rank. Two DP groups of eight rows
produce 16 unique tokens and 96 top-six assignments.

Place communication buffers on 64-byte boundaries and reserve at least one
cache line for each packed signal buffer. The L3 tail and MoE drivers allocate
64 bytes for each signal while retaining the logical `[rank, 1]` counter view;
this prevents signal cache maintenance from touching neighboring payloads.

The shared expert stores its W2 result in FP32 GM before a separate Vector
task converts it to BF16, matching the routed expert's output pattern.
This avoids the pinned A5 local C2V startup defect
([pypto#2829](https://github.com/hw-native-sys/pypto/issues/2829)), which also
reproduces independently on the tail validation toolchain. The extra GM
traffic and task have not been benchmarked.

Window counters start at zero. Attention, MoE, and residual epochs start at
one; advance Attention/residual epochs by one per call and `moe_epoch_base`
by `ceil(S / MOE_TOKENS)`. Keep each window with its counters, and do not
overlap invocations on the same windows. Each combine counter receives
`N_LOCAL_EXPERTS` payload notifications plus one consumption notification per
round. The next dispatch waits for every rank, including itself, to consume
the preceding round before reusing windows. The separate dispatch payload
counter advances by `N_LOCAL_EXPERTS` per round.

`l3_tp_ep_layer_tail` allocates the L3 windows and launches all ranks, taking
stacked weights and one active token count per rank. It starts after
Attention ReduceScatter; callers supply the mode-specific Attention inputs
and cache metadata separately. DSA context parallelism is outside this boundary.

The complete tail and all twelve Attention modes with ReduceScatter pass A5 code
generation; the tail also passes binary compilation. A3 TP4/EP8 checks cover
reduction, mHC/residual reconstruction and byte-transport diagnostics. A5
TP2/EP4 and TP4/EP4 full-tail checks cover actual FP8/MX experts, dispatch,
combine and repeated window reuse with nonzero structured weights, unequal
token counts and empty shards. Every valid value meets the elementwise
tolerance, and TP replicas agree exactly. See [PR #1305](https://github.com/hw-native-sys/pypto-lib/pull/1305)
for revisions, cases and numerical evidence.

A5 TP4/EP8 device acceptance, integrated Attention/model execution and
performance measurements remain outstanding. Tail validation starts after
Attention ReduceScatter and does not use real checkpoint weights.

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

Full and Reindex validate Top-K eligibility, uniqueness, logical-position
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
