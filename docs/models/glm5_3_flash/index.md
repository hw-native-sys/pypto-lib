# GLM-5.3-Flash

`models/glm5_3_flash/` contains the implementation of
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (`glm5_next`,
`Glm5NextForConditionalGeneration`) on **Ascend a2a3**. It includes component
kernels, Torch goldens, W8A8 checkpoint staging, composed MoE FFN entries,
chunk-capable forward kernels, MTP computation, and a 24-block vision tower
with image/video preprocessing. PyPTO Serving owns resident generation,
request lifecycle and greedy MTP verification.

Each core component owns its kernels, cache operations, Torch references and
validation: `kda.py`, `attention.py`, `indexer.py`, `mtp.py`, and `vision.py`.
The MoE, expert, normalization and mHC operators compose into `decode_layer.py`
and `prefill_layer.py`. `prefill_fwd.py` exports the forward programs, cache-maintenance kernels,
tensor layouts, CPU weight payload planning and independent reference checks.
The serving runner owns device allocation, cache lifecycle, runtime dispatch
and generation. `config.py` contains model configuration; `utils.py` contains
checkpoint reading and shared host helpers.
Symbolic tensor dimensions are declared alongside the kernels that use them.

The component entries run their bounded cases by default. Use `--component`
to select a stage, such as `kda.py --component convolution` or
`attention.py --component prefill`; existing case and checkpoint options apply
to that stage. `prefill_fwd.py` runs CPU weight checks without a checkpoint;
`--check reference` runs the independent CPU backbone checks.

## Why W8A8, and why a2a3 forces it

The released checkpoint is FP8 blockwise (`e4m3`, 128x128, dynamic activations).
The a2a3 cube cannot execute it: `Intrinsic_mmad` in
CANN's `Ascend910_9392.ini` SoC config offers only `s32s8s8` / `u32u8u8` / `u8s8` and the
fp16/fp32 forms, with no fp8 or MX entry. INT8 is therefore the only quantized
path on this platform, and the deployment weights are the msmodelslim W8A8
conversion that vLLM Ascend serves on a 16-card A3 node. The int8 cube fractal
is MKN `16, 32, 16`, so every quantized K tile must be a multiple of 32.

## Checkpoint shape

| Property | Value |
| --- | ---: |
| Hidden size | 4,096 |
| Backbone layers | 45 (+1 MTP layer) |
| Linear-attention (KDA) layers | 34 |
| Sparse-attention (NoPE MLA + kpool indexer) layers | 11 (+ the MTP layer) |
| Attention heads | 64 |
| MLA `q_lora_rank` / `kv_lora_rank` | 1,536 / 512 |
| MLA `qk_head_dim` / `v_head_dim` / `qk_rope_head_dim` | 256 / 256 / **0** |
| Indexer heads / head dim / `index_topk` / `index_kpool` | 32 / 128 / 2,048 / 4 |
| KDA heads / head dim / conv kernel | 64 / 128 / 4 |
| Routed experts / active experts / shared | 288 / 8 / 1 |
| Dense layers / dense intermediate | 3 / 12,288 |
| MoE intermediate | 2,048 |
| Hyper-connection width / Sinkhorn iterations | 4 / 20 |
| Vocabulary | 154,880 |
| Maximum position | 1,048,576 |

The attention schedule is three KDA layers then one sparse layer, repeating:
layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 and 43 are sparse, every other
backbone layer is KDA. The first three layers (all KDA) use a dense MLP;
the remaining 42 and the MTP layer are MoE. Checkpoint layer 45 is the MTP
layer: an MLA layer with its own indexer and a sparse MoE, reached through
`enorm` / `hnorm` / `eh_proj`, and it carries **no** `hc_*` weights, so it uses a
plain residual rather than the four-stream mHC.

Summing every parameter group from the checkpoint index gives 321.2 G, which
matches the published 320 B total, so the reconstruction above is complete.

## What differs from DeepSeek-V4, which is the nearest port in this repo

`Glm5NextTextHyperConnection` subclasses `DeepseekV4HyperConnection` unchanged, so
the mHC coefficient algebra is identical to `models/deepseek_v4_1_flash/hc.py`.
Three things are not:

- **The hyper-connection stream is BF16.** It is seeded by replicating the BF16
  embedding (`inputs_embeds.unsqueeze(2).expand(-1, -1, 4, -1)`) and every layer
  casts the mixes into that dtype. DeepSeek-V4 carries the stream in FP32, so the
  tile dtypes differ throughout and a kernel cannot be lifted across unchanged.
  At width 4 this is 32 KB of live residual per token instead of 64 KB.
- **The final collapse is an unweighted mean**, not a learned head applying the
  last layer's delayed pre-mix, so `mhc_head` takes no mix input.
- **The text backbone has no RoPE.** `qk_rope_head_dim` is 0 and the config validator
  rejects any other value, so the rope/nope split disappears from the main
  attention — and the indexer has none either, despite `indexer_rope_interleave`
  being set in `config.json`. That field is inherited from the GLM-MoE-DSA base and
  is dead here: `Glm5NextTextConfig` sets `rope_parameters = AttributeError()`,
  `Glm5NextTextModel.forward` passes `position_embeddings=None`, and
  `Glm5NextTextIndexer.forward` never touches a cos/sin table. So this port needs
  none of the `rope_tables.py` / interleaved-rope machinery that
  `deepseek_v4_flash_mtp` carries.

Two more differences bite at the kernel level: the router scores with `sigmoid`
where DeepSeek-V4.1 uses `sqrt(softplus(.))`, and 288 experts do not fit the
`SCORE_PAD = 256` sort window that `models/deepseek_v4_flash_mtp/gate.py` uses —
the pad has to grow to 512 and the `pl.sort32` + `pl.mrgsort` merge tree gains a
level.

## Indexer dimensions

The indexer uses separate dynamic symbols for independent pool dimensions:

| Symbol | Extent |
| --- | --- |
| `POOL_EVENTS_DYN` | Pool-write events in `pool_token_slots`, `pool_slots`, and `pool_valid` |
| `POOL_PAGES_DYN` | Entries in the gathered `pool_blocks` page table |
| `POOLS_DYN` | Columns in `index_scores` |
| `INDEX_BLOCKS_DYN` | Physical blocks backing `pool_cache` and `pool_scale` |

For example, the default decode fixture has 16 write events, 1024 page-table
entries, and 4096 score columns. A shared dynamic symbol requires equal
extents; binding these quantities to one symbol can underallocate the gather
scratch. Its live row count is `POOL_PAGES_DYN * INDEX_STATE_BLOCK_SIZE`.

## Deployment target

The resident production forward supports TP8/EP8/DP1 on eight logical NPU
devices and TP16/EP16/DP1 on sixteen logical devices. TP and EP overlap on the
same device group. A dual-chip A3 board exposes two logical devices. The
checkpoint's vLLM Ascend recipe uses sixteen ranks; smaller component-only
bring-up shapes remain available through `--tp` and `--ep`.

| Per-device geometry | TP8 / EP8 | TP16 / EP16 |
|---|---:|---:|
| MLA heads / KDA heads | 8 / 8 | 4 / 4 |
| Dense intermediate width | 1,536 | 768 |
| Routed experts | 36 | 18 |
| Native resident weight plan | 39.20 GiB | 20.38 GiB |
| MoE padded owner rows at packed capacity 576 | 80 | 48 |

The packed batch has 16 stable request slots for either device count. Each MoE source
owns interleaved token rows, padded to `ceil(capacity / (TP * 16)) * 16` rows.
Shared experts remain complete and replicated. KDA projection tasks cover at
most 512 channels, and KDA output normalization tasks cover at most 64 head
rows, so the wider TP8 shapes fit the local buffers. Use separate compilation
directories and processes for each topology; dimensions freeze at import time.

The **indexer is replicated, not head-sharded**: its score sums
over all 32 heads before the top-k, so sharding would force a cross-rank reduction
of partial scores on every sparse layer, and the kernel is small enough that 16-way
redundant compute is the cheaper trade. Weight plans exclude caches, activations,
communication buffers, and upload temporaries. TP8/EP8 has been exercised with
the published W8A8 checkpoint at packed capacity 576 on eight 64-GiB A3 devices;
larger capacities require their own memory and numerical validation.

### Chunked prefill and incremental requests

`pypto_serving.model.glm53.npu_runner.Glm53ModelRunner.forward_chunks()`
consumes only the new tokens for each participating request. Callers pass
stable request slots and their committed
prefix lengths; returned logits follow the caller's request order. MLA latent
cache, raw/pooled Indexer cache, KDA convolution state and KDA recurrent state
remain resident between chunks. `reset_requests()` clears only the selected
slots before release or reuse. Empty request segments retain unique KDA state
rows and do not participate in sampling. Prefix sharing is opt-in in Serving.
If a dispatch or partial cache reset fails, the affected slots require a
successful reset before another chunk can use them. Host sequence lengths
commit only after a successful forward.

`Glm53ModelRunner(..., prefix_snapshot_entries=2)` enables two immutable
device snapshot banks; the default 0 allocates none. `capture_prefix(slot,
(bank, generation, length))` copies all six request-cache families after an
aligned prompt boundary completes. `restore_prefix()` copies a confirmed
descriptor into a reset slot and publishes its committed length only after
successful completion. Generations must increase on capture; stale or
unaligned descriptors are rejected. A copy failure poisons runner execution and
requires close/reopen. Serving owns snapshot admission, in-flight leases,
worker acknowledgement and eviction.

The two banks retain 59.664 MiB per rank at TP8/context 512 or about 42.066 MiB
at TP16/context 512, without reducing the sixteen live slots. The constructor
checks the full payload against `prefix_snapshot_max_bytes` before allocation
(default 128 MiB per rank); runtime overhead is additional. Larger contexts
need an explicit budget and their own memory/numerical validation. Snapshot
copies use an auxiliary typed program registered before worker preparation;
the six production programs and auxiliary programs are recorded in layout
manifest version 6 by the standalone serving command. Earlier manifests
require recompilation.

The serving runner defaults to NPU final RMSNorm and full-vocabulary
projection for its 16 returned rows. BF16 hidden rows and final norm weights
are widened to FP32; normalized activations, projection weights and logits
remain FP32. Each rank owns a vocabulary shard, padded with zero weights to a
multiple of 64 rows: 19,392 rows at TP8 and 9,728 at TP16. The additional device weight is
317,718,528 bytes (303 MiB) per rank at TP8 or 159,383,552 bytes (152 MiB) at
TP16, excluding workspace and runtime overhead. The host strips each shard's
padding before concatenating the complete logits in vocabulary order.

`Glm53ModelRunner(..., lm_head_backend="cpu")` or
`GLM53_LM_HEAD_BACKEND=cpu` selects the CPU reference. That backend retains
the complete widened FP32 head, adding 2,537,553,920 bytes (about 2.36 GiB) of
host RAM alongside the loader's BF16 table. Embedding lookup remains on the
CPU. Both head backends retain the full 32-row readback and finiteness checks
on every rank. The selected backend and device weight allocation are logged
as `GLM_LM_HEAD_BACKEND` and `GLM_DEVICE_LM_HEAD_BYTES_PER_RANK`.

An optional `Glm53ModelRunner(..., host_num_threads=16)` sets the
dedicated owner process's PyTorch CPU thread count after all weight uploads
complete. Startup and rank workers retain their initial setting. Close and
entry-error cleanup restore the previous owner count. The default `None`
preserves the existing setting. The process-wide option assumes one exclusive
owner runner; it does not configure device AICPU threads.

Under the frozen TP8/EP8 A3 workload (128 input tokens, eight output tokens,
32-token chunks, one warmup and three timed groups), selecting sixteen owner
threads raises concurrency 1/2/4 throughput from 0.953/1.890/3.699 to
1.054/2.094/4.081 tokens/s. Startup stays at four threads and is excluded.
All 25 saved request outputs remain bitwise identical.

The query `--capacity` and per-request `--max-seq-len` are independent compiled
dimensions. Query capacity bounds the sum of new chunk tokens; context capacity
bounds logical request lengths. `--cache-tokens` independently bounds the
shared physical page pool, aligned to 128 tokens. Its default is the smaller
of sixteen full request contexts and 1,048,576 aggregate tokens. A request may
grow to the configured context limit only while physical pages remain free.
Allocation fails atomically when the shared pool is exhausted. Reset clears
owned payloads and scales before releasing pages for another request. Indexer pool-close events include
only newly completed pools and may read members written by earlier chunks.
Saved artifacts include both dimensions and must be recompiled when either
changes. Serving's `Glm53ModelRunner.forward()` resets an independent batch
of sixteen nonempty requests.
`pypto_serving.model.glm53.generation.generate()` chunks prompt tokens and
then appends only new decode tokens for 1–16 requests.

The shared page allocator updates `block_table` and `pool_blocks` on each
query batch. KDA request rows and the KDA constants remain fixed for the
serving runner's open lifetime. Prefix capture copies logical rows through
the page table; restore reserves destination pages before copying the
immutable snapshot.

The report returned by `pypto_serving.model.glm53.generation.generate()`
preserves its timing boundaries: total `seconds` starts at runtime
preparation, after CPU checkpoint setup; each
step's `seconds` starts after input preparation and H2D copies. Separate
`checkpoint_setup_seconds`, step `input_setup_seconds`, and total/step
`end_to_end_seconds` retain the excluded setup costs. These are host wall times,
including dispatch, completion waits and output processing, rather than
device-only kernel times.

Serving accepts 1–16 active requests and uses the common scheduler to split
prefill into chunks, sample completed prompts, batch incremental decode and
release finished/cancelled requests. Configure the new-token budget with
`--max-num-batched-tokens`, the per-request context with `--max-model-len`, and
the prompt chunk limit with `--long-prefill-token-threshold`. Serving keeps
nonterminal prompt chunks aligned to the KDA recurrence's 16-token boundary,
deferring unfinished work when the remaining budget is too small. The final
prompt tail can have any length, and decode still appends one token. Direct
low-level callers can choose other partitions, but these change the existing
BF16 KDA rounding boundaries; numerical comparisons must use the same
partitions in the kernel and reference paths.
Convolution bulk rows, request-head rows and decode use the same sequential
four-tap FP32 accumulation before SiLU and BF16 conversion. Reassociating the
request-head sum can cross a BF16 midpoint at a new chunk boundary and amplify
through later quantized layers. Aligned internal KDA chunks therefore also
require strict agreement with the whole-prefix execution.

Indexer scoring uses the same per-query implementation in prefill and decode.
The previous four-query scorer could produce zero valid scores on repeated A3
runs with fixed inputs and private pool segments. The per-query path preserves
the original score formula, quantization, TopK ordering and owning numerical
checks; changing attention index order would not correct those score values.

### Earlier Serving measurements

The measurements below predate native MLA activation quantization and the
shared physical page allocator. They do not measure the current native path.

Measure one persistent Serving engine with the published checkpoint, fixed
prompt/output lengths, explicit chunk boundaries and prefix sharing disabled.
Report startup separately from warm TTFT, inter-token latency and total output
tokens per second. Host dispatch spans include runtime and synchronization
overhead and must not be presented as device-only kernel times.

Serving's `Glm53ModelRunner.reset_requests()` clears selected request caches
with one distributed device operation. It preserves unselected slots byte
for byte; resetting all sixteen slots also clears unused physical KDA rows
and pooled Indexer scratch.
Lengths commit only after the clear completes, and failed clears leave the
touched slots invalid until a successful reset.

On TP8/EP8 A3, query capacity 128 and context capacity 512, alternating the
device reset and the frozen host-copy implementation on the same allocations
gave the following synchronous API medians (three warmups, ten timed rounds):

| Reset scope | Host copies (ms) | Device clear (ms) |
|---|---:|---:|
| First slot | 923.099 | 1.853 |
| Last slot | 929.052 | 1.360 |
| Two sparse slots | 1850.579 | 1.394 |
| All slots and physical padding | 6767.180 | 2.041 |

These timings include host dispatch and completion waiting. Exact-byte NPU
checks cover all six cache families at TP8 and TP16, including unselected
NaN/Inf payloads, repeated resets and full physical clearing. The TP8 full-model
chunk/reorder/decode/reuse matrix also retains bitwise-identical logits.

Communication windows persist independently of request caches. Each resident
program keeps its own invocation count across layers, forwards and slot resets.
Dense layers use TP epochs `2*n+1` and `2*n+2`; attention halves use `n+1`;
MoE FFN halves use `n+1` for both MoE exchange and TP token restoration.
Host submissions retain at most two pending handles through the public
`DistributedWorker.submit()` API. Native FIFO execution orders the programs,
and existing reader fences remain active. Collective counters commit only
after successful completion; all pending handles drain before the vocabulary
head and output readback. Errors drain returned handles and invalidate
runner execution.
This removes repeated whole-window zero restoration.
It does not change expert sharding or add TP slices to shared experts.

A failed dispatch invalidates the communication state for the whole
`Glm53ModelRunner`. Close and reopen the serving runner before further
forwards or request resets.
Old scalar ABIs and exhausted INT32 notification counters are rejected.
Two-pass chip-swimlane replay is also rejected for these stateful runners.
Real-weight TP8 and TP16 chunk/decode/reuse checks pass; the TP8 disposal probe
additionally completes one MoE dispatch, injects a host failure before epoch
publication, and verifies that all later state operations stop before native
API calls.

The earlier retained FP32 CPU LM-head weight preserved all 25 saved request outputs
bitwise. Under the frozen TP8/EP8 A3 Serving workload (128 input tokens, eight
output tokens, 32-token chunks, concurrency 1/2/4, one warmup and three timed
groups), output throughput rises from 0.843/1.681/3.292 to
0.909/1.803/3.489 tokens/s versus the preceding output-budget step. Startup is
excluded. A separate full-FP32 NPU head trial failed nine of 4,956,160 values
at the unchanged all-point `rtol=atol=1e-5` gate and was rejected.

The new opt38 head performs final normalization on the NPU and uses complete,
padded vocabulary tiles. Its TP8 and TP16 real-weight component checks pass all
2,478,080 values (16 rows by 154,880 logits) at the unchanged all-point
`rtol=atol=1e-5` CPU gate, with identical top-1 IDs. Native full-network and
Serving measurements are complete. Native hidden rows match the CPU path;
token histories and top-1 IDs match in both measurements.
The native comparison has 3 of 4,956,160 values outside the existing CPU gate;
Serving has 11 of 12,390,400. Independent FP64 diagnostics pass all native
values at the same numeric limits. Acceptance remains pending a reference
decision; the CPU gate has not been relaxed and opt38 is not recorded as accepted.

The following groups describe reference mechanisms and their numerical
acceptance requirements.

| Group | GLM work | DeepSeek reference | Acceptance |
|---|---|---|---|
| 1 | Sweep chunk sizes 32/64/128, bound async tail dispatch, batch request-cache clearing, remove obsolete query headroom, and compute only active CPU head rows with a retained FP32 head. | Serving DSpark `e114182`, `a816a74`; V4 `eec9235` | Retain the frozen baseline; preserve token IDs, request reuse, chunk alignment and budget limits; measure both client latency and throughput. |
| 2 | Profile and optimize the 42 MoE FFN transactions: separate device work from communication/window-reset/runtime costs, then specialize W8A8 tiles and active-row work. GLM already uses direct expert slabs and reverse-window combination. | V4 `e3673ca`; V4.1 `eeda7fb` | Keep original operator gates, complete shared experts and EP8/16; accept changes only when whole-step wall time improves. |
| 3 | Fuse attention and FFN submissions to target 89 to 47 launches first, reuse scratch/metadata slots, and retire collective signals before reducing window resets. | V4.1 `3652e16`, `de44572`; Serving DSpark `a816a74`, `9db614a` | Validate one fused layer before the full backbone; preserve reader-consumed fences and request-cache lifecycle at TP8/16. |
| 4 | Complete acceptance of the new FP32 device vocabulary head, then integrate device embedding. Keep complete logits for general sampling; use compact candidates only for supported greedy execution. | V4.1 `3627f8f`; Serving DSpark `bc8a830` | Opt38 component, native and Serving measurements are complete; acceptance awaits the reference decision with existing CPU-gate differences retained. Device embedding is not integrated. |
| 5 | Pack streamed weights in dtype-correct NZ, evaluate single-use weight cache bypass and reuse attention operands. | V4 `3ff92eb`; V4.1 `a47e652` | Benchmark layout/cache policy separately, retain INT8/BF16 formats and require end-to-end wall-time improvement. |
| 6 | Revisit grouped Indexer scoring after its defect is isolated, extend context/concurrency coverage, and evaluate compiler/runtime upgrades as a separate pinned stack. | PyPTO/simpler optimization history; existing DeepSeek whole-forward compositions | Run the frozen scorer repro, operator gates and chunk/reorder/reuse matrix before comparing service performance. |

These references supply mechanisms, not projected GLM speedups. In particular,
`3ff92eb` reports an A3 DeepSeek decode result, while V4.1 `eeda7fb` and
`a47e652` report A5 component results. Their percentages do not establish a GLM
gain. GLM currently keeps weights and six programs resident and ping-pongs the
hidden stream, so those existing features are baseline behavior. A whole
45-layer L2 fusion previously exceeded this compiler's dimension-proof limits;
the submission work starts with bounded combinations. Cross-request prefix
sharing uses the opt-in snapshot contract described above.

The pinned PyPTO revision `6a7c862` also rejects bounded attention-plus-MoE
compositions with Error104 dimension-equality checks in KDA recurrence and
sparse attention. Single-attention controls compile. Both a combined L2 entry
and a paired-L2 L3 entry reproduce the failure, so the 89-to-47 submission
target requires compiler support before it can be measured.

### MoE expert parallelism

The resident forward skips the complete shared expert on ranks with no
locally owned source tokens. Such ranks still receive and compute routed
experts for peers and participate in EP exchange and TP token restoration.
The local add and scatter publish zeros without reading inactive shared
scratch. This preserves active-row arithmetic and complete shared weights.

`decode_layer.py` and `prefill_layer.py` process each rank's local token rows.
Routed experts are distributed by expert ID across EP ranks; every local expert
retains its full intermediate dimension. The shared expert is a complete
replicated FFN on each rank, with no TP weight slices or output reduction.
Each rank adds its local shared output to the routed output returned by EP
combine. Token batches may differ across ranks.

The standalone MoE entries select `--ep` independently of attention TP. Their
fixtures use distinct rank-local activations and replicated weights for the
router, RMSNorm and shared expert. The generated routing fixture fixes eight
global experts spread across EP ranks to bound receive capacity; checkpoint mode
uses the saved router and correction bias. The common
configuration defaults attention TP to the selected EP size when `--tp` is
omitted; explicitly configured attention topologies retain their own checks.

Callers that hold a replicated attention stream must assign each token to one
MoE source rank before dispatch and restore their required output layout after
MoE. Those token-layout operations belong to the surrounding composition.

Prefill callers must also size the runtime heap for the receive capacity and
padded expert tiles. `prefill_moe.py` uses `prefill_ring_heap()` for routed
scratch. `prefill_layer.py` adds the capacity-sized Gate buffers, padded shared
expert buffers and FFN intermediates through `prefill_layer_ring_heap()`.
Both helpers retain a 16 MiB allocation reserve and round larger requirements
up to a power of two. At 64 rows per rank both entries keep the 256 MiB default
for EP2/EP4 and select 512 MiB for EP8 or 1 GiB for EP16 on ring zero. An enclosing
forward must additionally budget its own live allocations and pass sufficient
runtime capacity; see [runtime ring sizing](../../debug-and-tune/ring-heap-and-scope-stats.md).

### Vocabulary operators

`lookup_embedding.py` gathers from one BF16 vocabulary shard using global token
IDs and replicates each row into the four BF16 mHC streams. IDs outside that
shard contribute exact zeros. The caller sums these contributions within its TP
group before entering the backbone.

The `lm_head` greedy operator selects the hidden rows requested by `row_indices`, applies the
loaded final RMSNorm, rounds the normalized inputs to BF16, and projects against
one BF16 vocabulary shard with FP32 accumulation. It returns each selected row's
maximum score and **shard-local** token index; ties choose the lowest index.
A `-1` row index produces `(-inf, -1)` and contributes no sampling candidate.
Other row indices must lie within the input rows.

The caller adds the vocabulary-shard offset to valid local indices and selects
the global maximum within its TP group, breaking ties by the lowest global ID.
These local operators do not perform TP communication. The LM-head Torch golden
returns the full local FP32 logits; its executable adapter selects the requested
rows and compares the maximum and local index.

The separate `lm_head_logits` operator accepts BF16 hidden rows and final norm
weights, performs FP32 RMSNorm without an intermediate BF16 cast, and projects
against one FP32 vocabulary shard. Its FP32 output includes padding to a
multiple of 64 vocabulary rows. The caller removes that padding and concatenates
TP shards to obtain complete logits for general sampling. The thin
`lm_head_logits_test` entry and `golden_lm_head_logits` reference use this
full-logits contract; the original greedy operator and its BF16 normalization
boundary remain unchanged.

`lm_head_tp_reduce` combines the local LM-head candidates through TP windows. It
adds each rank's vocabulary offset, selects the maximum score and resolves ties
to the lowest global token ID. Invalid local IDs, NaN scores and `-inf` scores
contribute no candidate; an all-absent row returns `(-inf, -1)`. Its notification
epochs start at one and increase whenever a caller reuses the same windows.
This collective returns a global argmax candidate; sampling remains a caller
responsibility. Embedding contributions can use the existing `tp_all_reduce`
on one FP32 stream; the caller replicates the reduced BF16 embeddings into the
four identical mHC streams.

The LM-head executable includes bounded prefill/decode and boundary cases.
Validation defaults to TP2/EP2 on two devices and
also checks the TP reducer against its Torch golden. Three different input
epochs reuse the same windows and notification counts; cases cover every winning
peer, lowest-global-ID ties, all-absent rows, invalid local IDs and NaN/`-inf`
scores. With EP larger than TP, each TP group is checked independently.
Local projection cases and the distributed case run in separate processes.
Use `--case tp_reduce` for the collective alone, or a local case such as
`--case decode` with one device. `--compile-only` skips device execution:

```bash
python models/glm5_3_flash/lm_head.py -p a2a3 -d 0,1 --compile-only
```

## Decode backbone validation

`models/glm5_3_flash/decode_fwd.py` composes prepared BF16 embeddings into
normalized hidden states. Layers 0–2 use KDA and Dense MLP; layer 3 uses DSA
and MoE. Each sublayer has mHC processing and TP reduction. MoE dispatches the
replicated batch from rank 0, runs EP experts, and restores replicated outputs.
Embedding lookup, LM head, sampling, MTP and serving integration are outside
this entry's scope.

The fixture has 16 requests, four available layers, two decode steps and
TP2/EP2. The default runs layer 0 only. State-row mappings vary between steps;
cache pages use non-identity mappings. Convolution, recurrent and indexer state
persist across the steps. Real-weight validation remains outside CI. The
four-layer/four-step TP2/EP2 and 45-layer/two-step TP16/EP16 fixtures pass the
independent propagated CPU golden on a2a3 hardware.

Run with two allocated devices from the repository root:

```bash
# Default: one layer, two steps, independent CPU golden.
PYTHONPATH=. python models/glm5_3_flash/decode_fwd.py -p a2a3 -d 0,1 --tp 2 --ep 2

# Explicit four-layer propagation.
PYTHONPATH=. python models/glm5_3_flash/decode_fwd.py -p a2a3 -d 0,1 --tp 2 --ep 2 --end-layer 4

# Isolate the DSA/MoE layer using the prepared input fixture.
PYTHONPATH=. python models/glm5_3_flash/decode_fwd.py -p a2a3 -d 0,1 --tp 2 --ep 2 --start-layer 3 --end-layer 4
```

`--start-layer` does not execute the omitted prefix. `--compile-only` compiles
without using devices. `--dense-dtype bf16` selects the optional BF16 Dense
path; the default uses dynamic W8A8 Dense and MoE operations.

`--checkpoint /path/to/checkpoint` stages the published indexed W8A8 weights
and checkpoint embeddings for token IDs starting at 2000. This fixture starts
with empty caches and recurrent state and keeps each request's state row fixed
across decode steps. It retains the independent reference and all existing
precision gates. Use `--steps 4 --end-layer 4` to include pooled-cache writes
in the real-weight fixture. `--golden-data` replays a saved input/reference
snapshot, and `--runtime-dir` selects a compatible compiled artifact.

The independent CPU golden preserves BF16 activation boundaries, INT8
activation quantization, INT32 accumulation and persistent state updates.
The reference rounds independent FP64 reductions, BF16 dot products and
selected unaries to their FP32 operation boundaries. Device kernels use
compensated FP32 arithmetic for KDA, MLA, mHC, TP and MoE reductions, then
preserve the official BF16 activation boundaries. KDA convolution evaluates
SiLU with one FP32 division before BF16 conversion. Dynamic quantization
consumes the BF16 RMSNorm result. Sparse attention rounds each block's
exponential weights to BF16 before PV while keeping the softmax denominator
in FP32. CPU and device accumulation may still differ.

`--validate-boundaries` additionally supports local diagnosis using captured
sublayer inputs. It checks the surrounding mHC edges, final normalization,
written cache rows and KDA states. TP replicas and untouched state rows must
match exactly. This mode does not establish independent multilayer accuracy.
The four trace fields are Attention output, FFN output, FFN input and normalized
Attention input. Their buffers are separate to preserve asynchronous reads.

The published-W8A8 fixture passes all nine output groups on a2a3 with 16
requests: 45 layers and two continuous steps at TP16/EP16, and four layers
and four continuous steps at TP2/EP2. The final-hidden outlier ratio is 0%
at every tested step, below the unchanged 5% limit at `atol=rtol=0.02`.
State/cache gates, untouched rows and replica equality also pass; the
four-step case includes nonzero pooled-cache payload and scale writes.
Dense and MoE retain native INT8 weights and INT32 GEMMs; attention weights
are staged as BF16. These are correctness results; the compensated
arithmetic has not been performance-benchmarked.

Shared indexer kernels live in `indexer.py`; `decode_indexer.py` and
`prefill_indexer.py` own their respective composition and validation. Decode's
empty-owner path retains communication dependencies while skipping empty
normalization and quantization work. GM view writes in MLA and sparse
attention avoid loop-carried aliases. These changes are required by the
composed execution path.

## What is quantized

Activations are quantized **dynamically per token** the way the a2a3 W8A8 sibling
does it: each row's amax, floored by `INT8_AMAX_EPS = 1e-4`, is rescaled to
`INT8_SCALE_MAX = 127`, rounded through int32 and fp16 to match device rounding.
The router kernel produces that INT8 view once and both the shared expert and the
EP dispatch payload reuse it.

`quant_model_description.json` and the tensor dtypes determine how a checkpoint
is loaded. The AscendV1 W8A8 checkpoints used for integration have the following
formats; a floating-point MLA projection is also accepted when its metadata and
stored dtype agree.

| Tensor | Format |
| --- | --- |
| Routed/shared/dense FFN `gate_proj`, `up_proj`, `down_proj` | INT8 with per-output-channel scales; staged as INT8 plus flattened FP32 scales |
| MLA `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `o_proj` | INT8 plus FP32 scales in the resident production path; activations dynamically quantized at each projection |
| MLA `kv_b_proj`, KDA/indexer projections, norms, router, embedding and LM head | BF16 |
| mHC function, scale and base; router correction bias | Source BF16; the function stays BF16, while scale, base and correction bias are staged as FP32 to match the kernels |
| MLA 512-latent cache | BF16 |
| Indexer state cache | FP32 today, 256 wide per token; the sibling port quantizes its indexer cache to INT8 with a per-row FP32 scale, which would save about 2.3 KB per token here |
| Hyper-connection stream | BF16 — see above, this is where GLM departs from DeepSeek-V4's FP32 stream |
| Logits | FP32 |

The checkpoint LM-head table remains BF16. The resident NPU head stages each
vocabulary shard as exactly widened FP32 values with zero-padded rows.

The loader requires `W8A8_DYNAMIC` with `group_size: 0`. For every requested
quantized weight it checks INT8 storage, positive finite scales, and zero offsets.
It rejects incompatible formats rather than casting FP8 or BF16 values to INT8.
The resident path explicitly selects `mla_mode="native"`, retains all four
INT8 MLA projection matrices and their scales, and quantizes BF16 activations
per row before INT32 cube products. `kv_b_proj` remains BF16. The separate
BF16 adaptation remains available for existing absorbed decode fixtures; it
is not the resident production path.

### Rotation assets

The quantization description can include `is_rot_used` and an optional QuaRot
rotation map. The loader preserves that metadata and exposes the indexed
`rot.weight` through `Checkpoint.rotation()` on request. Optional files need not
be loaded to stage the 45-layer text backbone. Converted weights are used as
stored; no additional global rotation is applied to the backbone activations.
The MTP `eh_proj` already contains the conversion's block-diagonal rotation;
its input is not rotated again. MTP embedding and vocabulary head tensors
are read independently, including checkpoints whose tables differ from the
backbone. `Checkpoint.vision_weights()` loads the complete BF16 visual tower.

The indexer's Hadamard-128 is a separate transform. The loader supplies the same
matrix as the indexer golden; it is not the checkpoint's hidden-size rotation.

## Loading weights

`ResidentPrefillWeights.iter_rank_uploads(rank)` yields
`(stack_name, cpu_tensor, byte_offset)` payloads. Layered norm, mHC, router,
KDA, MLA, Indexer and MoE stacks arrive one layer slice at a time. Dense
stacks and unstacked residents (`norm.final_norm`, `mtp.*` and `hadamard`)
each arrive as a whole payload at offset 0. The serving runner allocates
the destination tensors and performs device copies; the lib payload planner
opens no device worker or execution context.

`models/glm5_3_flash/utils.py` provides an indexed CPU reader. It opens only
requested tensors and TP slices instead of materializing all checkpoint shards.
The checkpoint configuration must match the compiled GLM dimensions and layer
schedule. Quantization metadata, shapes, dtypes and topology are checked before
the returned weights reach a kernel.

```python
from models.glm5_3_flash.utils import Checkpoint

checkpoint = Checkpoint("/path/to/vllm-ascend-a3-w8a8")
layer = checkpoint.layer(3, tp_rank=0, tp=16, ep_rank=0, ep=16)
vocabulary = checkpoint.vocabulary(tp_rank=0, tp=16)

# Load and consume one layer at a time when full residency is unnecessary.
for layer_id, weights in checkpoint.iter_layers(tp_rank=0, tp=16, ep_rank=0, ep=16):
    ...

# Or retain all 45 layers and the layer-external weights for this rank.
model = checkpoint.model(tp_rank=0, tp=16, ep_rank=0, ep=16)
```

`layer` groups `input_norm`, `attn_hc`, `ffn_hc`, `attention`, `ffn`, and an
`indexer` for sparse-attention layers. `model["layers"]` is keyed by layer ID;
its other entries are `embedding`, `lm_head`, `final_norm`, `vocab_start` and
`vocab_stop`. Embedding and LM-head rows are TP-sharded; the final norm is
replicated. Dense FFNs are TP-sharded along their intermediate dimension.
Routed experts are EP-sharded by expert ID, and shared experts remain complete
on every EP rank. Attention follows the existing head partition; the indexer
remains replicated. Prefill and decode can reuse the same staged weights.

Shared-expert INT8 gate/up and down matrices are packed into NZ order once
per layer at resident upload. The fused gate/up matrix is packed after fusion;
INT8 uses 16-row blocks and 32-element C0 lines. Checkpoint-reader tensors remain
logical ND, and shared FP32 scales, routed expert weights and Dense weights remain
ND. The 43 stored MoE slices include the reserved MTP tail; the backbone executes
42 MoE layers. Standalone/default/checkpoint FFN fixtures pack the same shared
inputs, and their goldens only undo the layout before the existing reference math.
Runtime manifest version 6 records the shared INT8 NZ and native MLA contracts; recompile older
runtime directories before using these uploaded weights.

On A3, frozen shared-expert cases with 16 and 17 rows retained bitwise-identical
FP32 outputs. With five warmups and 100 timed rounds, median effective device
latency changed from 67.60 to 59.27 us and 78.16 to 65.35 us, respectively.
Separate real-weight TP8 and TP16 resident regressions retained all 25 and 29
saved output vectors bitwise. These layout comparisons preserve the reference
math and acceptance thresholds.

The existing FFN validation entries keep generated weights as their default.
Add `--checkpoint` to load real weights into the same kernels, goldens and
precision checks:

```bash
# Run these inside a two-device allocation; TASK_DEVICE comes from the allocator.
python models/glm5_3_flash/decode_layer.py -p a2a3 -d "$TASK_DEVICE" --ep 2 \
  --checkpoint /path/to/vllm-ascend-a3-w8a8 --layer 3 --num-tokens 3
python models/glm5_3_flash/prefill_layer.py -p a2a3 -d "$TASK_DEVICE" --ep 2 \
  --checkpoint /path/to/vllm-ascend-a3-w8a8 --layer 3 --tokens 64 --num-tokens 3
```

These entries execute the MoE FFN half of a layer, so `--layer` must be 3–44.
By default, checkpoint mode mixes deterministic synthetic activations with the
selected layer's real FFN mHC weights. To supply activations from another stage,
use `--hidden hidden.pt`, containing a CPU BF16 tensor `[EP, active_tokens, 4096]`
after mHC pre-mixing and before RMSNorm. The active token count is inferred from
that tensor unless supplied explicitly; it must fit the entry's capacity.
This component fixture is not a complete model forward or a text-generation test.

## Prior art, and what it is actually worth

None of the references below is drop-in code for this repo, but two of them are
worked designs for the hardest kernels on our exact platform.

| Reference | Where | What it gives us |
| --- | --- | --- |
| Chunked KDA in pypto, tuned on Ascend910B3 | cann-recipes-infer, `integration/vllm/ling-3.0-flash/npu_patch/.../ops/pypto/kda/chunk_kda_impl.py` | A complete chunked delta-rule forward at `T=4K, H=4, K=V=128` — our per-rank TP16 shape — with a design write-up covering the M1-M4 stages, the 8x8-leaf block inverse, and the two bottlenecks whose fix was worth 3.8x |
| Recurrent KDA in pypto | its `fused_recurrent_kda_impl.py` sibling | The decode-step counterpart |
| Lightning indexer, MLA prolog, hc_pre, sparse attention in pypto | cann-recipes-infer, `ops/pypto_python/impl/` | Algorithm and tiling references for the indexer and MLA items |
| vLLM Ascend `glm5next` | `vllm_ascend/models/glm5next/` | The only NPU port of this architecture: module decomposition, cache specs, W8A8 wiring, MTP. **It does not implement the kpool indexer** — `sparse_attn_indexer_kpool.py` raises and says the upstream is CUDA-only |
| CANN recipes GLM-5 / GLM-5.2 | cann-recipes-infer, `models/glm_5{,_2}/` | Ascend-native MLA + DSA + MoE with a 16-rank W8A8 A3 config. No KDA and no mHC |
| GLM-4.5 pypto operators | the open-source pypto mirror, `models/glm_v4_5/` | MoE gate, expert selection and the shared-expert W8A8 patterns |

The two pypto references above target a **different pypto frontend**
(`pypto.frontend.jit` / `pypto.Tensor` / `pypto_impl` / auto-tiling). The pypto
this repo pins exports only `jit`, `language`, `ir`, `runtime` and friends, so
those files are design references, not code to copy: the port is from implicit
tiling to the explicit `pl` tile DSL.

## Platform envelope

| a2a3 (`Ascend910_9392`) | Value | a5 for contrast |
| --- | ---: | ---: |
| AIC / AIV per die | 24 / 48 | 36 / 72 |
| `PLATFORM_MAX_BLOCKDIM` / max cores | 24 / 72 | 36 / 108 |
| UB (safe) | 192 KB (~184 KB) | 248 KB |
| L1 / L0A / L0B / L0C | 512 / 64 / 64 / 128 KB | — |
| HBM per die | 64 GiB (~61.3 GiB visible) | — |
| fp8 / MX `mmad` | **none** | supported |
| 4-bit in-core dtypes | **not supported** | FP4 supported |
| `Acc -> Vec` and `Vec -> Mat` memory edges | **absent** | present |
| AIC/AIV cross-core pipe | through GM | on-chip |
| GM access granularity | 512 B | 128 B |
| bf16 atomic add into GM | supported | not supported |

Some of these change how a kernel is written rather than how fast it runs. The
missing `Acc -> Vec` edge forces a GM round trip where a5 would fuse. INT8 can only
reach GM through a Vec tile or the quantized fix-pipe path. And not in the table but
worth knowing before the first MIX kernel: a2a3 requires dual-AIV dispatch even for
a no-split MIX kernel, or the AIC cross-core handshake deadlocks.

### What stream B's `a2a3sim` bring-up surfaced

Each of these was a compile or simulator failure, not a guess:

- **The simulator needs a real GCC 15.** A `g++-15` shim pointing at GCC 13 builds
  simulator kernels that get every BF16 matmul wrong while FP32 still passes, so the
  failure looks like a kernel bug. `deepseek_v4_flash_mtp/decode_compressor_ratio4.py`
  is a quick canary: it passes in CI and fails under such a shim.
- **Do not mix the Tensor and Tile levels.** `pl.slice`, bracket slices, `pl.full`,
  `pl.matmul_acc` and slice assignment are Tensor level; `pl.load`, `pl.tile.full`,
  `pl.create_tile` and `pl.store` are Tile level. A Tensor-level result is written with
  `dest = pl.assemble(dest, pl.set_validshape(src, rows, cols), offset)`, a Tile-level
  one with `pl.store`; `pl.write` works on a Tensor-level on-core buffer where
  `pl.tile.write` does not.
- **`tmp_tile` is Tile-only.** `pl.row_sum` / `pl.row_max` on a Tile must take one;
  on a Tensor they must not. High-precision `rsqrt` on a Tile is
  `pl.tile.rsqrt(x, tmp)`, not `high_precision=True`.
- **FP32 reductions need a 32-byte column**, and a row-major tile needs a 32-byte row,
  so an `[H, 1]` column wants `H % 8 == 0` and is best built as `[1, H]` then
  reshaped.
- **A matmul is only split into L0 tiles when M reaches 16.** Below that the
  128 x 512 KV operand of the sparse attention stays whole and overflows the 64 KB
  right buffer; the TP16 attention pads its 4 heads to 16 zero-filled rows for this.
- **No mutable scalar flags across branches.** A Python int set in one `if` and read in
  another becomes a phi the backend cannot materialise; write each branch's work in
  place, as the donors do. Loop-carried tile state goes through `pl.yield_` on every
  path.
- **In a mixed cube/vector scope, no branch may decide whether loop-carried state is
  updated.** When the carried tiles flow through an `if` / `else`, the partitioner
  materialises their pre-loop `pl.full` initialisation in the cube half as well, and
  `ccec` rejects `vector_dup` / `set_vector_mask` for `dav-c220-cube`. `a2a3sim`
  compiles both halves with `g++` and never sees it, so this reaches CI as an a2a3-only
  failure (run 35555578466). Let the branch pick the block's inputs and keep the merge
  unconditional, or split the scope by hand with `pl.split_aic` / `pl.split_aiv` as
  `deepseek_v4_flash_dspark/prefill_sparse_attn.py` does. The check needs no card:
  compile with `-p a2a3 --compile-only` and assert that the generated
  `kernels/aic/*.cpp` carries no vector op between `#if defined(__DAV_CUBE__)` and its
  `#endif` (`TMUL` there is address arithmetic and is expected).
- **`pl.spmd` takes a Var, not a call.** Bind `pl.tensor.dim(...)` to a name first.
- **Script entries put the repository root first on `sys.path`** to resolve
  model imports and the shared `golden` harness when invoked by file path.

## MTP proposals and visual inputs

Execution is provided by `pypto-serving`; install that package and set
`PYPTO_LIB_ROOT` to this checkout. The standalone command uses the same
`Glm53ModelRunner` and compiled-kernel preparation as the serving worker.
`prefill_fwd.py --check weights` and `--check reference` remain independent
CPU checks in this repository.

`Glm53ModelRunner(..., enable_mtp=True)` prepares checkpoint layer 45.
After each target chunk, call `update_mtp(next_token_ids)` with the next input
for each active request. It shifts the query token IDs left within each
request while retaining target positions, uses post-final-norm target
feedback, and zeros the position-zero embedding. The plain residual block
uses its own MLA, MoE, embedding, shared-head norm and vocabulary head.

`propose_mtp(request_slots, count)` returns one to three draft IDs per request.
The first proposal computes its own indexer TopK; later steps reuse that
request's selection and recompute K/gate writes and tail expansion. Recycled
feedback uses the MTP shared-head norm exactly once. Speculative cache lengths
roll back before target verification.
`pypto_serving.model.glm53.generation.generate(..., spec_tokens=3)` verifies
each draft against the target greedy token and discards the remaining queue
on rejection. Verification is sequential: the target KDA state advances only
through confirmed tokens. This mode does not claim speculative speedup.

`VisionProcessor.image()` accepts an RGB uint8 tensor, Pillow image or file.
`video()` accepts sampled RGB frames with timestamps; `video_file()` samples
with FFmpeg/ffprobe at the configured rate in stored frame orientation,
ignoring rotation metadata. Pillow is required for image files. Preprocessing preserves the checkpoint's resize budget, CLIP
normalization, temporal patch pairing and spatial merge-group order.
The standalone serving command executes patch embedding, all 24 noncausal
frame-segmented attention/MLP blocks, downsampling and the merger on an
allocated NPU.
Position-only FP32 RoPE constants are prepared with the input metadata and
gathered on device. The attention reference rounds independent FP64
exponentials, sums and reciprocals at the FP32 boundaries; compensated
reductions retain those boundaries in the device implementation.
`pypto_serving.model.glm53.generation.generate(..., visual_inputs={0: [visual]})`
expands image/video markers and replaces the corresponding embedding rows
before seeding all four streams.

The `pypto-serving` standalone generation CLI accepts `--spec-tokens`,
`--image`, `--video` and repeated `--video-frame` inputs. Visual inputs attach to prompt zero; the Python API
supports multiple requests and ordered visual inputs. `--vision-runtime-dir`
loads a separately compiled single-rank vision artifact. Build CPU artifacts
before scheduling NPU execution, following the
[compile/runtime workflow](../../run-and-validate/compile-runtime-workflow.md).

```bash
python -m pypto_serving.model.glm53.generation --compile-only \
  --tp 16 --ep 16 --capacity 2048 --max-seq-len 1048576 \
  --cache-tokens 1048576 --runtime-dir /path/to/runtime

# Run inside the matching device allocation.
python -m pypto_serving.model.glm53.generation --tp 16 --ep 16 \
  --capacity 2048 --max-seq-len 1048576 --cache-tokens 1048576 \
  --runtime-dir /path/to/runtime --checkpoint /path/to/checkpoint \
  --prompt "Describe the image." --image /path/to/image.png \
  --vision-runtime-dir /path/to/vision-runtime --spec-tokens 3
```

Component entries retain generated inputs by default and accept checkpoint
weights through the same kernels and independent Torch references:

```bash
python models/glm5_3_flash/attention.py --component projection --mode native --checkpoint /path/to/checkpoint
python models/glm5_3_flash/attention.py --component output --mode native --checkpoint /path/to/checkpoint
python models/glm5_3_flash/mtp.py --checkpoint /path/to/checkpoint
python models/glm5_3_flash/lm_head.py --case fp32 --checkpoint /path/to/checkpoint
python models/glm5_3_flash/lm_head.py --case mtp_fp32 --checkpoint /path/to/checkpoint
python models/glm5_3_flash/vision.py --component forward --checkpoint /path/to/checkpoint
```

The full-logits head reference preserves FP32 normalization boundaries and
rounds independent FP64 dot products to FP32. Its all-element acceptance gate
remains `atol=rtol=1e-5`; CPU FP32 GEMM reduction error is not used as the
expected value. Whole-model generation and long-context capacity require
separate device validation in the intended topology and memory budget.
