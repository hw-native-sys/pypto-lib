# GLM-5.3-Flash

`models/glm5_3_flash/` contains the implementation of
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (`glm5_next`,
`Glm5NextForConditionalGeneration`) on **Ascend a2a3**. It includes component
kernels, Torch goldens, W8A8 checkpoint staging, composed MoE FFN entries,
chunk-capable forward kernels, MTP computation, and a 24-block vision tower
with image/video preprocessing. PyPTO Serving owns resident generation,
request lifecycle and generation.

Each core component owns its kernels, cache operations, Torch references and
validation: `kda.py`, `attention.py`, `indexer.py`, `mtp.py`, and `vision.py`.
The MoE, expert, normalization and mHC operators compose into `decode_layer.py`
and the shared chunk-capable `layer.py`. `prefill_fwd.py` and `decode_fwd.py`
each own their complete backbone loop and L3 entries without importing each
other. Both compose the shared operators in `layer.py`, which also owns their
tensor layouts and communication-epoch bounds. `prefill_fwd.py` also owns
cache-maintenance kernels, CPU weight payload planning and independent
reference checks.
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
| `INDEX_BLOCKS_DYN` | Physical blocks backing `pool_cache` |

For example, the default decode fixture has 16 write events and 25 page-table
entries. A shared dynamic symbol requires equal extents; binding these
quantities to one symbol can underallocate the gather scratch. Its live row
count is `POOL_PAGES_DYN * INDEX_STATE_BLOCK_SIZE`.

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
| Native resident weight plan | 39.35 GiB | 20.45 GiB |
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

### Complete native forward interface

`prefill_fwd.py:l3_prefill_fwd` and `decode_fwd.py:l3_decode_fwd` each submit
one L3 orchestration for all 45 backbone layers, final normalization and the
NPU vocabulary head. They share the operators in `layer.py`. Serving owns
allocation, submission, completion and request state; cache reset uses a
separate auxiliary program.

Prefill uses `--capacity` query rows; decode uses thirty-two stable request slots.
MoE owner buffers retain `ceil(capacity / (TP * 16)) * 16` rows. Query, owner,
request and head buffers have distinct tensor contracts. Composed KDA and
sparse-attention operators derive query, request and state-pool extents from
their input tensor dimensions; active-token counts remain runtime values.
This requires PyPTO commit
[`ff2e0a4`](https://github.com/hw-native-sys/pypto/commit/ff2e0a4ec3015c7bead91ca123ead3282f09f81e)
or a later revision containing the dependent-shape parameter cloning fix from
[PyPTO #3271](https://github.com/hw-native-sys/pypto/pull/3271).

`forward_parameter_names()` defines argument order, `forward_tensor_layouts()`
provides per-rank shapes and dtypes, and `forward_geometry()` records ABI version
11 and capacities. Each invocation takes one `forward_epoch`; the four phases
own independent communication windows and counters. Decode gathers embeddings
from resident BF16 vocabulary shards on the NPU before entering the backbone.
Prefill keeps host-provided embeddings.

### Chunked prefill and incremental requests

`pypto_serving.model.glm53.npu_runner.Glm53ModelRunner.forward_chunks()`
consumes only the new tokens for each participating request. Callers pass
stable request slots and their committed
prefix lengths; returned logits follow the caller's request order. MLA latent
cache, raw/pooled Indexer cache, KDA convolution state and KDA recurrent state
remain resident between chunks. `reset_requests()` clears selected private working
rows before slot reuse. Empty request segments retain unique KDA state
rows and do not participate in sampling. Serving enables grouped prefix sharing by default.
If a dispatch or partial cache reset fails, the affected slots require a
successful reset before another chunk can use them. Host sequence lengths
commit only after a successful forward.

Serving declares an attention group for MLA/Indexer history and a rolling
KDA-state group for convolution and recurrent checkpoints. The shared grouped
allocator owns physical pages, prefix matching, references and eviction. The
runner lowers its page IDs through `CacheTables`; the library does not allocate
or index a second prefix cache.

`kda_state_rows` contains one seed row, one private working row, and a destination
row for each 16-token recurrence chunk. A negative destination skips publication.
At completed 128-token boundaries, the convolution and recurrent kernels write
the assigned checkpoint rows. A prefix hit reads those rows without modifying
them; the new request continues in its private working row. All supported
prefill/decode entries use this same metadata contract, recorded by forward ABI
version 11. Earlier runtime artifacts require recompilation.


The serving runner executes NPU final RMSNorm and full-vocabulary
projection for its 16 returned rows. The head retains BF16 weights and rounds
the GEMM result to BF16 before widening it to FP32, matching the reference
head's dtype boundary. Each rank owns a vocabulary shard, padded with zero weights to a
multiple of 64 rows: 19,392 rows at TP8 and 9,728 at TP16. The additional device weight is
317,718,528 bytes (303 MiB) per rank at TP8 or 159,383,552 bytes (152 MiB) at
TP16, excluding workspace and runtime overhead. The host strips each shard's
padding before concatenating the complete logits in vocabulary order.

`lm_head.py` retains `reference_lm_head_logits()` for independent CPU
validation. Serving runs the NPU head and concatenates its vocabulary shards
on the host. Decode embedding lookup runs on the NPU; the additional BF16 shard
occupies 158,597,120 bytes (151.25 MiB) per rank at TP8 or 79,298,560 bytes
(75.625 MiB) at TP16. Prefill embedding lookup remains on the CPU. The runner retains the full
32-row hidden readback and finiteness checks on every rank. Device weight
allocation is logged as `GLM_DEVICE_LM_HEAD_BYTES_PER_RANK`.

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
shared physical page pool, aligned to 640 tokens. Its default is the smaller
of thirty-two full request contexts and 1,048,960 aggregate tokens. A request may
grow to the configured context limit only while physical pages remain free.
Allocation reserves all cache groups for a request atomically. Request release
drops ownership; shared immutable pages remain eligible for prefix reuse. Indexer pool-close events include
only newly completed pools and may read members written by earlier chunks.
Saved artifacts include both dimensions and must be recompiled when either
changes. Serving's `Glm53ModelRunner.forward_chunks()` consumes the new tokens for
each participating request.
The validation driver `tests/glm53_generation.py` chunks prompt tokens and
then appends only new decode tokens for 1–16 requests.

The shared page allocator updates `block_table` and pooled-page mappings on each
query batch. KDA working rows remain request-local; seed and checkpoint rows are
mutable dispatch metadata. KDA constants remain fixed for the runner's open lifetime.

The report returned by the validation driver
preserves its timing boundaries: total `seconds` starts at runtime
preparation, after CPU checkpoint setup; each
step's `seconds` starts after input preparation and H2D copies. Separate
`checkpoint_setup_seconds`, step `input_setup_seconds`, and total/step
`end_to_end_seconds` retain the excluded setup costs. These are host wall times,
including dispatch, completion waits and output processing, rather than
device-only kernel times.

Serving accepts 1–32 active requests and uses the common scheduler to split
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

### Query absorption performance

`absorb_query` schedules independent heads with `pl.parallel`, preserving
compensated FP32 accumulation and the BF16 output boundary.

Standalone measurements against `b931df2`: a2a3, PTOAS 0.65, 17 query rows,
single-device TP-shard shapes, 100 timed rounds after five warmups. Each
pair reuses frozen inputs and golden outputs; times are device-wall medians.

| TP shard | Before (ms) | After (ms) | Speedup |
|---|---:|---:|---:|
| TP1, 64 local heads | 77.8978 | 17.8008 | 4.38x |
| TP2, 32 local heads | 38.9932 | 9.4154 | 4.14x |

Outputs are bitwise equal to baseline and pass the unchanged comparator.
In TP8/EP8 serving with PTOAS 0.67, absorption code generation is unchanged
and no measurable benefit is observed: C1 layer-3 rank-0 stage medians are
14.865 -> 15.095 ms for prefill and 2.471 -> 2.484 ms for decode, with a
128-token prompt. These stage intervals include dependency and scheduling
waits; standalone speedups do not establish a serving improvement.

### mHC projection performance

The narrow `mhc_linear` projection uses independent token/output-channel tasks
on the vector cores. Each task reduces 128-element K chunks and retains both
reduction parts with compensated FP32 accumulation. RMS normalization and
Sinkhorn mixing retain their existing implementation and precision checks.

At whole-model row capacities, the complete `mhc_mixes` component measures
40.833 -> 0.759 ms for 16 rows and 44.620 -> 3.174 ms for 128 rows. These are
device-effective medians on a2a3 device 0 with PTOAS 0.67, 100 timed rounds
after five warmups, replaying frozen inputs and golden outputs. All three
mix outputs are bitwise equal to the original implementation and pass the
unchanged `atol=1e-5, rtol=0.001` comparator. Component timings are not
whole-model speedups.

Whole-model generation on 2026-11-04 compares this projection against the
retained attention-optimized baseline: serving PR #286 (`b1138f4`) and lib
PR #1631 (`88c0b2d`), with identical attention changes in both variants.
The published W8A8 checkpoint runs on a2a3 devices 0-7, TP8/EP8/DP1,
PyPTO `ce0eb7c`, PTOAS 0.67 and CANN 9.0.0. Frozen prompts have 128 input
and 16 output tokens, greedy sampling, a 32-token prefill chunk, a 128-token
batch budget and prefix caching disabled. Each concurrency discards one
warmup and measures three rounds; startup and profiling are excluded.
Metrics below are host-observed serving medians, including dispatch and
synchronization, compared with the retained unprofiled baseline samples.

| Concurrency | Throughput before/after (token/s) | TTFT before/after (s) | TPOT before/after (s) |
|---|---:|---:|---:|
| 1 | 0.091105 / 0.158884 | 55.142 / 39.950 | 8.031 / 4.050 |
| 4 | 0.361711 / 0.631067 | 55.507 / 40.273 | 8.094 / 4.077 |

Throughput improves by 74.4% / 74.5%; TPOT falls by 49.6% for both cases.
All 320 generated token IDs across 20 requests, including warmups, match
the baseline. These measurements cover this prompt length and concurrency;
they do not establish performance for longer contexts or larger batches.

### KDA projection and router component performance

The initial vector implementation used shared row dots with 128-element K
reductions and compensated FP32 accumulation for KDA QKV, low-rank gates,
output projection and router logits. mHC reuses the same primitive. BF16 projection boundaries,
router TopK and all existing comparison thresholds remain unchanged.

Against `8fe4823`, a2a3 onboard measurements use TP8 shard shapes, PTOAS 0.67,
100 timed rounds after five warmups and frozen input/golden replay. Values
are median device-effective latency of the complete component; KDA uses
device 0 and routing uses device 1, with each pair on the same device.
KDA processes all capacity rows; gate cases specify capacity / active rows.
In the captured TP8 C1 workload, MoE token compaction gives gate capacity 16
with one active row for decode and four for a 32-token prefill chunk.
The 128-row gate cases below are additional coverage, not that serving shape.

| Component | Rows | Before (ms) | After (ms) | Speedup |
|---|---:|---:|---:|---:|
| KDA front-end projection | 16 | 79.531 | 10.540 | 7.55x |
| KDA front-end projection | 128 | 185.619 | 78.925 | 2.35x |
| KDA output, including normalization | 16 | 9.451 | 3.047 | 3.10x |
| KDA output, including normalization | 128 | 56.531 | 23.921 | 2.36x |
| Router, including normalization and TopK | 16 / 1 | 11.338 | 0.282 | 40.21x |
| Router, including normalization and TopK | 16 / 4 | 11.355 | 0.437 | 25.96x |
| Router, including normalization and TopK | 128 / 32 | 23.791 | 1.980 | 12.01x |
| Router, including normalization and TopK | 128 / 128 | 77.646 | 7.396 | 10.50x |

Every measured output is bitwise equal to baseline. Single-row and ragged
KDA, zero-active and ragged routing, and mHC regression checks pass. mHC
remains at 0.759 / 3.174 ms for 16 / 128 rows. Component speedups are not
serving gains; the complete generation measurements follow below.

KDA projection/output entries accept `--tp 8 --ep 8 --tokens 16` or
`--tokens 128`. The router accepts `--num-tokens 16 --active-tokens 1`;
omitting `--active-tokens` keeps all rows active. Both support `--save-data`
and `--golden-data`. Preserve each shape's snapshot separately when reusing
compiled dynamic-shape kernels.

### KDA reduction tile refinement

The follow-up uses separate tiling for the three projection stages. QKV and
first-stage gate projections use N=16, K=512 with one pipeline stage; the
second-stage gates use N=64, K=128; the final output projection uses N=32,
K=256 with two pipeline stages. The shared reduction scales its integer
accumulator by a power-of-two column bound to keep the wider reductions
within INT32 range. Single-chunk reductions omit cross-chunk compensation;
BF16 rounding and the FP32 exchange boundary remain unchanged. Each row-dot
launch uses at most 32,256 blocks; a strided in-core loop processes any
remaining work items. This bound also applies to router row dots.

The comparison below starts from the already-vectorized implementation in
PR #1697, not the earlier scalar baseline. TP8/EP8 uses capacity rows 16 and
128 from the whole-model decode/prefill workload. Both versions run on the
same a2a3 device, with PTOAS 0.67, five warmups and 100 timed rounds. The
isolated kernels reproduce the production SPMD bodies; their inputs are
frozen from the existing component fixtures, with gate intermediates derived
once using the unchanged reference. Values are median device-effective time.

| Operator or component | Rows | Before (ms) | After (ms) | Latency reduction |
|---|---:|---:|---:|---:|
| `kda_projection_qkv` | 16 | 9.109 | 7.622 | 16.3% |
| `kda_projection_qkv` | 128 | 70.946 | 59.082 | 16.7% |
| `kda_projection_output` | 16 | 0.228 | 0.224 | 1.7% |
| `kda_projection_output` | 128 | 1.614 | 1.509 | 6.5% |
| `kda_output_proj` | 16 | 3.009 | 2.647 | 12.0% |
| `kda_output_proj` | 128 | 23.733 | 20.880 | 12.0% |
| Complete KDA front-end | 16 | 10.539 | 8.413 | 20.2% |
| Complete KDA front-end | 128 | 78.901 | 65.912 | 16.5% |
| KDA output including norm | 16 | 3.047 | 2.692 | 11.6% |
| KDA output including norm | 128 | 23.914 | 21.087 | 11.8% |

Every measured output is bitwise equal to the vector baseline. TP8/TP16
single-row and ragged 17-row KDA cases, shared mHC/router callers, and exact
BF16 comparisons for zero, constant, cancellation and varying-magnitude dot
products pass. Existing golden thresholds are unchanged. KDA and router also pass at 2,048
and 8,192 rows, including TP16 KDA at 8,192 rows. Boundary cases exercise
253-row output projection, 505-row QKV and 1,793 active router rows. Router
replay comparison derives its active extent from the executed input scalar
and output shape, including when snapshot metadata overrides CLI defaults.

The earlier full-model swimlane interval for `kda_projection_output` includes
scheduling overlap with QKV. It must not be compared to these isolated timings
as if it were pure compute time. No whole-model performance run includes this
follow-up yet; the generation results below describe the prior implementation.

### KDA QKV active capacity rows

The shared KDA projection takes `query_start_loc` and uses its final offset
as the number of packed input rows for QKV. Inactive QKV capacity rows are
explicitly zeroed; active rows retain the same compensated FP32 reduction
and BF16 rounding. Both compute and padding launches respect the runtime's
32,256-block limit. Empty packs perform only the padding writes. The other
projection outputs retain their existing full-capacity computation.

Single-card TP8/EP8 measurements collected on 2026-11-05 compare against
main `eb76b3b` on a2a3 device 1, using PyPTO `ff2e0a4`, PTOAS 0.72, five
warmups and 100 measured rounds. The metric is median device-effective wall time with
profiling disabled. QKV-only cases extract the same production SPMD body,
including padding writes. The before/after pair replays identical frozen
synthetic inputs and goldens; inactive input rows are zero in this timing
fixture. Capacity 16 with one/four active rows follows the decode workload;
capacity 128 with 32 active rows covers a partial prefill chunk. Fully occupied
cases distinguish padding removal from arithmetic throughput.

| Capacity / active rows | QKV before (ms) | QKV after (ms) | QKV speedup | Complete projection before / after (ms) |
|---|---:|---:|---:|---:|
| 16 / 1 | 7.618 | 0.725 | 10.51x | 8.415 / 1.547 |
| 16 / 4 | 7.618 | 2.103 | 3.62x | 8.411 / 2.849 |
| 16 / 16 | 7.618 | 7.619 | 1.00x | 8.414 / 8.416 |
| 128 / 32 | 59.053 | 14.888 | 3.97x | 65.865 / 21.578 |
| 128 / 128 | 59.053 | 59.057 | 1.00x | 65.864 / 65.862 |

All measured outputs are bitwise equal to the baseline and saved golden.
The bounded default projection entry additionally checks single-row, empty,
padded, ragged-tail and full ragged inputs on TP8 and TP16, with nonzero
inactive input rows. Full prefill and decode composition compile successfully.
These measurements describe component latency. No whole-model performance
or generation validation has been run on the current-main PR branch.

Use `kda.py --component projection --tp 8 --ep 8 --tokens 16
--active-tokens 1` to select a capacity/occupancy pair. Omit selection flags
for bounded correctness coverage. `--save-data` and `--golden-data` operate
on one selected case; snapshots include `query_start_loc`, so snapshots from
the earlier projection signature must be regenerated.

### Combined projection whole-model performance

The combined mHC, KDA and router changes were measured on 2026-11-04 with
the same serving PR #286 / lib PR #1631 revisions, checkpoint, devices 0-7,
TP8/EP8/DP1 and toolchain as the mHC experiment above. The targeted projection
changes were applied to that retained lib checkout, preserving its forward
ABI and the identical attention optimization in both comparison variants.
Each concurrency uses the same frozen 128-token prompts, 16 output tokens,
one warmup and three measured rounds. Profiling is disabled, and the
623.59-second model startup is excluded from the serving metrics.

| Concurrency | mHC-only throughput (token/s) | Combined throughput (token/s) | Combined TTFT (s) | Combined TPOT (s) | Throughput ratio vs mHC-only |
|---|---:|---:|---:|---:|---:|
| 1 | 0.158884 | 0.556300 | 17.690 | 0.738 | 3.50x |
| 4 | 0.631067 | 2.181827 | 18.095 | 0.749 | 3.46x |

Relative to the retained attention-only baseline, throughput changes from
0.091105 to 0.556300 token/s at concurrency 1 and from 0.361711 to 2.181827
token/s at concurrency 4: 6.11x and 6.03x, respectively. TPOT falls from
8.031 to 0.738 seconds and from 8.094 to 0.749 seconds. All 320 output token
IDs across 20 requests, including warmups, match the original, attention-only
and mHC-only saved runs. The reference runs are retained measurements, not
new baseline reruns; these results cover this prompt length and concurrency.

### Earlier Serving measurements

The measurements below predate native MLA activation quantization and the
shared physical page allocator. They do not measure the current native path.

Measure one persistent Serving engine with the published checkpoint, fixed
prompt/output lengths, explicit chunk boundaries and prefix sharing disabled.
Report startup separately from warm TTFT, inter-token latency and total output
tokens per second. Host dispatch spans include runtime and synchronization
overhead and must not be presented as device-only kernel times.

Serving's `Glm53ModelRunner.reset_requests()` clears the selected private KDA
working rows before reuse. It preserves shared attention and checkpoint pages.
The initial full reset uses a separate sentinel and clears all physical storage.
Lengths commit only after the clear completes, and failed clears leave the
touched slots invalid until a successful reset.

The earlier split-program implementation kept communication windows
independent of request caches. Each resident stage kept its own invocation
count across layers, forwards and slot resets.
Dense layers used TP epochs `2*n+1` and `2*n+2`; attention halves used `n+1`;
MoE FFN halves used `n+1` for both MoE exchange and TP token restoration.
Host submissions retained at most two pending handles through the public
`DistributedWorker.submit()` API. Native FIFO execution ordered the programs,
and existing reader fences remained active. Collective counters committed only
after successful completion; pending handles drained before the vocabulary
head and output readback. This removed repeated whole-window zero restoration.
The complete-forward interface above replaces that host stage schedule.
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
| 3 | Replace the former 89-stage host schedule with complete prefill/decode orchestration; reuse scratch and metadata while retaining collective fences. | V4.1 `3652e16`, `de44572`; Serving DSpark `a816a74`, `9db614a` | Validate one fused layer before the full backbone; preserve reader-consumed fences and request-cache lifecycle at TP8/16. |
| 4 | Complete acceptance of the new FP32 device vocabulary head and decode embedding. Keep complete logits for general sampling; use compact candidates only for supported greedy execution. | V4.1 `3627f8f`; Serving DSpark `bc8a830` | Opt38 component, native and Serving measurements are complete; acceptance awaits the reference decision with existing CPU-gate differences retained. Decode embedding is integrated through the vocabulary-sharded lookup and TP reduction. |
| 5 | Pack streamed weights in dtype-correct NZ, evaluate single-use weight cache bypass and reuse attention operands. | V4 `3ff92eb`; V4.1 `a47e652` | Benchmark layout/cache policy separately, retain INT8/BF16 formats and require end-to-end wall-time improvement. |
| 6 | Revisit grouped Indexer scoring after its defect is isolated, extend context/concurrency coverage, and evaluate compiler/runtime upgrades as a separate pinned stack. | PyPTO/simpler optimization history; existing DeepSeek whole-forward compositions | Run the frozen scorer repro, operator gates and chunk/reorder/reuse matrix before comparing service performance. |

These references supply mechanisms, not projected GLM speedups. In particular,
`3ff92eb` reports an A3 DeepSeek decode result, while V4.1 `eeda7fb` and
`a47e652` report A5 component results. Their percentages do not establish a GLM
gain. The previous implementation kept weights and six stage programs
resident and ping-ponged the hidden stream. Those historical measurements do
not measure the complete-forward implementation above. Cross-request prefix
sharing uses the same grouped cache contract.

Earlier whole-forward probes reported Error104 dimension-equality failures.
The current composition distinguishes independent query, owner and head axes,
uses named dimension expressions, and specializes existing primitives for the
frozen physical extents. Complete 45-layer prefill and decode code generation
and native compilation pass on the matched toolchain. These compile checks
establish build compatibility; numerical device validation and performance
measurements remain separate gates.

### MoE expert parallelism

The complete forward selects the MoE transport at compile time. Prefill uses
`prefill_moe.py`; decode uses the same `decode_moe.py` implementation as the
standalone decode MoE and FFN entries. The shared `layer.py:moe_layer` retains
mHC mixing, token ownership and TP output restoration for both phases.

The resident forward skips the complete shared expert on ranks with no
locally owned source tokens. Such ranks still receive and compute routed
experts for peers and participate in EP exchange and TP token restoration.
The local add and scatter publish zeros without reading inactive shared
scratch. This preserves active-row arithmetic and complete shared weights.

`decode_layer.py` and `layer.py` process each rank's local token rows.
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
scratch. `layer.py` adds the capacity-sized Gate buffers, padded shared
expert buffers and FFN intermediates through `prefill_layer_ring_heap()`.
Both helpers retain a 16 MiB allocation reserve and round larger requirements
up to a power of two. At 64 rows per rank both entries keep the 256 MiB default
for EP2/EP4 and select 512 MiB for EP8 or 1 GiB for EP16 on ring zero. An enclosing
forward must additionally budget its own live allocations and pass sufficient
runtime capacity; see [runtime ring sizing](../../debug-and-tune/ring-heap-and-scope-stats.md).

### Routed expert measurements

`expert_routed.py` pairs the INT8 Gate/Up projections around a resident L1
input tile, uses 128-column projection fragments and groups two W2 output
tiles per epilogue block. Activation blocks publish partial row maxima for
the quantizer, and W2 projection tiles expose completion independently to
their consumers. The prefill heap budget includes the maxima cache.
The routed-expert implementation shared by prefill
and decode retains its 16-row tiles and existing rounding and quantization
boundaries.
The standalone entry supports `--save-data` and `--golden-data` for replay.
Its input and weight/scale pairs share lazy initializers, so frozen replay
constructs specifications without generating random activation or weight tensors.

The measured pypto-lib revisions are:

- Mainline baseline: [`521058090d2792cc2e6344c6c0ca1e670c9291c5`](https://github.com/hw-native-sys/pypto-lib/commit/521058090d2792cc2e6344c6c0ca1e670c9291c5).
- Paired-projection baseline: [`e02a7c039713a0e0610bdcf768e15f48ce6ec3fb`](https://github.com/hw-native-sys/pypto-lib/commit/e02a7c039713a0e0610bdcf768e15f48ce6ec3fb).
- Optimized device kernel: [`9c79b577207f3bbf0f31e94b36cc5f5b127bee3b`](https://github.com/hw-native-sys/pypto-lib/commit/9c79b577207f3bbf0f31e94b36cc5f5b127bee3b).

All three measurements used this matched toolchain:

| Component | Revision / version |
| --- | --- |
| PyPTO | `ce0eb7c07fad364b6b169ee99a55950283e60dae` |
| simpler runtime | `d8f997fabd1b6988b69a94feef61ab669dda0eea` |
| PTOAS | `0.67` |
| PTO-ISA | `b1d460ee8c001585ede36b68ba3678dc5a226bbf` |
| CANN | `9.0.0` |
| PyTorch | `2.9.1+cpu` |

On A3 device 0, with hidden size 4096, intermediate size 2048 and 18 local
experts, single-card `effective_us` medians were measured over 100 rounds
after 5 warmups. Each optimized case replayed the corresponding mainline
frozen inputs and golden outputs and passed the existing comparator.
Rows below are received routed rows distributed across the local experts.

| Received rows | Mainline (us) | Optimized (us) | Latency reduction |
| ---: | ---: | ---: | ---: |
| 1 | 81.45 | 74.65 | 8.3% |
| 17 | 481.92 | 407.79 | 15.4% |
| 96 | 507.81 | 431.31 | 15.1% |
| 512 | 878.12 | 706.64 | 19.5% |

Relative to the paired-projection baseline (`e02a7c0`), independent W2
completion and cached activation maxima reduce these component medians by
3.5% to 5.1%.

These component measurements do not establish serving throughput gains.

### Vocabulary operators

`lookup_embedding.py` gathers from one BF16 vocabulary shard using global token
IDs and replicates each row into the four BF16 mHC streams. IDs outside that
shard contribute exact zeros. The caller sums these contributions within its TP
group before entering the backbone. `tp_lookup_embedding` composes this lookup
with the existing FP32 TP reducer and mHC seeding. The complete decode forward
uses its own TP windows and counts the embedding reduction as the first
collective of each invocation. Inactive query rows are zeroed before seeding.

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

## Validation

`decode_fwd.py` exports the complete production decode L3 entry. Component validation stays with the operators in `kda.py`,
`attention.py`, `indexer.py`, `dense_mlp.py`, `decode_layer.py` and `layer.py`,
using their Torch references. `prefill_fwd.py --check reference` provides the
independent CPU backbone checks; its default invocation checks weight loading
and tensor contracts.

Production multi-step decode, request reordering, slot reuse and prefix
restoration are exercised through the serving runner and the complete L3
programs described in the incremental request interface above.

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
stacks and unstacked residents (`norm.final_norm` and `mtp.*`)
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
Routed experts are EP-sharded by expert ID, and the shared expert is TP-sharded
along its intermediate dimension. Attention follows the existing head partition; the indexer
remains replicated. Prefill and decode can reuse the same staged weights.

Shared-expert INT8 gate/up and down matrices are packed into NZ order once
per layer at resident upload. The fused gate/up matrix is packed after fusion;
INT8 uses 16-row blocks and 32-element C0 lines. Checkpoint-reader tensors remain
logical ND, and shared FP32 scales, routed expert weights and Dense weights remain
ND. Rank-stacked L3 arguments use logical ND handles so host orchestration
selects an integer rank; each rank-local shared matrix retains its NZ compute
contract and consumes those same packed bytes. The 43 stored MoE slices include
the reserved MTP tail; the backbone executes 42 MoE layers.
Standalone/default/checkpoint FFN fixtures pack the same shared
inputs, and their goldens only undo the layout before the existing reference math.
Runtime manifest version 11 records the complete prefill/decode ABI, phase
geometries, head backend, shared INT8 NZ and native MLA contracts; recompile
older runtime directories before using these uploaded weights.

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
python models/glm5_3_flash/layer.py -p a2a3 -d "$TASK_DEVICE" --ep 2 \
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

## Standalone generation and visual inputs

Execution is provided by `pypto-serving`; install that package and set
`PYPTO_LIB_ROOT` to this checkout. The standalone command uses the same
`Glm53ModelRunner` and compiled-kernel preparation as the serving worker.
`prefill_fwd.py --check weights` and `--check reference` remain independent
CPU checks in this repository.

MTP operators and their standalone Torch references remain in `mtp.py`.

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
The validation driver with image/video inputs
expands image/video markers and replaces the corresponding embedding rows
before seeding all four streams.

The standalone validation driver accepts `--image`, `--video` and repeated
`--video-frame` inputs. Visual inputs attach to prompt zero; the Python API
supports multiple requests and ordered visual inputs. `--vision-runtime-dir`
loads a separately compiled single-rank vision artifact. Build CPU artifacts
before scheduling NPU execution, following the
[compile/runtime workflow](../../run-and-validate/compile-runtime-workflow.md).

```bash
python /path/to/pypto-serving/tests/glm53_generation.py --compile-only \
  --tp 16 --ep 16 --capacity 2048 --max-seq-len 1048576 \
  --cache-tokens 1048576 --runtime-dir /path/to/runtime

# Run inside the matching device allocation.
python /path/to/pypto-serving/tests/glm53_generation.py --tp 16 --ep 16 \
  --capacity 2048 --max-seq-len 1048576 --cache-tokens 1048576 \
  --runtime-dir /path/to/runtime --checkpoint /path/to/checkpoint \
  --prompt "Describe the image." --image /path/to/image.png \
  --vision-runtime-dir /path/to/vision-runtime
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
