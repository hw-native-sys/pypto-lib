# DeepSeek V4 Decode Optimization

This page is a case study rather than a reference. It follows
[`models/deepseek_v4_flash_mtp/`](../../models/deepseek_v4_flash_mtp/) — a
43-layer DeepSeek V4-Flash build with MTP speculative decoding, W8A8
quantization, three attention paths, and a 256-expert MoE — from its first
kernels to its current state, and records which levers moved the number, which
did not, and what each one cost.

The mechanisms themselves live elsewhere:
[Performance Tuning](performance-tuning.md) for how to measure and capture,
[Cube Tile Tuning](cube-tile-tuning.md) for choosing tiles,
[Dependencies and Scheduling](dependency-and-scheduling.md) for the task graph
and the scheduler, and [Precision Tuning](precision-tuning.md) for thresholds
and rounding. Read those for *how*; read this for *in what order, and what to
expect*.

Numbers in parentheses are pypto-lib pull requests, kept so a claim can be
traced back to the change and its measurement. Every measurement quoted is the
one the change's author reported at the time.

## The shape of the work

Five groups of work, and they are not interchangeable. Each one only pays off
once the previous one is solid.

| # | Question it answers | Typical per-change gain |
|---|---|---|
| 0. Contracts, golden, per-operator precision | Can any later claim be believed? | none — this is the admission ticket |
| 1. General levers: partitioning, parallelism, tiling, fusion, mix kernels | Is each kernel doing the minimum work on the maximum cores? | 10–60 % |
| 2. Operator-specific rewrites | Is the algorithm itself the wrong shape for this hardware? | 10–40 % on that operator |
| 3. Scheduling | Are the right tasks issued in the right order? | 2–15 % |
| 4. Serving integration and lowering | Is time being spent outside the kernels entirely? | 2–8 %, plus host-side milliseconds |

The gains shrink as you go down, but the later ones cannot be reached early:
section 3 is invisible while a kernel is still 8× over-fetching its weights, and
section 4 is pointless while the golden cannot certify a rewrite.

---

## 0. Contracts, golden, and per-operator precision

The opening stretch of this work produced no performance change at all. That
was correct. Almost every optimization below alters numerical ordering, so
without a trustworthy reference none of them can be accepted.

### Derive the operator split from the reference model

The kernel boundaries are not invented. They are read off the official
HuggingFace **DeepSeek-V4-Flash** torch implementation — the modeling code that
ships with the checkpoint is the specification, and each kernel entry
corresponds to a span of it worth scheduling as one unit.
[config.py](../../models/deepseek_v4_flash_mtp/config.py)'s `FLASH` preset
mirrors that checkpoint's `config.json` field for field, so a shape or
hyper-parameter question is answered by the reference rather than guessed.

Three things follow from partitioning along the reference's own structure:

- **Each golden becomes a short transcription** of a torch function that already
  exists, instead of a re-derivation. That is what keeps a golden trustworthy
  while the kernel beside it is rewritten five times.
- **The reference fixes what a kernel is allowed to fuse.** A boundary that
  exists in the torch code because the two sides have different shapes or dtypes
  is real; one that exists only for readability is a fusion candidate.
- **The checkpoint is also the fixture source.** The synthetic tensors the
  standalone harnesses generate are calibrated against real-weight statistics
  from that checkpoint — see the fixture comments in the compressor, indexer and
  expert modules — so a kernel is exercised at the distribution it will actually
  see. See
  [Precision Tuning](precision-tuning.md#8-test-with-real-weights-and-matched-data-distribution).

### Freeze the deployment contract first

`config.py` is a per-directory singleton every kernel imports as a bare sibling
module: batch, speculative token count, page size, the per-layer attention
schedule, the quantization layout. It is a compile-time constant source, so it
is not a runtime option — when a sibling build needed a different speculative
token budget it got its own directory rather than a flag.

Decide before writing kernels:

- **Token shape.** `DECODE_BATCH × DECODE_SEQ` — here 4 requests × 2 rows
  (MTP verifies one draft token) = 8 token rows per step. This single number
  drives every parallelism decision in section 1.
- **Quantization layout.** W8A8: INT8 weights with FP32 per-output-channel
  dequant scales; activations quantized **dynamically per token** at the INT8
  matmuls, amax floored by `INT8_AMAX_EPS` and rescaled to `INT8_SCALE_MAX`.
  No calibration data, no static activation scale.
- **What stays wide.** The inter-layer hyper-connection hidden state, the
  compressor states, and all dequant scales stay FP32.

The quantization choice pays off later in unexpected places: because symmetric
per-token INT8 quantization is invariant to a positive scalar, the gate router
can defer its `inv_rms` past the quantization step entirely (section 2.5).
Contracts that look like bookkeeping decide which fusions are legal.

### Keep the golden's computation order identical to the kernel's

This is the standing requirement, not a per-case judgement:

> The golden must compute the same thing **the same way** — identical operation
> order and identical dtype at every step.

Floating-point arithmetic is non-associative and lossy at every narrowing, so an
algebraically equivalent reorder still changes the last bits. A golden that
follows torch's natural order while the kernel accumulates in tile order does
not report a tolerance; it reports noise, and that noise hides the real error
the moment one appears. The mechanics are in
[Precision Tuning](precision-tuning.md#2-make-the-kernel-and-golden-implementations-identical).

The consequence for optimization work is the rule this section exists to
establish:

> When a kernel reorders for performance, the golden is rewritten to that
> order — and only after proving that what changed was the *order*, not the
> *semantics*.

The clearest case is the zero-gather attention rewrite (section 2.1). It
required the reference to be rewritten into the kernel's physical block order,
because a BF16 PV accumulation is not associative. That was accepted only
because the new order was first shown equivalent to the old one in FP32, to
~1e-7 across identity, partial, overlay and rotated fixtures (#629). The
compressor's widened pooling tile (section 2.4) is the same discipline applied
in the cheap direction: the reordering was proven bit-identical up front, so the
golden did not have to move at all.

---

## 1. General levers

Everything in this section transfers to any kernel. Ordered by what paid most.

### 1.1 Kill serial vector reductions

The first-order cost on a decode path is rarely arithmetic. It is a reduction
running on one vector lane while everything waits.

| Change | Effect |
|---|---|
| Serial `attn_norm_rms` reduce → 2-way partial sums plus a final reduce | the original bottleneck (#339) |
| Fold `qr_quant_amax` into `qr_norm_apply` — each task also writes a partial amax | residual scope 30 µs → 1.7 µs (#339) |
| Split `hc_head`'s sum-of-squares over K slices, each writing its own row; consumer sums and applies rsqrt inline | the `inv_rms` buffer disappears entirely (#822) |

Two details worth copying:

- **`PARTIALS=2`, not 4 or more.** Two-way keeps the FP32 addition
  deterministic, which is what keeps a downstream tensor validating across
  devices. More parallelism here buys little and costs reproducibility.
- **Fold the amax into the pass that already reads the data.** The partial amax
  is computed on the same normalized tile the original scope would have re-read
  from GM, so it is bit-identical and the quantization scale is unchanged.

These levers were the core of the series that took the projection family from
1868 µs to 545 µs (−70.8 %) — a series that also widened K tiles and deepened
the pipeline (sections 1.2 and 1.6).

### 1.2 Tiling: align to the cache line before tuning anything else

The single largest structural win in the tree came from a tile that was too
narrow to fill a cache line.

`qr_proj` / `kv_proj` used an N-tile of 32. Each weight row read therefore
filled **64 B of a 512 B L2 line — an 8× over-fetch**. Moving to split-K
(zero-seed plus atomic-add) allowed N-tile 256. Together with a RMSNorm+RoPE
fusion, the projection's decode case went end to end from **936 µs to 407 µs
(−56 %)** — a2a3 swimlane, 5-run median (#578).

The follow-up pushed `QPROJ_MM_N_TILE` to 1024 — full lines, zero weight
over-fetch — and *reduced* the K tile from 512 to 128 to buy an 8-slice
non-degenerate `stage=2` pipeline, lifting L0C occupancy from 25 % to 50 %.
`qproj_matmul` went 56.3 µs → 36.0 µs **with no change in task count** (#718).

The walls that bound this, all hit repeatedly (see
[Cube Tile Tuning](cube-tile-tuning.md#model-the-three-practical-constraints)):

| Wall | Value | What it forced here |
|---|---|---|
| L0C accumulator | `TM*TN*4 ≤ 128 KB` | `TN=512` needs `TM=64`; measured no faster end to end |
| Vector UB | 192 KB | `hc_pre`'s fused scope had to drop `D_TILE` 512 → 256 |
| `alloc_tile` minimum | 32 B | a `[1,1]` FP32 reduction result is 4 B and is **rejected** |
| CANN template | — | `Q_PROJ_OUT_CHUNK=256` triggered `ACL_ERROR_RT_AICORE_TIMEOUT` |

The 32 B floor is worth internalizing, because it shows up whenever a reduction
collapses to a scalar. The workaround is always the same: allocate a physical
tile that clears the floor and mark only the rows you use as valid — a
`[8, FFN_D_TILE]` tile with `[1, FFN_D_TILE]` valid for a per-row norm (#784),
or an `[8, 808]` grid view for a vocab-wide max scan (#985).

### 1.3 Raise parallelism: find the second axis

With 8 token rows and 48 vector cores, `pl.spmd(T)` uses one sixth of the
machine. Every parallelism win in this tree is the same move — **find a second
independent axis and fan over the product**.

| Operator | Original grid | New grid | Result |
|---|---|---|---|
| `merge_norm` | `spmd(T=8)`, serial 4-iteration head-tile loop inside | `spmd(T × head-tile) = 32` | fills one wave (#651) |
| `hc_head_reduce` | one task — one core streamed 512 KB while 23 idled | (token-tile × D-slice) | 58.8 → 42.6 µs (#822) |
| inverse RoPE | per-token rebuild | head-parallel + `ROPE_OUT_TILES` | AIV occupancy 67 % → 89 % (#525) |
| MoE dispatch / combine | one `pl.at(CORE_GROUP)` — 2 cores of ~72 | `pl.spmd(N_RANKS × N_LOCAL)` | (#705, #736) |
| `ffn_norm` | whole-tensor | one token per core | 12.6 → 6.2 µs (#784) |
| dispatch + combine scatter | single-core | parallel | span 1514 → 1028 µs (#473) |

**More blocks is not monotonically better.** Two calibrated stopping points:

- `ROPE_OUT_TILES=4` (256 tasks) over-tiles — per-task work approaches dispatch
  cost and the RoPE span balloons to 107 µs. The default is 2 (#525).
- `merge_norm` uses 32 blocks, not 64: 64 needs two waves on 48 cores for no
  additional work in flight (#651).

### 1.4 Merge operators: a scope boundary is a cost

Each scope boundary is a GM write, a GM read, and a dispatch. Collapsing them
is often worth more than tuning what is inside.

- `hc_pre`: **five `pl.spmd` scopes fused into one**, deleting four GM
  intermediates that existed only to bridge them (#533).
- The indexer's RoPE went **four-in-one** — slice matmul → cos/sin apply →
  assemble matmul → write, in a single `CORE_GROUP` scope: ~1357 → ~1094 µs
  (−19 %) (#401).
- **18 M-axis pad/unpad scopes deleted** across the decode pipeline (#653). The
  cube is row-independent in M, so a `valid_shape` slice feeds the A operand
  directly; the boxed pad rows only ever landed in output pad rows nobody read.
- `hc_head`: five scopes down to three (#822).

### 1.5 Mix kernels: what to fuse, and what to split

Fusing a matmul with its vector epilogue keeps the accumulator scope-local and
removes a handshake. It does not always work, and the rule discovered here is
narrow:

> **Fuse an FP32-out matmul with its vector epilogue. Do not fuse INT8
> (INT32-accumulate) scopes.**

The INT8 scopes — `qr_proj+write`, `qr_hadamard+quant`, `score_accum+store` —
trip a ptoas `pto.subview valid_shape` failure even under `UP_DOWN`, because a
column slice plus `row_sum` conflicts with a row split. Leave them separate.
When a fused scope overflows its buffer, add an inner `pl.range(ROW_CHUNK)`;
do **not** halve the group size (#371).

The opposite move matters just as much. Routed-expert cube and vector stages
were **decoupled** — a pure-cube scope writes INT32 to GM, a separate vector
scope does dequant, SwiGLU and the routing-weight multiply (#594). The reason is
precise: decoupled, cube and vector size their N fragments **against their own
bottleneck** (L1/Mat versus UB) instead of sharing one compromise. Splitting the
gate and up matmuls into separate cube tasks then left each holding a single
weight L1 tile, freeing Mat for an N fragment of 256.

> Fuse when two stages share a bottleneck. Split when they do not.

### 1.6 Pipelining and balance

- `pl.pipeline(stage=2→4)` needs iteration depth: 32 K blocks support a 4-deep
  ping-pong, while a loop left with 2 blocks after retiling does not (#339).
- **Check PMU before pipelining.** The gate/up K loop is MTE2-bound (~80 %) and
  gains from a pipeline; the w2 K loop is scalar-bound (~97 %) and was
  deliberately left serial (#473). See
  [Performance Tuning](performance-tuning.md#4-read-pmu-utilization).
- `pl.split(UP_DOWN)` fixes vector stragglers in mixed regions. In `proj_b` the
  INT8 GEMM finished early and the whole region's wall was set by whichever of
  the two vector lanes got the larger share of the dequant epilogue; splitting
  each task halved the imbalance for −16 % (#547).

---

## 2. Operator-specific rewrites

These do not transfer directly. The reasoning does.

### 2.1 Attention: make the gather disappear

The largest concentration of decode gains, reached in four steps. Each step is
a different answer to "what is the gather actually costing?"

**Step 1 — recognize it as an access-shape problem, not a compute problem.**
`gather_kv` is a pure `GM→UB→GM` copy: MTE-bound, with vector and scalar idle.
A per-row `GM→GM` copy **fully serializes MTE2 and MTE3** — codegen caps the
`GM→GM` ping-pong at 2-deep regardless of the `pl.pipeline` stage, and the
measured `MTE2_busy + MTE3_busy ≈ wall` confirms it (#539).

**Step 2 — stage in UB to coalesce.** Fill a UB tile one row per scattered load
(so loads stream on MTE2 with no buffer-reuse WAR), then flush the whole block
with one wide store. Zero the tile with the otherwise-idle vector unit so `-1`
and padding slots keep that zero (#539, #571).

**Step 3 — fuse the gather into its consumer.** Bulk-zero plus batched
`rope_pack` first (−14 %, #509), then the KV gather folded directly into
`qk_pv` via `gather_row` (#615). In the same period `qk_pv` was batched to
**M=32**: the shared sparse-KV tile is extracted L1→L0 once per two head tiles
instead of once per head tile, giving 2× reuse — **cube core-time −52.9 %,
vector −53.5 %** (#535). M=64 was rejected: its softmax tile and the co-resident
QK+PV L0C accumulators exceed budget.

**Step 4 — do not gather at all.** The sliding-window path attends the ring page
and the overlay tensor as **direct GM slices**, masked by a precomputed
physical-order bias — no gather, no scatter, no `gather_row`. The hybrid path
zero-gathers its window block and keeps `gather_row` only for the block that is
genuinely scattered (indexer-selected compressed rows). **SWA −14.2 %, HCA
−18.6 %** (#629).

Pruning rides alongside the same insight — work on invalid slots is still work:

- Sparse-K blocks cut from 5 to 2 in SWA/HCA by dropping all-`-1` padding
  blocks (#516).
- SWA specialized further to a **single** block: `PADDED_TOPK` halves and the
  cross-block online-softmax merge loop is deleted outright (#630).
- CSA records block-level validity while composing its sparse indices and lets
  `qk_pv` skip fully invalid blocks — but **`merge_norm` keeps the original
  merge order**; skipping merge blocks was tested and caused numerical drift
  (#641).

The cost to know about: these paths are hard to model on a simulator. Validate
on a real device.

### 2.2 Hyper-connections: two small operators, two failure modes

The hyper-connection stack is 4 streams wide, and its two operators fail
differently. `hc_pre` mixes once per sublayer — twice a layer, 86 times a
forward — so every fixed overhead it carries is paid over and over. `hc_head`
runs once at the tail, but at 8 token rows it was badly under-parallelized.

- **`hc_pre` — fuse.** Five scopes to one (#533), later folded into a single
  syncall task with the gate phase absorbed (#684).
- **`hc_head` — make the matmul pure-cube.** A dedicated cast scope streams x
  to FP32 once so the head projection is a clean cube kernel rather than a mixed
  cube+cast one; split-K with zero-seed and atomic-add FP32 partials fills idle
  cubes. Standalone ~199 µs → ~66 µs, about 3× (#606).
- **`hc_head` — then fan it again** (#822): the reduce spread over
  (token-tile × D-slice), `LINEAR_OK` 8 → 16, and — a detail worth stealing —
  zero the accumulator with `pl.create_tensor(init_value=0)` on the AICPU
  instead of a dedicated seed kernel, which had been spending ~5 µs of cube
  critical path to zero 1 KB. 58.8 → 42.6 µs, and run-to-run spread narrowed
  from 45–65 µs to 39–43 µs.
- **Both — settle the dtype of the whole stream.** The residual stream was BF16
  in GM, so every kernel boundary paid a cast: FP32→BF16 on write, BF16→FP32
  on the residual read (once per stream), plus a dedicated cast scope in
  `hc_pre`.
  Making the stream **FP32 end to end** deleted ~78 cast and dispatch tasks
  (1621 → 1543) and ~1 MB of staging, for **decode_layer −7.4 %** (#732).

That last one generalizes: **a dtype chosen per kernel becomes a tax at every
boundary.** Decide it for the stream, not the operator.

### 2.3 MoE: two independent problems

**The expert GEMM.** Profiling said vector-bound (AIV ~89 % busy) with 1184
tasks pressuring the dispatcher — the swimlane showed the ready queue frequently
empty and ~21 % idle spin. The fix was to halve the task count via a larger
receive tile (with the inner and quant tiles reduced so the vector working set
still fits UB at the larger M), and to fuse the two per-row w2 scales — the
per-row dequant scale and the routing weight — into a single row scale computed
once per block instead of an extra broadcast-multiply per output tile
(`exp_w2_aiv` −35 %). About 13 % overall (#445).

**The EP collectives.** This is where the traps are.

- Push the dispatch payload as INT8 directly rather than widening to FP16 and
  narrowing again — the receive window and buffer halve (#499).
- **Every publish and barrier notify must use AtomicAdd.** `Set` races with
  reordered notifies and deadlocks waits at world sizes above 2. Give the data
  phase its own completion window rather than reusing the count phase's (#499).
- **Parallelize one task at a time.** A single change that SPMD-parallelized
  dispatch and combine together hung 8-rank prefill with an AICPU stream-sync
  timeout and had to be withdrawn wholesale. It was re-landed in five steps, each
  validated against both a 2-rank case with a golden (catches numerical
  breakage) and an 8-card real-weight prefill (catches the hang), with the
  per-step results tabulated in the change itself (#743). **That table is the
  most reusable artifact in this history.**
- Split the handshake out: a wait task separate from the push grid, fenced with
  explicit `deps`, keeps the push a pure one-sided scatter.

### 2.4 Indexer and compressor: quantize on write, widen the pool

- **Quantize the indexer cache on write.** Each new compressed KV row is stored
  INT8 with a per-position FP32 dequant scale, so the score path reads the paged
  INT8 cache directly instead of re-quantizing the whole KV history every step —
  deleting an O(seq) per-step pass. Per-row quantization is position-independent,
  so this is **numerically identical** to the old score-time quantization (#725).
- **Widen the pooling tile.** The compressor's `softmax_pool` head-chunk loop
  collapsed from 8 tiles of 64 columns to a single 512-column tile, cutting
  vector op count and GM load transactions 8×. This is **bit-identical**, and the
  proof is worth stating in the form it was: the online softmax is *per-column
  elementwise* — every column carries its own running max, sum and output with
  no cross-column interaction — and all four state regions are contiguous
  full-width slabs, so one wide slice reads exactly what eight narrow slices
  concatenated read (#624).
- **Store weights transposed** where the consumer wants the transposed load
  path, so they arrive via `b_trans` (−7 % wall) (#628, #653).

The pattern in the last two: **a pure tiling or layout change that you can prove
bit-identical is the cheapest optimization there is** — it needs no new golden
and carries no numerical risk.

### 2.5 The router: squeezing a 40 µs operator

The gate is small, so it only yields to several changes at once. Four, for
~40 µs → ~33 µs across the in-pipeline window (#784):

1. **Defer the per-token `inv_rms`.** Store `x*gamma` and apply the reciprocal
   norm late — as a row scale on the logits, and (because symmetric per-token
   INT8 quantization is invariant to a positive scalar) only once on the
   quantization scale. The sum-of-squares and the scaled activation share **one**
   pass over the input.
2. **Stay FP32-native.** No BF16 round trip before the matmul.
3. **One token per core** for the norm: 12.6 → 6.2 µs, using the physical-tile
   trick from 1.2 to clear the 32 B floor.
4. **Vectorize the score gather**: 48 per-expert scalar reads become one batched
   `pl.gather` with `set_validshape` / `fillpad`, 9.9 → 5.2 µs.

The matmul deliberately stays FP32×FP32 to match the A2/A3 reference; only a
different platform target uses BF16 there.

A second example of the same "replace the algorithm, not the tiling" move:
greedy sampling used to sort all 505 vocab chunks purely to take each chunk's
maximum, then locate the winner with two serial scalar scans. It now folds each
row over an `[8, 808]` grid view and streams it **once**, carrying a running
maximum alongside the block that set it; a strict greater-than on the block
update preserves `torch.argmax`'s first-occurrence tie order. Both serial scans
are gone (#985).

---

## 3. Scheduling

Once the arithmetic is mined out, dispatch count and graph shape become
first-order. The mechanisms are documented in
[Dependencies and Scheduling](dependency-and-scheduling.md); what follows is
what they bought here.

### 3.1 Remove tasks and barriers

The output projection's amax was narrowed from **per row** to **per row and
`O_LORA` group**, with each group's INT32 partial dequantized by its own group
scale. That removes the full-row reduction **barrier** between the two
projection stages: each group's second matmul fires as soon as its own
quantization lands. Vector-side work dropped ~86 % and the call shed **160
AICPU dispatches (1002 → 842 tasks)** (#620).

It is the one change in this section that is **not** numerically neutral. A
narrower amax is a finer quantization grid, so the INT8 scales and the
projection output both differ from the per-row form — a scheduling win bought
with a numerics change, and accepted on an `attn_out` golden pass across all
three attention variants rather than on the task count alone. Section 0's rule
applies here too: the golden moved with the kernel.

Splitting a projection by K rather than by N took it from 26 µs to 10.7 µs
(#749) — the axis you split on is a scheduling decision, not only a tiling one.

### 3.2 Early dispatch

Applied along whole chains, since a consumer needs *every* direct producer
flagged. One caution learned explicitly: a **shared** kernel is deliberately
left unflagged when its other callers have not been benchmarked — changing its
early-resolve state silently alters paths nobody measured (#915).

### 3.3 Order sibling consumers with a dummy edge

The sharpest trick in the tree. When a normalization retires, five consumers
become ready **simultaneously** and race for cores — but only one is on the
critical path. Hanging a `pl.system.task_dummy` off the producer's TaskId and
routing the other four through it leaves the auto-tracked edges untouched while
making those four strictly later, so the critical consumer is dispatched first
(#749).

It costs a dispatch hop, and it costs a source change: the producer must switch
to the `with pl.spmd(...) as tid` capture form, because the `for ... in`
form yields no TaskId. Standalone entries with no producer pass an empty
`task_dummy(deps=[])`. Both idioms are catalogued in
[Deliberately delaying a task](dependency-and-scheduling.md#deliberately-delaying-a-task).

### 3.4 Delete redundant edges, and anchor waits correctly

83 redundant edges were removed in one pass — 8 cross-stage, 50 dispatch-to-
expert, 25 expert-local — after excluding every edge touching an allocation
task, and while preserving required transitive ordering (#803).

**Where a wait scope is anchored decides whether it helps or hurts.** A wait
with no dependency at all is dispatched the moment the scheduler reaches it,
then **spins holding a core group** — starving the very scatters it is waiting
on. Anchoring it on an upstream read makes it spin *alongside* the local push
instead of trailing it (#820, #840). The anchor has to be chosen with care: one
candidate view failed to compile because it is a cube output whose inferred
layout differs from what the outlined scalar-read kernel declares.

Folding the notify into the pushing blocks removes another launch from the
cross-rank critical path — valid only because those puts are single-shot and
drain themselves before the notify issues (#820).

### 3.5 A scheduling change that did not hold up

A fusion that pushed each activated row across the fabric from inside the expert
GEMM measured real core-time and drain gains — and **no end-to-end wall-time
difference**. Worse, its row-count handshake carried no epoch dimension: both
counters accumulated until a clear that runs once per program rather than once
per wave, so in a multi-layer forward a faster rank's next-epoch rows could
satisfy a slower rank's current expectation. Withdrawn in full (#975, #978).

Two lessons, both general: **core-time gains that do not become wall-time gains
are not gains** (see [Performance Tuning](performance-tuning.md)), and **any
counter-based handshake reused across waves needs an epoch dimension.**

---

## 4. Serving integration and lowering

The final group of changes stops looking at kernels and looks at a serving
step.

### 4.1 Make weights and state resident

Static weight shards upload once to their card and skip the per-dispatch
host↔device transfer. Build the resident set as *stacked layer weights minus
cache pools, plus the RoPE tables and head norms*, and **exclude** everything
per-step: KV and state caches, slot mappings, block tables, position ids,
sparse indices, activations, outputs (#687).

On the speculative path the same treatment covers outputs and recurrent state:
outputs kept resident skip their per-round copy-back, an initialized pool
uploads once and keeps its device handle across rounds, and recurrent
tail/draft/position/length fields live in stable per-request device slots that
accepted tokens are committed back into under a generation guard (#894, #895,
#917).

A side effect worth knowing: **residency also removes rank launch skew.**
Re-staging weights every round makes ranks start at different times.

### 4.2 Lower host work onto the device

Building position-dependent metadata on the host cost **12.3 ms** on the main
decode path and **4.9 ms** on the speculative path per serving profile. A shared
device-side metadata builder derives the same slot and index metadata inside the
*existing* rank-local decode graph — adding no separate dispatch — and argmax
sampling was folded into the grouped LM-head graph (#862).

### 4.3 Collapse L2 submissions

A speculative step originally submitted three L2 programs: main decode,
verification, draft decode. Fusing verification into main decode brought it to
two, with critical-rank host decode latency **45.159 → 42.980 ms** (#884).
Inlining the draft layer into the same per-rank callable brought it to one
(#901) — and that change honestly claimed *only* a dispatch-count reduction,
because the 8-card profile showed lower bind overhead but no material
steady-state gain. **Report what you measured, not what you expected.**

### 4.4 Rewrite the collectives

Both LM-head collectives went from one `CORE_GROUP` task each to an SPMD push
with the notify folded in, a wait-only scope, and a parallel gather:

- **Combine** fans the vocab all-to-all over blocks. Every put — including the
  self-targeted one — uses the draining put primitive; the non-draining
  remote-store does not drain before the folded notify issues, so a peer could
  gather tiles still in flight.
- **Dispatch** flips from pull to push. The window widens to one row slot per
  logit row so each card publishes its own slot, instead of blocking on
  `TP_SIZE - 1` remote loads per K tile.

**Fast card 528 → 301 µs**: combine 234 → 62 µs, dispatch 70 → 30 µs (#840).

### 4.5 Use L2 against inter-layer eviction

The most system-level change in the tree, and the one with the most instructive
constraints.

Every decode attention layer streams its whole weight set from HBM once per
forward, and in a full forward that traffic is always **cold**: the MoE between
two layers pushes 427.8 MB through L2, so nothing an attention layer read
survives to the next one. One SDMA CMO warm per layer now covers every weight
that layer reads, in consumer-deadline order, **anchored at layer entry** where
the cores are still busy with the previous stage so the warm overlaps them —
anchoring it later measured worse.

**Fast rank p50 40132.0 → 39287.9 µs (−2.10 %)**, with each attention block
returning to its standalone speed and MoE unaffected, as expected (#963).

Three hard constraints, each established by its own negative result:

| Constraint | Evidence |
|---|---|
| **One scope, one context.** | Splitting the warm across two `pl.at` scopes puts two SDMA streams in flight, halving aggregate throughput (285 → 153 GB/s) and turning a −1.1 % gain into a +0.7 % loss. |
| **All weights or none.** | Warming one projection alone is *worse than not warming*: the warm costs a near-fixed ~20 µs in its segment while the saving scales with coverage. |
| **The warm set must fit L2.** | 157.9 MB and 146.9 MB sets fit inside 192 MiB and win; a 268.4 MB set (1.33× L2) evicts itself and costs 3 %. |

The warm is a cache hint with no destination — deleting the scope changes no
value. That is what makes it safe to tune aggressively.

---

## See also

- [Performance Tuning](performance-tuning.md) — measurement, capture, and the
  L2 / L1 / L0 tuning rules
- [Cube Tile Tuning](cube-tile-tuning.md) — choosing row, N and K tiles against
  the compiler's memory report
- [Dependencies and Scheduling](dependency-and-scheduling.md) — how edges form,
  when the scheduler issues, early dispatch, and dummy-task idioms
- [Precision Tuning](precision-tuning.md) — rounding modes, dtype alignment, and
  threshold selection
- [DeepSeek V4-Flash (MTP)](../models/deepseek_v4_flash_mtp.md) — the model this
  page follows, top down
