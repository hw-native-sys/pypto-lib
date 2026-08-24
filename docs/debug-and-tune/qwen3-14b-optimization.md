# Qwen3-14B Optimization

This page is a case study rather than a reference. It follows
[`models/qwen3_14b/`](../../models/qwen3_14b/) — a 40-layer dense GQA model that
runs prefill, decode, and sampling on a **single card** — from its first kernels
to their current state, and records which levers moved the number, which did
not, and what each one cost.

There is no TP, no EP and no fabric here, so nothing in this history is a
collective or a placement decision. Every gain has to come out of one chip's
tiles, its task graph, and the boundary between it and the host — which is what
makes the order the work happened in worth recording.

The mechanisms live elsewhere:
[Performance Tuning](performance-tuning.md) for how to measure and capture,
[Cube Tile Tuning](cube-tile-tuning.md) for choosing tiles,
[Dependencies and Scheduling](dependency-and-scheduling.md) for the task graph
and the scheduler,
[Ring Heap and Scope Stats](ring-heap-and-scope-stats.md) for scope placement,
and [Precision Tuning](precision-tuning.md) for thresholds and rounding. Read
those for *how*; read this for *in what order, and what to expect*.

Numbers in parentheses are pypto-lib pull requests, kept so a claim can be
traced back to the change and its measurement. Every measurement quoted is the
one the change's author reported at the time.

## About the numbers

This tree predates the current benchmark convention, so its figures come in
three flavours that are **not** comparable with each other. Each quoted number
names which one it used:

| Convention | What it is | Where it appears |
|---|---|---|
| *total test time* / *task count* | the runtime-profiling summary over one dispatch | the earliest work, up to roughly #300 |
| *L2 device span* / *makespan* | the chip swimlane's device-domain window | most of the middle of the tree |
| *TTFT* / *TPOT* | end-to-end generation latency, host included | the device-sampling change (2.2) |

A TPOT number includes host dispatch and is the only one that answers "did the
product get faster"; a device span answers "did the kernel get faster". They
move independently, and this page says which one a change moved.

None of the three is what a new measurement should use. Today's number is the
`mean=` field of `PYPTO_BENCH=1`'s `[RUN] effective_us` line, which is also what
daily CI reports — see [Performance Tuning](performance-tuning.md). Do not
compare a fresh `effective_us` against a figure on this page.

## The shape of the work

| # | Question it answers | Typical per-change gain |
|---|---|---|
| 0. Shape constraints, measurement surface, golden | Can any later claim be believed, and which shape constant is load-bearing? | none — this is the admission ticket |
| 1. General levers: tiling, task count, parallelism, fusion | Is each kernel doing the minimum work on the maximum cores? | 20–90 % |
| 2. Operator-specific rewrites | Is the algorithm itself the wrong shape for this hardware? | 10–40 % on that operator |
| 3. Scheduling | Are the right tasks issued in the right order? | 2–20 % |

One number dominates this tree's history and it belongs to group 1: prefill went
from 572.5 ms to 77.4 ms of device span while its task count fell from 351,050 to
31,706 (#662). Nothing in groups 2 or 3 is worth attempting against a forward shaped
like that — an 11× task-count reduction is not a tuning result, it is a signal
that the task layout itself was wrong.

---

## 0. Shape constraints, measurement, and the golden

### 0.1 Find the shape constant that forbids the obvious fusion

Qwen3-14B has 40 attention heads over 8 KV heads, so each KV head carries
**5** Q heads. Five is odd and prime, and that one number shaped the whole
attention thread.

Every `SplitMode.UP_DOWN` row-split fusion — the standard way to fuse a cube
matmul with its vector epilogue on this hardware — **rejects it**:
`SplitVectorKernel requires an even split dimension, got 5`. Drop `UP_DOWN` and
the fallback fails differently: the `row_expand_div` output's `valid_shape`
stays uninitialised inside a mixed root, so a `[5, HEAD_DIM]` subview fails IR
validation (#349).

The tree spent three changes finding an answer:

1. **#349 — give up the fusion.** Keep the `Q_HEAD_PAD → 5` trim outside every
   mixed root, in a vector-only `fa_attn_row` scope. Three scopes where one
   would do.
2. **#360 — change the model.** `NUM_KV_HEADS` 8 → 10, making `q_per_kv = 4`.
   That makes the fusion legal and **is not Qwen3-14B**. It was flagged as a
   synthetic shape in the change itself and did not survive.
3. **The form that held — pad, and stop row-splitting.** Operate on
   `Q_HEAD_PAD = 16` physical rows, trim to the 5 real ones with valid-shape, and
   run the root under `SplitMode.NONE` dual-AIV no-op replay rather than
   `UP_DOWN`. That needed the toolchain to accept the `valid_row=0` subview the
   replay rewrites the trim into — ptoas >= 0.43 and the pto-isa
   `GetValidRow`/`GetValidCol` relaxation, both named as hard prerequisites in
   #420. [constants.py](../../models/qwen3_14b/constants.py) still asserts the
   geometry it requires: `Q_HEAD_PAD % 4 == 0` and
   `Q_HEAD_PAD // 2 >= Q_HEAD_BATCH`.

Two things generalize past this particular five:

> A model dimension that is odd, prime, or simply not a multiple of the split
> width silently removes a whole class of fusion from the menu. Find it before
> planning the kernel, not after the fusion fails to compile.

> When a shape blocks a fusion, **pad the physical tile and mask** — do not
> change the model, and do not split a scope you would rather keep whole.
> The padded form costs rows the kernel throws away; the split form costs a GM
> round trip and a dispatch at every layer.

### 0.2 Build the measurement surface before tuning anything

Several changes in this tree bought no speedup at all. They are the reason the
later ones can be believed.

| Change | What it bought |
|---|---|
| `name_hint=` on every `pl.at` scope (#276) | a swimlane whose rows carry stage names instead of generated symbols |
| PMU collection wired into the decode entry (#300) | per-pipe utilization, at levels 1 / 2 / 4 |
| Per-kernel Insight trace export (#303) | a cycle view of one generated kernel, from the real build directory |
| `--decode-steps N` (#795) | a **steady-state** decode sweep |
| `--enable-dep-gen` restored (#797) | the producer→consumer task graph, capturable again |

Two of those deserve expanding, because both correct a mistake that is easy to
repeat.

**`--decode-steps` (#795).** `--validate-fwd` dispatched `decode_fwd` exactly
once, so every timing run measured a *cold single invocation*, not a decode
step. The flag runs N autoregressive steps, feeding the sampled token back and
growing the context by one each time, starting at `MAX_SEQ - decode_steps` so
the KV pool sized for `MAX_SEQ` holds the whole sweep. Only then does a median
over 20 invocations mean anything (~37.7 ms/step at batch-16, 40 layers, context
3338→3358).

**`--enable-dep-gen` (#797).** The flag had been hardcoded off with a rationale
that did not hold: `dep_gen` was blamed for perturbing core occupancy and
tripping a full-occupancy `SyncAll`. `dep_gen` is an **AICPU-side** collector —
it occupies no AICore — so it cannot starve an AICore barrier. A two-layer
capture ran clean and still validated. The real constraint is unglamorous: the
per-run SHM record buffer overflows on a full 40-layer graph, so capture with a
small `--fwd-layers`. **A disabled diagnostic with a wrong reason attached is
worse than a disabled diagnostic**, because nobody re-tests it.

### 0.3 A 40-layer BF16 golden needs a comparator, not a looser tolerance

The fused multi-layer decode is numerically correct at single-layer scope and
still fails a strict element-wise check at 40 layers, because BF16 accumulates
1–2 ULP per layer:

| layers | 1 | 2 | 4 | 6 | 8 | 10 | 40 |
|---|---|---|---|---|---|---|---|
| mismatches @ `atol=3e-3` | 0 | 4 | 5 | 17 | 39 | 80 | 531 / 81920 |

Smooth geometric growth, no systematic bias, cosine > 0.999. The answer was a
**pass-rate comparator** rather than a wider envelope: `make_pass_rate_compare`
passes when at least a threshold fraction of elements satisfy the run-level
`atol`/`rtol` (#241). A real systematic bug still fails fast — most elements
skew the same direction and the rate collapses — while the ULP long tail is
absorbed. CI then runs it as a three-tier sweep so sensitivity stays uniform
with depth (#247):

| layers | threshold | measured | role |
|---|---|---|---|
| 1 | 1.000 | 1.000000 | single-layer kernel correctness |
| 10 | 0.999 | 0.999939 | medium-depth accumulation |
| 40 | 0.98 | 0.989795 | full-depth regression |

**A loosened tolerance is a debt with a due date.** #349 raised `atol`/`rtol`
from `3e-3` to `1.5e-2` to get CI green around a pypto cross-lane GM race, with
the upstream issue named at the call site. #366 restored `3e-3` and deleted the
comment the same week the upstream fix landed. Every temporary tolerance in this
tree carries the issue number that will retire it.

---

## 1. General levers

Everything in this section transfers to any kernel. Ordered by what paid most.

### 1.1 Task count is a first-order cost

The largest win in the tree, by a wide margin, was not arithmetic. Prefill was
emitting a third of a million tasks per forward.

| Scenario | Metric | Before | After | Change |
|---|---|---:|---:|---:|
| 40 layers, batch 1 × 128, swimlane, 3 runs | task count | 351,050 | 31,706 | **11.1× fewer** |
| same | total test time | 572.5 ms | 77.4 ms | **7.40× faster** |
| real-weight serving replay, 5 runs | AICore tasks | 333,562 | 36,434 | 9.16× fewer |
| same | prefill device span | 579.4 ms | 77.0 ms | 7.53× faster |

The change (#662) restructured the layer's task layout around phases —
RoPE/KV-cache, attention, projection, RMSNorm, MLP, residual — so the
orchestration emits fewer, larger, better-balanced tasks. No runtime change was
needed. **Before you tune a tile, count the tasks.**

The same lever shows up smaller three more times, and each is a different way to
delete an edge rather than a task:

- **Funnel fan-out through a node that is already on the path.** Four zero-fill
  accumulator seeds each fanned out to their split-K atomic-add consumers:
  85 + 85 + 85 + 50 = **305 seed→atomic-add edges** for the AICPU to track.
  Routing them through the attention task instead — which gates on all four
  seeds, letting the atomic-adds drop their direct edge — leaves **4**.
  Ordering is preserved transitively, and the seeds finish long before the
  attention's own dependencies are ready, so the new edges are off the critical
  path (#682).
- **One dispatch has one TaskId; do not build an array of them.** Per-tile and
  per-slab TaskId arrays for `q_proj` / `k_proj` / `v_proj` / `dcr_xgamma` /
  `x_gamma0` were replaced by the single completion TaskId each SPMD dispatch
  emits (#762).
- **Capture the dispatch instead of fencing it.** A dummy `attn_fence` task
  existed only to bridge attention → `out_proj` across a manual-scope boundary.
  Taking the dispatch's TaskId with `with pl.spmd(...) as tid` and using it as
  the dep directly removed the task (#489).

### 1.2 Fill L0B, then fill the cache line

Two separate walls, hit in that order.

**L0B (#190).** `out_proj` shared the pipeline's global `K_CHUNK = 128`, which
fills the right-operand buffer to a quarter. Giving it its own
`OUT_PROJ_K_CHUNK = 512` (with `OUT_PROJ_N_CHUNK = 64`), and raising the
attention page/tile size to 256 so the QK/SV K and V tiles use L0B more fully,
were both part of #190, whose composite effect is the table in section 1.6.

The attention change is structural as well as arithmetic: a 4x wider sequence
tile shrinks `ctx_blocks` 4x, so the online-softmax accumulation chain gets four
times shorter. What it cannot move is L0A and L0C, which stay bounded by
`M = BATCH_PAD = 16` — the padded decode batch is the M of every matmul, and no
tiling choice changes that.

**The 512 B L2 line (#223, #489, #758).** A BF16 tile 128 columns wide reads
256 B of a 512 B line. Widening to 256 columns is the same instruction count and
half the fetches:

- decode projections moved to 512 B-aligned 256-wide tiles (#223); K=512,N=256
  and K=256,N=512 were both tried and **exceed the 512 KB mat-buffer verifier**.
- the LM head's cube tasks were widened to one full line (perf hint PH001) and
  the per-chunk `pl.parallel` fan-out collapsed into a single grid-stride
  `pl.spmd` over `LM_HEAD_CORES` persistent blocks, storing through
  `set_validshape` straight to `out` — bit-identical, because the vocab columns
  are disjoint and there is no split-K (#489).
- prefill's projection matmuls were MTE2-bound on exactly this sub-line
  over-fetch. Cube K/N tiles to 256 took a 4-layer batch-1 128-token prefill
  from **8600 µs to 6400 µs (−26 %)** (#758).

### 1.3 Stream a weight once, not once per token block

Prefill's MLP weights are ~178 MB each. The layer-major loop re-streamed them
for every token block.

#758 restructured the layer into two phases: phase 1 runs the per-token
attention path through post-attention RMSNorm and stores its two hand-offs
(post-norm activations, first residual) for the whole packed token dimension;
phase 2 runs gate/up/down as flat band-grouped token-tile sweeps. Tiling the MLP
matmuls at **M=128** — two 64-token tiles at once — reuses each weight slab
across 128 rows in L1/L0 instead of 64.

Alongside it, gate → up → silu → down was fully fused per band, each band's down
partial atomic-added into a residual-seeded FP32 accumulator, and gate/up merged
into one `spmd(24)` with a core shift so the heavy-core sets are disjoint and no
core carries more than 3 N-tiles. Prefill went **6400 → 5800 µs (−9 %, −33 %
cumulative with the tiling above)**.

One detail generalizes: the vector epilogues (silu, down+residual) are UB-bound,
not MTE2-bound, so they were **decoupled to a finer 64/128 fragment** while the
cube tiles stayed at 256. The rule that decides it: *fuse when two stages share a
bottleneck, split when they do not.*

### 1.4 Raise parallelism: flatten to one SPMD grid over the product axis

The repeated move is to replace `for b in pl.parallel(batch)` wrapping a
`pl.at(CORE_GROUP)` with a single flat `pl.spmd` over the *product* of the
independent axes, and decode the indices inside the block.

| Operator | Original grid | New grid | Result |
|---|---|---|---|
| attention | `pl.parallel(user_batch)` × per-batch dispatch | flat `pl.spmd(BATCH × TOTAL_Q_GROUPS // 2)` = 64 | one pool to load-balance instead of 16 small ones; task count unchanged, per-batch launch + barrier overhead gone (#387) |
| online softmax | per-batch | flat `pl.spmd(BATCH × TOTAL_Q_GROUPS)` = 128 | writes straight into `attn_out`, deleting the per-batch `attn_row` intermediate (#387) |
| RMSNorm, Q/K/V proj, QK-norm | `pl.parallel` + `pl.at` wrappers | top-level flat `pl.spmd` | same shape as attention and the MLP (#420) |

What it costs: the flat form promotes per-batch scratch to global scratch — the
attention accumulators grew by ~32 MB when they stopped being per-`b` (#387).
That was well inside budget here; check it before assuming.

What it cannot absorb: `rope_kv_cache` stayed inside `pl.parallel(user_batch)`
because the KV-cache slot write and the attention's cache read cannot share an
InCore region — codegen fails with `Tensor view not found for parameter`. A
cross-region barrier is required, and that is a correctness constraint, not a
tuning choice (#387).

### 1.5 Merge operators, and know which merges this shape forbids

Each scope boundary is a GM write, a GM read, and a dispatch.

- **Matmul + vector epilogue → one mixed root.** `out_proj + residual`,
  `down_proj + residual`, and `gate + up + silu` each collapsed into one mixed
  `pl.spmd` with an `UP_DOWN` row split (#310, #387). The accumulator stays on
  L0C across the K-loop and reaches the vector side through the C2V boundary
  move — no GM round trip. `gate_up_silu` additionally lets the two cube matmuls
  share a **single K-loop**, so each activation chunk is loaded from L1 once and
  feeds both weights.
- **What it deleted:** the `fp32_chunk_gm` scratch (~160 KiB per call) and the
  `gate_group` / `up_group` FP32 GM bridges.
- **Why `UP_DOWN` is mandatory here, not optional:** without it the per-core UB
  budget is exceeded under `--max-seq` (#387).
- **And why it does not reach the attention trim:** `Q_HEAD_BATCH = 5`. See 0.1.

Four hard walls found while merging, all worth remembering as shapes rather than
as bugs:

| Wall | Value | What it forced |
|---|---|---|
| AIC L0B, double-buffered | 64 KB | `BLOCK_SIZE` 256 → 128, because a 256-token K/V tile is 32 KB and any cube+vec fuse double-buffers it: `Right buffer usage 131072 bytes exceeds platform limit 65536` (#349) |
| mat-buffer verifier | 512 KB | rejected both K=512,N=256 and K=256,N=512 for the decode projections (#223) |
| `pto.subview` valid-shape | — | an odd split dim has no legal form inside a mixed root (#349) |
| GM store paths | — | one InCore function may not mix `tile.store` / `tensor.assemble` with scalar `tensor.write` into the same GM tensor; ordering and cache-line coherence are unguaranteeable across the two (#971) |

### 1.6 Pipelining, and the composite result

`pl.pipeline` reached the decode kernels inside the same change as the retiling
of section 1.2, and #190's branch history separates the two contributions:

| Build | total test time | tasks | kernels |
|---|---:|---:|---:|
| parent | 3010.86 µs | 1128 | 19 |
| pre-pipeline baseline | 1497.40 µs | 625 | 17 |
| + pipeline | 1433.74 µs | 625 | 17 |
| + this change | **1348.78 µs** | — | 17 |

**2.23× against the parent, 9.9 % against the pre-pipeline baseline.** The
composition is the point: pipelining alone was 4.3 %, and it is the retiling
around it that made the rest.

---

## 2. Operator-specific rewrites

These do not transfer directly. The reasoning does.

### 2.1 Attention: four in-house shapes, then a vendor kernel

The longest thread in the tree, and the one with the most instructive ending.

**Shape 1 — fuse everything into one mixed root (#318).** QK matmul + softmax +
SV matmul + online softmax in a single `UP_DOWN` mixed root, one
`fa_fused_aic` / `fa_fused_aiv` pair. This is the shape the decode path wanted,
and section 0.1 is the story of why it could not have it.

**Shape 2 — three scopes, because 5 is odd (#349).** With `Q_HEAD_BATCH = 5` the
single root is unrepresentable, and the file that held it was deleted. Split into `fa_qks` (cube QK + tail-masked vec
softmax), `fa_svo` (cube SV + vec online-softmax recurrence, **one root per
`gi` stream** — sharing one root across both streams caused ~10 % numerical
drift), and a vector-only `fa_attn_row` trim. Scope-2 cube time fell 130.9 →
96.2 µs (−26 %) and **total wall time did not move**, because at this sequence
length the LM head and MLP dominate. That is reported here exactly as it was
reported then.

**Shape 3 — back to one root, via CV boundary moves (#360, #420).** Dispatch
`pl.spmd(TOTAL_Q_GROUPS // 2)` with each block owning a Q-group pair and
pipelining them at `stage=2` (which supplies the ping-pong buffering that
`chunk=2` used to). Both cross-lane handoffs became **boundary moves, not GM
round trips**: cube QK → vec softmax is C2V, vec exp → cube SV is V2C. #420 then
pulled the online-softmax recurrence inside, keeping `mi`/`li`/`oi` in UB across
the runtime block loop, seeded with `mi=-INF, li=0, oi=0` so the first iteration
reduces to the seed case without a peeled body — and deleting three GM scratch
tensors.

**Shape 4 — absorb RoPE and the normalizations (#656).** `rope_qkv`, attention
and online softmax collapsed into one mixed root, replacing two grid dispatches
and their cross-kernel edges with in-kernel `pl.system.syncall` barriers, and
folding QK-norm into the RoPE step in-register. Three phases split per region
with `pl.split_aiv`: rope (NONE, 32-lane), attention (`UP_DOWN` row-halving),
online softmax (NONE, 48-way). It was reverted for wrong serving output and
re-landed unchanged once three silent-data-corruption defects were fixed
upstream — the kernel had been correct throughout (#703, #756).

**Then: buy it (#765).** The whole in-house fused attention was replaced by an
external CCEC mixed kernel derived from CANN **FusedInferAttentionScore**,
bound through `pl.jit.extern`, consuming vLLM's active-TND query and BSND paged
KV layouts directly with its own runtime tiler, GM metadata and workspace.

The honest result: **~235 µs at 98 % core occupancy, against ~233 µs for the
in-house baseline and ~233 µs for vLLM's own FIA.** A dead heat. It was adopted
anyway, and the reasons are the interesting part — an ABI that matches vLLM
without a materialized reshape, a maintained kernel, and one thing the in-house
version could never do:

**#796 — fold RoPE *into the extern*.** QK-norm and RoPE moved inside the CANN
kernel, deleting the standalone RoPE dispatch and its producer→consumer edge
entirely. The generated RoPE body runs on its original 32 logical AIV workers;
the remaining workers skip it but still join the mixed-core barrier. Two
redundant `dsb(DSB_DDR)` calls around the `SyncAll<false>` boundary were removed
after establishing on C220 / CANN 9.0.0 that the cache-backed metadata barrier
suffices — validated over ten 40-layer runs: 9 clean, 1 reference near-tie, 0
failures, all 24,330,240 logits within `5e-2`. A contract regression test now
ties the C++ worker guard to the Python `ROPE_CORES` value.

> A vendor kernel at parity is still worth adopting if it lets you delete a
> boundary the in-house one could not.

The methodology that made #796 possible is written up separately in
[CCE In-Core Profiling](cce-incore-profiling.md).

### 2.2 Sampling: host → device, then approximate → exact

**Get it off the host (#639).** Greedy sampling and token embedding moved onto
the device, with `REAL_VOCAB` added so kernels can distinguish the real
vocabulary from the padded device one, and a reverse scan over equal best logits
so ties match `torch.argmax`'s smallest-id rule. Measured on a full 40-layer
128-token generation: **8.583 s → 6.450 s end to end (~25 %)**, with decode TPOT
48.9 → 47.5 ms.

Read that split carefully. The kernels got 1.4 ms/token faster; the *run* got
2.1 s faster. The gain is almost entirely the per-token host sampling, host
embedding lookup and host/device synchronization leaving the generation loop.
The change's own summary says so: *"the main reason is not a large kernel-level
speedup"*. Prefill's embedding followed later (#774).

**Then make it exact (#769 → #787).** The first device top-k took the best 4
candidates from each 64-token vocab chunk and merged them — cheap, and
**approximate**: nothing prevents five of the global top-32 from landing in one
chunk. The replacement splits the 151,936 real logits into 74 groups of 2048
plus a 384-token tail, computes an exact top-32 per group with `sort32` +
`mrgsort`, and merges the 75 × 32 = 2400 candidates in a 4096-entry padded
buffer.

The correctness argument is one sentence and is why the rewrite is provable:
*if an entry is not in its group's top-32, at least 32 entries in that group rank
ahead of it, so it cannot be in the global top-32* — therefore the union of the
group top-32 sets contains the global top-32. The golden was changed to compare
directly against `torch.topk(logits[:, :REAL_VOCAB], 32)`, and adversarial
fixtures were added with all 32 winners concentrated in one 512-token region and
spread across 32 regions. 100 rounds: min 427.1 µs, median 427.8 µs.

**An exact algorithm is easier to validate than an approximate one.** The
approximate version could only ever be checked against itself.

### 2.3 Compressing the KV cache, and what the toolchain forbids

TurboQuant stores the KV cache as 4-bit Lloyd-Max codebook indices (#503).
The bring-up detail worth copying is not the quantizer: it is that
`turboquant_kv_dequant_chunk` was factored out of the dequant logic duplicated
across prefill and decode, and accepted on the grounds that it is
**bit-identical** to the inline code it replaced — the gather → renormalize →
scale → unrotate sequence did not change, only its call site did. A refactor
that can claim bit-identity needs no new golden.

#646 then packed two 4-bit indices per `UINT8`, halving the cache from
`HEAD_DIM` to `HALF_DIM` bytes per row. The packing is more instructive than
the saving, because it had to route around two toolchain constraints and the
detours are exact rather than approximate:

| Constraint | Detour |
|---|---|
| bitwise `shls` / `ands` / `or_` are **tile-only** and reject GM-slice tensors | pack as `hi*16 + lo` with unified arithmetic; unpack with `idx % 16` / `idx / 16`, which is exact in floating point because 16 is a power of two |
| `concat` deadlocks at runtime and assemble-of-gather hits a codegen `tmov` shape mismatch | keep the two halves separate all the way through renorm, scale and unrotate |

That second detour is only legal because the algebra allows it:
`||dec||^2 = sum(lo^2) + sum(hi^2)` and `dec@R^T = lo@R_lo^T + hi@R_hi^T`. The
layout follows from it — byte `c` holds `idx[c + HALF_DIM]*16 + idx[c]` — and
both host goldens mirror the pack and unpack rather than reimplementing them.

> When a primitive is unavailable, look for the algebraic identity that lets you
> avoid needing it. A detour you can prove exact is not a workaround.

---

## 3. Scheduling

Once the arithmetic is mined out, dispatch shape becomes first-order. The
mechanisms are in [Dependencies and Scheduling](dependency-and-scheduling.md);
what follows is what they bought here.

### 3.1 A disjoint-slice write is not a parallel write

The sharpest single finding in the tree.

`dcr_xgamma` — the fused layer output plus the next layer's `x*gamma` (#546) —
was written as `DOWN_ON` separate `pl.parallel` + `pl.at` tasks, each writing a
**disjoint** slice of `out` and `normed_out`. The in-code comment asserted 5-way
parallelism. On device it ran on **one core**: the writers WAW-serialized on the
shared tensors, ~25 µs at the chunk tail and a ~45 µs span at every layer
boundary, visible in the swimlane.

The root cause is specific and worth knowing by name: `OutWindowExternalizer`
bails out (`HasUnwindowableSiblingOutputWriter` /
`HasDuplicateExternalizedOutputParent`) when a parallel task writes **multiple**
outputs. The dcr + x_gamma fusion writes two, so the externalization is skipped,
the writes stay full-tensor, and the runtime tensormap serializes them.

The fix was to convert it to a single `pl.spmd(DOWN_ON)` dispatch, whose grid
blocks are parallel **by construction** and need no static disjointness proof.
The fusion is kept — still one task emitting both outputs, no extra GM round
trip. The five writers went from serial-on-one-core to ~4 µs on five cores, and
a 2-layer makespan from 2114.6 to 2090.4 µs.

> The compiler must *prove* `pl.parallel` slices disjoint before it will let
> them run concurrently. `pl.spmd` asserts it. When a "parallel" region shows up
> on one core in the swimlane, this is the first thing to check.

### 3.2 Critical wave: dispatch the tiles the consumer needs first

When 50 output-projection tiles become ready at once and the consumer only needs
some of them to start, splitting the dispatch pays.

`out_proj` splits its 50 `(n, k)` tiles into two waves: the last 24 are one SPMD
dispatch gated **directly** on the attention TaskId, and the other 26 are
deferred behind an unflagged `pl.system.task_dummy(deps=[attn_done_tid])`. The
direct dispatch deliberately carries no `allow_early_resolve`. #794 applied the
same pattern to `gate_proj` / `up_proj`: the leading `GATE_UP_SPMD_N = 6` N-tiles
go out as `pl.spmd` per K-split gated on the corresponding cast, and tiles
`n ≥ 6` route through a per-K `task_dummy` funnel.

The two ranges write disjoint columns and atomic-add over K, so the split is
value-equivalent to the fused form — the only thing that changes is issue order.
Both idioms are catalogued in
[Deliberately delaying a task](dependency-and-scheduling.md#deliberately-delaying-a-task).

### 3.3 Early dispatch, and the one place it hangs

`allow_early_resolve=True` on the `rope_qkv` SPMD grid took a 4-layer L2 makespan
from 4266 to 4119 µs (**−3.4 %**). The same hint on the `qk_norm`
`pl.at(CORE_GROUP)` scope **hangs the device with 507018** (#602).

> Early dispatch is valid on `pl.spmd` grids. It is not a scope-level knob.

The flag is now carried on most of the decode path's producer dispatches — the
seeds, the projections, `residual_rms_cast`, `down_proj`, `dcr_xgamma`. The
`out_proj` critical-wave dispatch (section 3.2) deliberately does **not** carry
it, and says so at the call site.

The same change (#602) carries two arithmetic wins that belong here because they
are about transaction count, not FLOPs: batching each `(KV head, batch)`'s
`Q_PER_KV` heads into one `[Q_PER_KV, HEAD_DIM]` tile replaced five single-row
load/compute/store trips that serialized on a single-buffered MTE2→MTE3 chain
(per-core makespan 14.75 → 3.60 simulator units, ~4×), and `pl.concat`-ing the
lo/hi RoPE halves into one full-width store halved the MTE3 transactions on the
tail that bounds the kernel. Folding the QK-norm reciprocal into `qk_norm` is
bit-exact — RoPE is linear in that per-row scalar — and deletes a misaligned
`[Q_PER_KV, 1]` column load plus two now-dead state tensors.

### 3.4 Place scopes on rings deliberately

The runtime exposes four HeapRings and maps `ring_idx = min(scope_depth, 3)`.
With `auto_scope=True` the compiler inserts ~7 nested scopes in prefill, so
everything past depth 3 folds into ring 3.

| | ring0 | ring1 | ring2 | ring3 |
|---|---:|---:|---:|---:|
| auto — heap | 0 | 0 | 2 MB | **1388 MB** |
| auto — task window | 0 | 1 | 4 | **27,692** |
| manual 4-ring — heap | 10 MB | 2 MB | 67 MB | 1312 MB |
| manual 4-ring — task window | 7131 | 6 | 3412 | **17,152** |

Setting `auto_scope=False` and placing three explicit `pl.scope()` wrappers —
ring 1 the layer loop, ring 2 the token block, ring 3 the per-token attention —
dropped ring-3 task-window pressure ~38 % (#500).

This is not a micro-optimization. Under auto-scope the 40-layer prefill
**deadlocks** with a 507018 AICPU sync timeout, and the uniform
`PTO2_RING_TASK_WINDOW` knob has no working value: 131072 deadlocks, 524288
clears it but OOMs the static arena at 6.25 GB. With manual rings the same
kernel passes at 262144 / 4 GiB. See
[Ring Heap and Scope Stats](ring-heap-and-scope-stats.md).

## See also

- [Performance Tuning](performance-tuning.md) — measurement, capture, and the
  L2 / L1 / L0 tuning rules
- [Cube Tile Tuning](cube-tile-tuning.md) — choosing row, N and K tiles against
  the compiler's memory report
- [Dependencies and Scheduling](dependency-and-scheduling.md) — how edges form,
  when the scheduler issues, early dispatch, and dummy-task idioms
- [Ring Heap and Scope Stats](ring-heap-and-scope-stats.md) — manual scope
  placement and per-ring heap and task-window pressure
- [CCE Extern Kernel](../pypto-coding/cce-extern-kernel.md) — how a hand-written
  mixed CCE kernel is authored and bound behind `pl.jit.extern`
- [CCE In-Core Profiling](cce-incore-profiling.md) — phase partitioning inside
  the external attention kernel
- [Qwen3-14B](../models/qwen3_14b.md) — the model this page follows, top down
