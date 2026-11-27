# Qwen3.8-27B Decode Optimization

This page is a case study rather than a reference. It follows the decode path of
[`models/qwen3_8_27b/`](../../../models/qwen3_8_27b/) — a 64-layer hybrid model
that interleaves three Gated DeltaNet layers with one gated-attention layer, in
W8A8 on a **single card** — and records which levers moved the number, which did
not, and what each one cost.

Prefill shares most of these operators and almost none of their tiling, because
the two regimes differ in the one axis every tile is chosen against: at prefill
`T = 8192` fills the cores by itself, and at a decode batch there is no token axis
left to fill them with. That is why this page is decode-only.

The mechanisms themselves live elsewhere:
[Performance Tuning](../../debug-and-tune/performance-tuning.md) for how to measure and capture,
[Cube Tile Tuning](../../debug-and-tune/cube-tile-tuning.md) for choosing tiles,
[Dependencies and Scheduling](../../debug-and-tune/dependency-and-scheduling.md) for the task graph
and the scheduler, and [Precision Tuning](../../debug-and-tune/precision-tuning.md) for thresholds
and rounding. Read those for *how*; read this for *in what order, and what to
expect*.

Unlike the other optimization pages in this tree, no claim here carries a pull
request number: this history predates the model's first merge, so each change is
named by what it did. Every measurement quoted is the one its author reported at
the time.

## About the numbers

Every figure on this page is `a2a3`, single card, the median of a `PYPTO_BENCH=1`
run. The round count is stated per table and is 30 or 50; a figure without one is
a median of 30.

Three conventions appear, and they are **not** interchangeable:

| Convention | What it is | When it is used |
|---|---|---|
| per stage | one operator's own `effective_us` | tile sweeps |
| per layer | one `decode_layer` call, every stage inside it | attributing a stage change to the layer |
| per token | the per-layer saving times the layer count, divided by the batch | planning a decode step |

A stage figure is an **upper bound** on what the layer gets. The norm fan-out
below is the clearest case: a 2.73x stage win where one of the two layer types
ends up slower. Quote which convention a number used, and compare like for like.

**The card is a variable on this box.** `q_proj` read 75.0 us in one session
against 71.7 on the same machine at the same tiling, and the fused `qkv_proj` read
90.2 then 96.8. Neither was drift: an A/B of one script across cards 2 and 3
reproduced both gaps in the same directions and nearly the same sizes. Two
consequences for anything measured here — re-time the baseline in the same process
as the variant, and name the card when the two disagree. Not every stage is
card-sensitive: `qk_rmsnorm_rope` reads 16.6 us on card 2 against 16.9 on card 3.

The copy roof quoted for scale is this box's **vector-unit copy** roof, 1234 to
1236 GB/s. It is the wrong instrument for a GM → L1 → L0B weight stream and is
cited only as a denominator.

## The shape of the work

A decode token streams **24640 MiB** through the 64 layers before `lm_head`, and
the split is what sets every priority below:

| | per layer | layers | share of the stream |
|---|---:|---:|---:|
| MLP | 255.15 MiB | 64 | 63.6% |
| GDN mixer | 111.08 MiB | 48 | |
| attention mixer | | 16 | |

The MLP is two and a half times the mixer and runs in every layer. Within the GDN
mixer, the three projections are 140.00 MiB against the 6.29 MB of recurrent state
a token touches, and the short convolution moves 0.164 MB. So the projections and
the MLP are a bandwidth problem, and everything else on the decode path is a
**task-count** problem.

## 1. The regime: decode is a weight stream

### 1.1 Doubling the batch is free

The three GDN projections cost **162.7 us at batch 16 and 164.5 us at batch 32**.
Each one reads its whole weight to produce a handful of output rows, and the weight
does not grow with the batch. This is the cleanest statement of the regime
available, and it is why every tile below is chosen for the weight stream and not
for the cube rows.

The same fact sets the padding contract. M must be at least 16 — below it the build
fails inside `row_expand_mul`, a matmul result being a full 16-row fractal while the
dequant scale column is `m_tile` rows — and padding to it costs nothing, because a
batch-1 token pays the same 162.7 us as a batch-16 launch.

### 1.2 Tiles: fill the cores, then sweep

Best of 72 configurations per projection, median of 30 rounds:

| projection | m_tile | n_tile | k_tile | n_group | us at M=16 | GB/s | of the copy roof |
|---|---:|---:|---:|---:|---:|---:|---:|
| `in_proj_qkv` | M | 128 | 512 | 4 | 65.6 | 798.8 | 65% |
| `in_proj_z` | M | 64 | 1024 | 4 | 47.1 | 668.3 | 54% |
| `out_proj` | M | 64 | 512 | 4 | 50.0 | 629.7 | 51% |

The sweep over `n_tile` and `k_tile` bought 4.3% at the layer, so the easy headroom
is gone; what is left would need a different streaming structure, because the weight
has to be read whatever encloses it.

**The block count turns out to be a wave count.** `q_proj` has 96 n-tiles at
`n_tile = 64`, fine enough to hit the cube's 24 cores exactly:

| blocks | 3 | 4 | 6 | 8 | 12 | 16 | **24** | 32 | 48 | 96 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| waves of 24 | .12 | .17 | .25 | .33 | .50 | .67 | **1.00** | 1.33 | 2.00 | 4.00 |
| best us | 321.5 | 245.8 | 157.8 | 126.5 | 94.7 | 81.3 | **71.7** | 88.5 | 76.7 | 92.8 |

Monotonic below one wave, because cores sit idle. Above it the curve is
non-monotonic, and `q_proj`'s minimum is at exactly one wave.

**That is a fact about `q_proj`, not a rule.** The same kernel at `N = 14336`, 17%
wider, inverts the ordering: 56 blocks (2.33 waves) is best at 90.9 us, while 28
blocks (1.17 waves) is 109.0 and 16 blocks (0.67) is 93.2. One wave is not a target
and a whole wave does not beat a partial one in general.

What does hold across both shapes is the starvation side: below about half a wave
the cost climbs steeply and monotonically — `q_proj` 98.4 us at 12 blocks, the fused
shape 144.4 at 8 and 552.2 at 2. **Fill the cores, then sweep**, and measure the
block count per shape, because the same kernel at a nearby width answers
differently.

### 1.3 The two projections that cannot reach one wave

`k_proj` and `v_proj` are the worst-behaved matmuls in the model. `N = 1024` gives
16 n-tiles at `n_tile = 64`, and an even `n_group` caps the block count at 8, a
third of a wave. They cost 28.0 and 27.7 us for 5 MiB each — **15% of the copy
roof**, where `q_proj` moves twelve times the weight in two and a half times the
time. Their curve is monotonic in blocks (28.0 / 33.3 / 49.8 / 86.0 at 8 / 4 / 2 /
1), so the tile matters a great deal even though the weight does not: these are
parallelism-starved, not launch-bound.

## 2. Fusing the attention projections

### 2.1 One matmul over a concatenated weight

q, k and v read the same quantised activation, so one `a8w8_linear` over the three
weights concatenated by row is the same arithmetic: no row is requantised,
`a8w8_linear` accumulates INT32 over the whole K and dequantises once. The fused
output is therefore **bit-identical** to the three separate ones at every tiling,
measured exact over 229376 elements at M = 16 and 458752 at M = 32, on random and on
real layer-3 weights.

What the fusion buys is a uniform loop. Same script, 50 rounds at M = 16, the two
runs back to back:

| | card 2 | card 3 |
|---|---:|---:|
| q + k + v | 130.3 | 126.2 |
| `qkv_proj` | **90.5** | **95.8** |
| the fusion saves, per layer | **39.8 us** | **30.4 us** |

The two cards disagree in opposite directions — `qkv_proj` runs 5.9% faster on card
2 while `q_proj` runs 4.8% faster on card 3 — so the gap is not a property of either
alone. Both figures are real; **30.4 us/token is the one to plan with**, being the
worse of the two measured. At 16 attention layers and batch 16 the per-layer and
per-token numbers coincide, which they do not at any other batch.

This does not reverse the earlier finding that fusing projections which share an
activation is not worth it: activation traffic does **not** fall. Counted from the
generated loop bounds, the fused and unfused forms read the same 100 MiB. The decode
gain is block count and launch count, which that finding never tested.

### 2.2 Why the output stays one tensor

Keeping three output tensors does not work, and why it does not is the point. A
block's store target would depend on its n-tile index, which is a runtime scalar
inside `pl.spmd` and cannot be branched on at trace time. The only escape is three
loop nests striding the same block set, and that cannot balance: q has 96 n-tiles at
`n_tile = 128` against k's and v's 8, so 56 blocks give the first block six tiles and
the last three where the mean is two. Only a block count at or below 8 balances, and
8 blocks costs 144.4 us.

So the fusion's content is one uniform loop over 112 identical tiles, which needs one
weight and one wide output — the consumer change is what it costs, not an incidental
tidiness.

### 2.3 What the consumers pay

The consumers read q and k by channel range out of the fused tensor and cost the same
as the narrow form to within 1.4%, so none of the producer's gain is given back.
`attn_gate_quant` is the exception: its body views `qg` by a flat reshape that is only
valid at exactly `[t, 2 * HQ * D]`, so it reads a narrowed copy instead. That copy is
one `pl.spmd` group over 393 KB at batch 16, against the 30 to 40 us per layer the
fusion buys.

With the RoPE fan-out of section 3.6 on top, the whole front end of an attention layer
goes **198.1 → 121.8 us on card 2 and 192.7 → 127.3 on card 3**, which is 76.3 and
65.4 us/token for the two items together at batch 16.

## 3. The vector stages: fan out what the token axis no longer splits

### 3.1 One task is the symptom

Every elementwise and norm stage sets its task count to a quotient of the token
count. At prefill that quotient fills the cores. At a decode batch it is **one**, and
one task is all it is: the residual norm measured inside the composed layer costs 52.4
us at batch 16 and 50.4 at batch 32 — flat in the batch, which is the shape of a
single core doing the whole job. About 1.3 MB of traffic in 52 us is 25 GB/s, one
core's share of the copy roof. The stage is not slow; there is just one of it.

The fix has one shape across five stages: the token axis is gone, so fan out over the
columns (or over heads), with a scratch carrying the partial reduction between two
`pl.spmd` groups.

**Both ends of every curve cost.** Too few tasks leaves cores idle with real work to
do; too many pays a per-task cost over and over. In three of the five stages the
finest tiling measured is *slower than the single task* it replaced.

### 3.2 `act_quant`: the largest win, and the cache line that forces a third pass

| col_groups | 17 | 34 | 68 | 136 |
|---|---:|---:|---:|---:|
| column width | 1024 | 512 | 256 | 128 |
| us at M = 16 | **27.4** | 34.1 | 55.7 | 102.3 |
| us at M = 32 | **29.7** | 37.2 | 56.7 | 92.2 |

Against the single task's 78.3 and 78.6 that is 2.86x and 2.65x, so 50.9 us per layer
and **203.6 us/token at batch 16** over the 64 layers, 97.8 at batch 32.

The combine pass has to be a single `pl.spmd(1)`: 16 FP32 scales are exactly one
64-byte line, and a line written from two blocks loses one of them silently. That is a
correctness constraint, not a tiling choice, and it is why this stage has three passes
where the residual norm has two.

`col_groups` must divide `INTER // 128 = 136` for the same kind of reason — an INT8
tile store whose width is not a multiple of 128 elements writes wrong values with no
compiler diagnostic. Both constraints are documented at
`act_quant.build_fanned_kernel`.

### 3.3 `quant_x`: the same shape, a much smaller win

| col_groups | shipped | 4 | 5 | 8 | 10 | 20 |
|---|---:|---:|---:|---:|---:|---:|
| column width | 5120 | 1280 | 1024 | 640 | 512 | 256 |
| us at M = 16 | 23.9 | **18.2** | 18.6 | 20.8 | 21.8 | 28.0 |
| us at M = 32 | 24.7 | **22.5** | 22.5 | 23.0 | 23.8 | 29.0 |

1.31x at M = 16 and 1.10x at M = 32, so 5.7 and 2.3 us per invocation. `quant_x` runs
**128 times per decode step** — the GDN block 48 times, gated attention 16, the MLP 64
— which is 45.2 us/token at batch 16 and 9.0 at batch 32.

Against `act_quant`'s 203.6 us/token for the same structural change, this stage was
already near the launch floor: there is only about 10 us of serial work here to divide,
where `act_quant` had 64. **The win shrinks with the batch**, which is the honest
reason to read this entry before reusing the pattern.

Four groups is four tasks on a 48-core machine and still wins, which is the recurring
rule: a task that carries enough work to amortise its own cost should be big, not
numerous. The floor is a buffer bound — 2 groups needs 327808 B of the 188416 B vector
buffer.

### 3.4 The residual norm: a 2.73x stage, and a layer that gives it back

| col_groups | 4 | 5 | 8 | 10 | 16 | 20 | 40 | 80 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| column width | 1280 | 1024 | 640 | 512 | 320 | 256 | 128 | 64 |
| us at M = 16 | **19.0** | 19.1 | 20.3 | 21.4 | 26.3 | 27.6 | 37.8 | 65.4 |
| us at M = 32 | 25.2 | 24.1 | **23.1** | 23.8 | 27.8 | 30.7 | 41.6 | 60.9 |

Against the one task's 51.8 and 49.5 that is **2.73x and 2.15x**, 32.8 us a call at
batch 16. Eighty groups of 64 columns is slower than the single task.

**What a layer gets is less than that, and in one of the two layer types it is
nothing.** Measuring all three builds in one process: the linear-attention layer gives
back 50.0 us of the 65.6 its two calls save at batch 16, and the full-attention layer
is 8.4 us *slower* at batch 16 and 35.0 at batch 32. This is the sharpest example on
the page of why a stage figure is an upper bound, and the reason is not yet understood.

The default is pinned at `col_groups = 5`, which is neither batch's best — the optimum
moves with the batch (4 at M = 16, 8 at M = 32) because a task's work doubles when the
token loop does. Five costs 0.5% at M = 16 and 4.3% at M = 32, against 7% and 9% for
pinning either end.

Unlike the quantisers, this form is **not bit-identical** to the one it replaces:
splitting the reduction reorders 5120 additions. Every term is a square, so there is no
cancellation, but it is a difference, and the gate is a tolerance against the float64
oracle rather than an exactness claim.

### 3.5 `gated_rmsnorm` and `attn_gate_quant`: one task per token

Both stages were the largest vector stage left on their layer's critical path — 68.5 us
and 42.5 us as a single task. Both split by token rather than by column, each task
publishing its token's amax into its own row of a scratch so that no two tasks share the
`act_scale` cache line.

`attn_gate_quant` was measured at both levels: **50.0 → 18.0 us at batch 16** (2.79x) and
50.8 → 20.8 at batch 32 (2.44x), taking the composed full-attention layer 759.3 → 736.4 us
at batch 16 and 1019.1 → 993.5 at batch 32 — 22.9 us/token over the 16 layers.

`gated_rmsnorm` has no decode-shaped specs of its own, because its `o` comes from the
prefill chunked chain, so it was measured only inside the composed block: the
linear-attention layer goes **785.8 → 727.4 us at batch 16** and 1000.0 → 953.3 at batch
32, which is 175.1 us/token over the 48 layers.

Both are **bit-identical** to the forms they replace: the amax reassociates from 8 lanes
to 16, and a maximum is exact under any association. The composed output is exact over
all 81920 elements against the shipped build's own device output, on random and on real
weights.

### 3.6 `qk_rmsnorm_rope`: fan over heads

Here the token axis is gone and the column axis is not the one to split, so the
`(token tile, head)` axis is flattened and spread over two `pl.spmd` loops — two,
because telling a query head from a KV head would be a branch on the block index, which
is a runtime scalar.

| q_blocks | 2 | 3 | 4 | 6 | 8 | 12 | 24 | 48 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| us at M = 16, `pl.range` | 29.8 | 24.3 | 21.4 | 19.0 | **18.0** | 18.4 | 19.8 | 23.9 |
| us at M = 16, pipelined | | | | **17.1** | 17.8 | 18.8 | | |

Pipelining a task's run of heads is worth a further 5%, and it moves the optimum from 8
blocks to 6 because a longer run becomes cheaper to walk. **17.1 us at M = 16 and 20.8 at
M = 32**, 3.06x and 2.49x against the unfanned 52.2 and 51.8. `q_blocks = 6, k_blocks = 2`
is best at both batches measured, so one pinned pair costs nothing — unlike `HEAD_TILE`,
where pinning costs 2.4x.

Bit-identical: every structure writes each head's output tile from exactly one task.

## 4. The recurrence and the convolution

### 4.1 The delta rule: nothing crosses to the cube

The output `sum_k S[v, k] q[k]` is a matmul of one row against the state, and writing it
as one costs **2.7x the whole rest of the kernel**: the crossing pushes a 32 KiB operand
to carry a single row. Computed as a reduction on the vector unit instead it costs 13%.

Per-task cost dominates the rest. The decay alone, one task per `[V_TILE, D]` block, is
4.5x the state-traffic floor, and halving `V_TILE` doubles the time — so each task walks
`HEAD_TILE` head-states rather than one.

### 4.2 `HEAD_TILE` tracks the batch

| batch | HEAD_TILE 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| 1 | **27.9** | 30.7 | 42.9 | 67.8 | 121.9 |
| 8 | 97.7 | 93.5 | 86.2 | **83.4** | 131.3 |
| 16 | 190.9 | 157.4 | 149.0 | 142.9 | **140.6** |

**The best value is the batch size**, because 48 AIV cores want at least 48 tasks and
`batch * H / HEAD_TILE` is the task count. This is the one constant on the decode path
where pinning a single value costs real time — 2.4x at the ends — which is why serving
builds its one- and two-row variants with their own tile.

`V_TILE` is 64 because a whole head's `[128, 128]` FP32 state is 64 KiB and the live set
at `V_TILE = 128` needs 280064 bytes of a 188416-byte vector buffer.

### 4.3 The short convolution: fewer, bigger tasks

This kernel is per-task cost, not bandwidth: a token moves 0.164 MB against the recurrent
state's 6.29 MB, which is 0.05 us of traffic at the measured marginal rate. Both tiling
knobs therefore exist to control the task count.

| batch | TOK_TILE 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| 1 | **12.6** | 13.1 | 12.8 | 13.2 | 13.1 |
| 8 | 21.4 | **19.2** | 19.7 | 25.5 | 25.7 |
| 32 | 42.9 | 30.5 | **26.7** | 29.5 | 40.9 |

The best task count is 5 at batch 1, 20 at batch 8 and 40 at batch 32 — never 48 for its
own sake, and never the minimum either. At batch 32, ten tasks (40.9 us) is as bad as a
hundred and sixty (42.9): too few leaves cores idle, too many pays about 0.2 us per task
over and over.

Against the recurrence, where every optimum landed on exactly 48 tasks, the rule that
covers both is: **when a task carries enough work to amortise its own cost, fill the
cores; when it does not, use fewer and bigger tasks.** A task here carries four multiplies
over `[1, C_TILE]`, so it does not.

Pipelining was measured rather than assumed. The prefill convolution found its largest win
exactly here — grouping row-tiles behind `pl.pipeline(stage=2)` took it from 2122 to 730.7
us, 2.9x — but that kernel is load-bound and had loads to hide. This one is bound by
per-task cost, so at the best tiling per batch pipelining gives 19.7 → 18.9 us at batch 8,
27.2 → 27.5 at batch 32, and nothing at batch 1. `stage=3` is worse than `stage=2`
everywhere, which reproduces the prefill's finding.

`TOK_TILE = 4` ships because it is within 2.6% of the per-batch best everywhere measured,
where 8 costs 33% at batch 8. `C_TILE` is 2048, the widest tile that does not straddle the
q/k/v boundaries in the channel axis.

## 5. Attention: one split, not four

`SPLITS` partitions each sequence's pages across several tasks. It buys the length of the
serial chain, not occupancy: the online-softmax carry is sequential in pages, so one task
walking 64 pages exposes 64 dependent rescales. Composed attention layer, median of 50
rounds:

| batch, context (pages) | splits **1** | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| 16, 320 (3) | **524.4** | 617.6 | 716.3 | 907.2 |
| 16, 1024 (8) | **572.6** | 625.6 | 747.8 | 972.5 |
| 16, 4096 (32) | **793.7** | 830.6 | 842.1 | 985.7 |
| 16, 8192 (64) | **1103.3** | 1136.2 | 1166.0 | 1223.1 |
| 48, 320 (3) | **724.0** | 1013.9 | 1301.7 | 1877.0 |
| 48, 4096 (32) | **1413.2** | 1501.6 | 1551.2 | 1961.7 |

One split wins in all six, monotonically, and the margin **shrinks with context and grows
with batch** — 1.366x at (16, 320) to 1.057x at (16, 8192), and 1.798x at (48, 320). The
trend across context is the one that would produce a crossover and it flattens instead, so
there is none inside the model's context.

**The batch axis dominates the context axis**, which is why an earlier measurement of this
constant said four. That one timed the kernel alone at batch 1 and 8, where four splits beat
one by 2.56x at 8192 context — correctly, because at batch 1 the unsplit kernel runs
`batch * HKV` = 4 tasks on 48 cores and splitting is the only parallelism available. At
batch 16 there are already 64 tasks, so the cores are full before any splitting and the
merge pass and partial traffic are pure cost. The padding contract requires a multiple of
16, so the region that chose four is a region the decode path cannot present.

Splitting slightly **improves** accuracy, because each partial chain accumulates fewer BF16
rescales: 2.404e-03 at one split against 2.296e-03 at eight. One split is therefore the
worse end of a 4.7% spread in that statistic, which the composed gate measures against the
quantisation floor rather than in isolation.

## 6. The MLP: the largest single term

| stage | us at M = 16 | GB/s | of the copy roof |
|---|---:|---:|---:|
| `quant_x` | 23.8 | | |
| `gate_up_swiglu` | 164.0 | 1087 | **88%** |
| `act_quant` | 78.3 | | |
| `down_proj` | 93.3 | 955 | 77% |
| | **359.4** | | |

**The two matmuls are the efficient part.** 88% and 77% of the copy roof beat every GDN
projection in section 1.2 (65%, 54%, 51%), so the widest projection in the model is also the
best-behaved weight stream in it. The sweep's best total is 359.1 — `down_proj` at
`n_tile=64 k_tile=1024 n_group=4` for 93.0 against 93.3 — which is 0.3% and not worth a
second triple to carry.

What cost was the pair of per-token quantisers: **102.1 us of the 359.4, 28% of the time for
0.3% of the traffic**, because each ran a single task at a padded batch of 16 where prefill
gives them 512. Both fan-outs of section 3 land here:

| composed MLP, median of 30 rounds | batch 16 | batch 32 |
|---|---:|---:|
| before the quantiser fan-outs | 374.2 us | 379.9 us |
| after | **290.0 us** | **300.8 us** |

84.2 us a layer at batch 16, **336.7 us/token over 64 layers** — and more than the two
stages' own savings add to, because a shorter stage also lets the next one start earlier.

Two constraints bind the tiles. The 512 KiB Mat buffer caps `n_tile * k_tile`, and binds
sooner on `gate_up_swiglu` because that kernel holds two weight tiles rather than one:
`n_tile=128 k_tile=1024` needs 557056 bytes and is refused, as is `n_tile=256 k_tile=512`.
And `a8w8_linear` requires an even `n_group`, so `down_proj` cannot run one n-tile per
block — which matters, because `down_proj` wants **20 blocks against 24 cube cores** and
falls off a cliff below that: 93.3 us at 20 blocks, 136.4 at 10, 243.3 at 5.

### 6.1 Fusing gate and up

`gate_up_swiglu` replaces two `a8w8_linear` instances plus the projection half of
`swiglu_quant`. It does **not** reduce activation traffic, and an earlier version of its
docstring claimed it did: counted from the generated loop bounds at `m_group 8, n_group 8`,
the unfused pair is two kernels of `8 x 4 x 10 = 320` activation tile loads per block and
the fused one is `8 x 8 x 10 = 640`. Reading the activation once per pass rather than once
per projection is exactly cancelled by amortizing each load over one n-tile instead of two.
Weight traffic is unchanged, and the projection pair measures 6025.2 us against the unfused
6036.8 — a wash, as identical traffic and identical cube work predict.

What the fusion buys is the intermediate: one `act` tensor where two were written and read
back, 570 → 285 MB each way at T = 8192, and the quantizer that reads it drops from 1137.9
to 794.4 us.

It adds a rounding the unfused chain does not have. `swiglu_quant` goes from the two BF16
projections straight to int8 without materialising the activation; writing `act` rounds it.
Measured at 99.01% of `act_q` exact against the unfused chain, every element within one
step. `act_dtype=pl.FP32` builds the control that adds no rounding and saves no storage.

## 7. Precision decisions

### 7.1 `out_proj` in BF16 to match the checkpoint

The deployed W8A8 checkpoint leaves the GDN `out_proj` unquantised, and the model matches
it. That one projection therefore reads `6144 * 5120 * 2` bytes, and a linear-attention
layer streams **140.00 MiB** of projection weight rather than 110.00 — over 48 layers, 1.41
GiB more per decode token, 5.8% of the whole stream. The cost was 2.2% of the decode number,
spent on accuracy deliberately.

`o_proj` is the same `K=6144 N=5120` matmul and one measured tile serves both, but the
checkpoint quantises the attention one and not the GDN one, so they are no longer the same
bytes and each layer type names its own entry.

## See also

- [Model overview](index.md) — architecture, public interfaces, supported configurations
- [Performance Tuning](../../debug-and-tune/performance-tuning.md) — how to take a number
- [Compile and runtime workflow](../../run-and-validate/compile-runtime-workflow.md)
