# Performance Tuning

A practical guide for tuning pypto-lib kernels on Ascend NPU (A3 / 910C).
The flow is two-tiered: first balance the inter-kernel schedule on the
AICPU side (L2 swimlane), then optimize each kernel's internal pipeline
(L1/L0 swimlane + PMU).

For the underlying levels see simpler's
[hierarchical_level_runtime.md](https://github.com/hw-native-sys/simpler/blob/main/docs/hierarchical_level_runtime.md):
L2 = one chip (AICPU + AIC/AIV cores), L1 = die / L2 cache, L0 = single
compute core.

---

## Part 1 — L2 tuning (inter-kernel schedule)

### Capture

Run the case with `--enable-l2-swimlane`. The runtime writes per-task L2
records and a merged swimlane JSON under the build directory:

```bash
python models/qwen3/14b/qwen3_14b_decode.py -p a2a3 -d 0 --enable-l2-swimlane
```

```
build_output/<ProgramName>_<ts>/dfx_outputs/
├── l2_perf_records.json
└── merged_swimlane_<ts>.json   ← open this
```

Two viewers work:

- Open `merged_swimlane_<ts>.json` in <https://ui.perfetto.dev/>.
- Or open `l2_perf_records.json` directly with the
  [pypto-toolkit VSCode extension](https://marketplace.visualstudio.com/items?itemName=CANN-PUB.pypto-toolkit).

The trace shows one lane per AICPU / AIC / AIV with task name, duration
and dependency edges — gaps and stalls are visible directly.

### What to look for

Look for these shapes on the swimlane that indicate a problem:

| Symptom | Likely cause | Fix |
|---|---|---|
| Cores idle while AICPU lane is solid | Kernels too small; AICPU scheduling is the bottleneck | Make kernels larger (item 2) |
| Long tail on a single AIC/AIV | One kernel is too big and serializes | Split it (item 3) |
| Cube / vector unit utilization low even though kernel is busy | Tile size under-fills the core buffers | Re-tile to fill `mat_left` / `mat_right` / vector buffers (item 4) |
| Cube lane busy while vector lane idle (or vice versa) | Vec/cube epilogue is split into separate kernels | Merge into a mixed kernel (item 5) |
| Sequential AICPU dispatch trail per region | Region issues one kernel per iteration | Use `pl.spmd` to dispatch a block fan-out once (item 6) |

### Tuning rules

#### 1. Use `pl.range` vs. `pl.parallel` correctly

`pl.parallel` declares iterations are independent — the compiler may
distribute them across cores. `pl.range` is strict sequential and forces
a dependency chain. Use `pl.parallel` whenever there is no carried state,
and reserve `pl.range` for accumulators or stateful loops.

```python
# qwen3_14b_decode.py: batch tile is independent — pl.parallel
for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
    ...
```

A `pl.range` over an independent dimension forces the swimlane into a
single lane; switching to `pl.parallel` is usually the largest single
win at this stage.

#### 2. Kernels too small — fold loops into `pl.at`

When the swimlane shows cores idling while the AICPU lane is fully
saturated, the AICPU dispatcher is the bottleneck. Target ~50 µs per
kernel on A3 / 910C (smaller kernels add dispatch overhead that the AICPU
can't hide).

Move tight sequential loops **into** the `pl.at` region so one larger
InCore kernel replaces many tiny ones:

```python
# Multiple short loop iterations folded into one InCore kernel
with pl.at(level=pl.Level.CORE_GROUP, name_hint="kproj"):
    for kb in pl.range(0, K_BLOCKS):   # all K-iterations in one kernel
        ...
```

#### 3. Kernels too big — split and parallelize

When one kernel dominates the swimlane and the rest of the chip waits on
it, the kernel is too coarse. Pull a `pl.range` out of the `pl.at` and
convert it to a `pl.parallel` chunk loop so each chunk becomes its own
InCore kernel scheduled across cores:

```python
# Before: one giant InCore region over all q_out blocks
with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj"):
    for q0 in pl.range(0, hidden, Q_OUT_CHUNK):
        ...

# After: each q-chunk is its own kernel, parallel across cores
for q0 in pl.parallel(0, hidden, Q_OUT_CHUNK):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj"):
        ...
```

#### 4. Tiling — fill the core-internal buffers

Each AIC / AIV core has fixed on-chip buffers: cube `mat` (left/right
operand staging) on AIC, vector working buffers on AIV. The tile sizes
declared in your `pl.slice` / `pl.matmul` (typically `BATCH_TILE`,
`K_STEP`, `Q_OUT_CHUNK`, …) control how full those buffers run each
iteration.

- Too small → buffers are under-utilized, cube/vector throughput drops
  proportionally, MTE2 issues many small loads.
- Too large → tile spills, the compiler falls back to smaller transfer
  units, or compile-time shape checks fail.

Pick tile sizes so that one tile of the cube left operand (`[BATCH_TILE,
K_STEP]`) and one tile of the right operand (`[K_STEP, N_CHUNK]`) each
sit just under the per-core buffer budget — i.e. cube `mat_left` /
`mat_right` are close to full. For vector-heavy regions, size the
working tile so the vector buffer is similarly filled.

Practical procedure:

1. Start from the natural problem dimensions (`BATCH`, `HIDDEN`, …).
2. Pick `K_STEP` and the output-chunk size so both cube operands fit
   without spilling.
3. Sweep one tile dim up/down by 2× and re-measure with PMU — keep the
   size that pushes the cube (or vector) unit closer to 100 %.

The K loop is then driven by `pl.pipeline(stage=2 or 4)` so the next
tile's MTE2 overlaps the current tile's compute (see Part 2 item 1).

#### 5. Mixed cube + vector kernels — fewer hand-offs

When the matmul (cube) and its epilogue (cast / add / norm — vector) sit
in separate `pl.at` regions, every projection generates two kernels and
an AICPU hand-off between them. Place both inside the **same** `pl.at`
and the compiler co-schedules cube and vector on the right unit
internally, removing the hand-off:

```python
with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj"):
    for kb in pl.pipeline(0, input_proj_k_blocks, stage=2):
        ...
        q_acc = pl.matmul_acc(q_acc, tile_a, tile_b)     # cube
    q_bf16 = pl.cast(q_acc, target_type=pl.BF16)         # vector
    q_proj[b0:b0 + BATCH_TILE, q0:q0 + Q_OUT_CHUNK] = q_bf16
```

#### 6. `pl.spmd` for parallel sub-kernel dispatch

`pl.spmd(N)` dispatches `N` blocks of an InCore body in parallel from
**one** AICPU schedule entry, instead of N successive `pl.parallel +
pl.at` dispatches. When a region has many parallel chunks and AICPU
overhead is visible per iteration on the swimlane, replace the explicit
`for ... in pl.parallel: with pl.at: ...` pattern with `pl.spmd`:

```python
# qwen3_32b_decode.py: one AICPU dispatch fans out Q_OUT_BLOCKS blocks
for qi in pl.spmd(Q_OUT_BLOCKS, name_hint="q_proj"):
    ...
```

Use `pl.spmd` once the per-iteration body is self-contained and the
AICPU lane shows a dispatch trail; keep the explicit form when you need
to nest named sub-regions inside the chunk.

---

## Part 2 — L1 / L0 tuning (intra-kernel)

Once L2 is balanced, individual kernels become the bottleneck. Two
artifacts drive intra-kernel tuning:

### Capture

PMU counters per kernel:

```bash
python models/qwen3/14b/qwen3_14b_decode.py -p a2a3 -d 0 --enable-pmu 2
# → build_output/<...>/dfx_outputs/pmu.csv
```

Per-kernel intra-core swimlane (MindStudio Insight / msprof simulator
trace), exported after a normal run:

```bash
python models/qwen3/14b/qwen3_14b_decode.py -p a2a3 -d 0 --export-kernel-insight
# → build_output/<...>/kernel_insight_all_funcs_<ts>/
```

Or run the exporter directly on an existing build via
[`tools/export_all_kernel_insight.py`](../tools/export_all_kernel_insight.py).

### Tuning rules

#### 1. `pl.pipeline` for ping-pong on the K loop

Inside a `pl.at` region, the reduction loop of a matmul (the K loop)
should be `pl.pipeline(..., stage=2 or 4)`. The compiler replicates the
loop body `stage` times for ping-pong buffering, so MTE2 (load) overlaps
with cube/vec compute on alternating tiles.

```python
# qwen3_14b_decode.py — stage=4 used for the largest input-proj K dim
for kb in pl.pipeline(input_proj_k_blocks, stage=4):
    ...

# stage=2 is the common default
for kb in pl.pipeline(0, hidden_blocks, stage=2):
    ...
```

A `pl.range` here forces strictly serial K iterations — the cube unit
will stall on every load. Always prefer `pl.pipeline` in the K loop.

#### 2. Watch `pl.slice` / `pl.assemble` granularity

MTE transfers prefer 512-byte aligned addresses and lengths on A3 / 910C.
Pick the trailing-dim tile size so the slice is a multiple of 512 B:

- BF16 (2 B/element) → trailing dim multiple of 256 elements
- FP32 (4 B/element) → trailing dim multiple of 128 elements
- INT8 (1 B/element) → trailing dim multiple of 512 elements

Misaligned slices fall back to slower paths visible as long MTE2 bars in
the kernel-insight swimlane. In the qwen3-14b kernels, all `K_STEP` /
`Q_OUT_CHUNK` constants are picked to keep the inner load 512 B aligned.

#### 3. Read PMU utilization

Recommended PMU counters to collect per kernel:

```
pmu_total_cycles
vec_busy_cycles        cube_busy_cycles        scalar_busy_cycles
mte1_busy_cycles       mte2_busy_cycles        mte3_busy_cycles
fixpipe_cycles
```

What each pipe means in context:

| Counter | Cube kernel (AIC) | Vector kernel (AIV) |
|---|---|---|
| `mte1_busy_cycles` | L1 → L0 (operand staging into cube) | — |
| `mte2_busy_cycles` | GM → L1 (operand load from device memory) | GM → UB (input load) |
| `mte3_busy_cycles` | — | UB → GM (output store) |
| `fixpipe_cycles`   | L0C → GM (cube result write-out) | — |
| `cube_busy_cycles` | cube compute | — |
| `vec_busy_cycles`  | — | vector compute |

The bottleneck pipe should sit near 100 % of `pmu_total_cycles`; the
others run overlapped underneath it. Targets:

- **Cube kernel**: `max(mte2_busy_cycles, cube_busy_cycles) / pmu_total_cycles ≈ 100 %`.
  Either the L1 load or the cube compute is saturated — whichever the
  shape is bound by.
- **Vector kernel**: `max(mte2_busy_cycles, vec_busy_cycles) / pmu_total_cycles ≈ 100 %`.
  Either the GM→UB load or the vector compute is saturated. For very
  store-heavy kernels, `mte3_busy_cycles` can be the bottleneck instead.

If both compute and MTE2 are well below 100 %, open the kernel-insight
swimlane: gaps usually mean (a) a missing `pl.pipeline` on the K loop,
(b) suboptimal instruction scheduling, or (c) incorrectly placed
synchronization barriers.
