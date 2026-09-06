# Benchmarking Rules

## Core Principle

**Wall time is the objective. Core busy time explains it.**

A change is a win when it lowers the end-to-end device wall time of the case
being tuned. Core busy time — the AIC/AIV time the swimlane and PMU attribute to
actual compute — is the **second** priority: use it to explain why wall time
moved, and to break ties between changes that measure the same.

| Wall time | Core busy time | Verdict |
| --- | --- | --- |
| ↓ | ↓ | Keep. |
| ↓ | ↑ | Keep — wall time wins. A shorter critical path bought with redundant work is a legitimate trade. |
| = | ↓ | Not a win yet. Keep it only if it unblocks a follow-up change, and say so — do not claim a speedup. |
| ↑ | ↓ | Revert. |

**Never report a busy-time or utilization improvement as if it were a speedup.**

For `models/deepseek_v4_flash_lowlat/`, use the
[layer and full-forward evaluation conventions](../../docs/debug-and-tune/performance-tuning.md#low-latency-layer-and-full-forward-comparisons):
single-dispatch level-1 layer effective times, full-forward BENCH effective
times summarized as the lowest per-rank median, and separate level-4 captures
for scheduling evidence. Record the rounds, warmups, fixture bank count, and
validation limits. These are tuning conventions; retain the headline mean
when reporting the CI metric.

Where the numbers come from — see
[`docs/debug-and-tune/performance-tuning.md`](../../docs/debug-and-tune/performance-tuning.md):

| Metric | Source | Quote |
| --- | --- | --- |
| Wall time | `PYPTO_BENCH=1` → `[RUN] effective_us (N rounds) …` | `mean=` (daily CI's per-case number is exactly this field) |
| Core busy time | chip swimlane per-task durations; PMU `*_busy_cycles` vs `pmu_total_cycles` | The bottleneck pipe's ratio |

A `*sim` platform prints `effective_us unavailable: no device-domain spans`. A
simulator run can confirm compile and correctness; it **cannot rank two
variants**. Every wall-time claim needs a real device.

## Reuse Everything — One Compile, One Golden, One Process

Iteration cost is dominated by compile, input generation, and the torch golden —
none of which is the thing being measured. Do not pay for them twice.

1. **Never re-run the same script to collect more samples.** The benchmark loop
   already repeats in-process: `PYPTO_BENCH_ROUNDS` timed rounds after
   `PYPTO_BENCH_WARMUP` discarded ones. N samples means one run with N rounds,
   not N runs.
2. **Freeze the golden once.** Capture with `--save-data`, then replay every
   later run with `--golden-data <work_dir>/data` — input generation and the
   torch recompute drop out. Regenerate only when specs, inputs, or the
   reference computation change. See the `test-with-golden` skill and
   [`docs/run-and-validate/save-and-replay.md`](../../docs/run-and-validate/save-and-replay.md).
3. **Reuse the compiled work dir for untimed iterations** —
   `--runtime-dir <build_output/…>` skips the pypto compile while validation
   logic, `golden_fn`, or the generated `.cpp` / `.pto` change. Reuse it only
   while the kernel source stays compatible with that build: the directory
   carries the program it was compiled from, so a DSL, spec, or shape change
   makes it stale and requires a fresh compile.
   **A `runtime_dir` replay cannot be benchmarked**: there is no live
   `CompiledProgram`, so the harness prints
   `[RUN] benchmark skipped: no live CompiledProgram (runtime_dir replay)` even
   with `PYPTO_BENCH=1`. A timed run compiles — budget for it rather than trying
   to extract a number from a replay.
4. **Batch a sweep into one process.** When comparing K tile sizes or constants,
   prefer a single process that runs all K variants over K invocations. When
   separate processes are unavoidable, they must still share one frozen golden
   directory.
5. **Cut rounds while iterating; restore them for the number you report.** 100
   rounds is ~0.1 s of device time for a decode step but minutes for a long
   prefill or a multi-card run. Any two numbers being compared must come from
   the same rounds/warmup, and CI's baseline is the 100 / 5 default.
6. **Keep the baseline's build and data until the comparison is reported.**
   Re-measuring a baseline you already measured is the same waste as re-running
   a variant.

Neither flag is universal: an entry may expose `--save-data`, `--golden-data`,
both, or neither. Check the target's `--help` first and add the missing argparse
option and its keyword forwarding through the `test-with-golden` skill before
running the loop below.

```bash
# Capture once: compile, generate inputs, compute the golden, freeze it.
PYPTO_BENCH=1 PYPTO_BENCH_ROUNDS=20 python <kernel>.py -p a2a3 -d 0 --save-data
```

```bash
# Every later timed run: same compile path, no input gen, no torch golden.
PYPTO_BENCH=1 PYPTO_BENCH_ROUNDS=20 python <kernel>.py -p a2a3 -d 0 --golden-data build_output/<ProgramName>_<ts>/data
```

## The Frozen Benchmark Datasets

`bench_data/layer{0,2,3}/data` holds the frozen input + golden sets the decode
layer is benchmarked on: layer 0 = SWA, layer 2 = CSA, layer 3 = HCA. They live
outside `build_output/` on purpose, because that directory is wiped routinely and
regenerating a set costs a full compile plus a torch golden. `bench_data/` is
gitignored. Reuse them:

```bash
PYPTO_BENCH=1 PYPTO_BENCH_ROUNDS=50 PYPTO_BENCH_WARMUP=5 \
  python models/deepseek_v4_flash_lowlat/decode_layer.py -p a2a3 --tp 8 \
  -d 0,1,2,3,4,5,6,7 --layer-id 3 --golden-data bench_data/layer3/data
```

**Never compare a number taken on one dataset against a number taken on another.**
How many experts a step activates is a property of the dataset, and it moves MoE
wall time by tens of microseconds.

### The activated-expert count, and why it is pinned below the maximum

The MoE balancer splits (activated experts + 1) work items over `NUM_CORES` = 24.
A step can activate at most `T * TOPK` = 48, and 48 is the one value that pushes a
core to a **third** round while every other core takes two -- a long tail worth
~70 us of `exp_routed_balanced` span. Real routing collides well below the
maximum: 8 tokens drawing 6 distinct experts each out of 256 lands on 44 on
average, and hits 48 about 1.5 % of the time.

Two routing paths feed it, and they are not equally exposed:

| Layers | Path | Selected by | Activation count |
| ------ | ---- | ----------- | ---------------- |
| `layer_id < N_HASH_LAYERS` (0, 1, 2) | hash | `tid2eid[input_ids]` -- a fixture tensor | whatever the fixture makes it |
| `layer_id >= N_HASH_LAYERS` (3+) | score | argsort of the gate scores | data-dependent, naturally ~44 |

The hash path is the one a fixture can get wrong, and it was: a packed
`tid2eid[v] = (v * TOPK + [0..5]) % N_EXPERTS` with `input_ids = arange(T)` routes
the T tokens to 48 *distinct* experts every single run -- simultaneously the
worst case for the balancer and a case real routing essentially never produces.
`moe.decode_route_rows()` now redraws those rows from a fixed seed
(`moe.MOE_ROUTE_SEED`) and rejects a draw that hits the maximum.

**When you regenerate a dataset**, check the count before trusting any number
from it:

```bash
python -c "
import torch
d='bench_data/layer0/data'
t=torch.load(d+'/in/tid2eid.pt'); i=torch.load(d+'/in/input_ids.pt')
print(int(t[0][i[0].flatten().long()].unique().numel()))"   # expect < 48
```

That check is only meaningful for a **hash** layer; on a score layer the count
comes from the gate scores and the tensors say nothing about it. Regenerate only
when specs, inputs, or the reference computation change -- and when you do,
re-freeze all three together and re-state the baselines, because every previously
recorded number was taken on the old sets.

## Parallel Sweeps — One Card per Variant

A single-card entry takes a plain `-d N`, so a box with 8 NPUs can measure 8
variants in the wall-clock time of one. Two conditions make it sound:

1. **Every variant measures its own baseline on its own device**, and quotes
   only the within-device delta. Device-to-device spread is the same order as a
   small effect, so a cross-device comparison silently invents or erases one.
2. **They share one frozen golden directory**, so no variant pays for input
   generation or the torch recompute.

Give each variant its own copy of the model directory — a git worktree does not
carry an untracked one, a plain copy does. When chasing a sub-1 % effect,
interleave baseline and variant back to back on each device so host-side
contention from the other cards hits both sides equally.

This does not apply to an L3 (`l3_*`) entry, which needs the whole card set.

## Interleaving Does Not Cancel Run Order

An A/B/A/B alternation is the usual defence against drift, and on an L3 decode
entry it does not work: the arm that runs **second in a pair is slower whichever
arm it is**, by ~6.7 us on `decode_layer.py`. A fixed alternation gives the two
arms different position distributions, so the bias lands entirely on one of
them and reads as a result.

Two ways out, and prefer the first:

- **One process, many rounds.** `PYPTO_BENCH_ROUNDS` already repeats in-process
  and every round shares one compile, one input generation, and one golden. 300
  rounds in one run is both a better sample and far cheaper than six 50-round
  runs — and it has no run-order bias to cancel, because there is only one run.
- **Balance the positions** when separate processes are unavoidable: A/B/B/A,
  then the same set with the order reversed, so each arm occupies each slot
  equally.

Report the **median** of the fastest rank, not the mean. On an L3 layer the
mean is set by a handful of outlier rounds — host dispatch skew lands in every
round — and moves far more between runs than the effect being measured.

## Distributed (L3) — Drop the Start Skew

Ranks do not start together. A late-dispatched rank spends the head of its
window waiting; that wait is not kernel time, it varies round to round, and it
lands inside the measured window.

**Report the fastest rank, not the headline.** The `effective_us` headline is
the per-round **max across ranks** (the round ends when the slowest card
finishes), so it carries the full start skew. For tuning comparisons, quote the
lowest per-rank mean from the `[RUN] rank N: eff_us … mean=` breakdown.

When the per-rank breakdown is not enough:

- `PYPTO_BENCH_RAW=1` prints every dispatch's sample per rank in order — use it
  to spot start-up drift, a bimodal rank, or one card lagging.
- Capture a per-rank chip swimlane and measure from the first real compute task
  instead of the window start, subtracting the leading wait explicitly.
- `host_union_mean_us` is the opposite convention — it *includes* start skew and
  host dispatch overhead by construction. Never use it as the kernel number.

Keep it honest:

- A persistent gap between ranks that is **not** start skew is a load-balance
  problem. It belongs in the report, not in the discarded wait.
- Fastest-rank is not end-to-end step latency, and the dashboard number is the
  headline max. State which convention a number used, and compare like for like.
- Warmup discards leading launches; it does not remove per-round start skew.

## Reporting a Result

Every benchmark number states: platform and device, rounds / warmup, the metric
and its convention (headline mean vs fastest rank), the baseline it is compared
against, and whether the golden was replayed. A number without those cannot be
reproduced or trusted.

### Every finished optimization reports three things

Report as soon as an attempt is **decided** — not batched to the end of the
session, and not only for the wins. A reverted change gets the same three parts
as a kept one; that is what stops the next session repeating it.

**1. Perf gain.** Baseline → variant with the delta, carrying every convention
above. Give the cumulative figure too when a session has stacked several
changes, so the running total never has to be reconstructed.

**2. Why it wins — or why it does not.** Name the mechanism from the **trace**,
not from the hypothesis that motivated the edit: which line item moved and by
how much (`sched_overhead_analysis` phase split, per-engine starvation,
submit / block / edge counts, per-block exec time, occupancy). Two cases deserve
extra words rather than fewer:

- The change won **for a different reason than predicted** — say so plainly. A
  win you cannot explain will not transfer to the next shape.
- The change **lost**, or won far less than the mechanism suggested it should.
  State which resource you freed and why nothing was waiting on it.

**3. The new trace.** Capture it and hand over the real files — merged swimlane,
CPM traces, the `sched_overhead_analysis` dump — rather than describing them.
Say what changed in the trace's *shape* (a phase that now overlaps, a dead band
that closed, a lane that emptied), not only what changed in the numbers.

This is the report, not the log entry. The lesson still goes to
[`optimization-lessons.md`](optimization-lessons.md)'s file; these three parts
are what the user reads at the moment the attempt is decided.

### Hand a trace over renamed, not as the harness left it

The compile directory is named after the *entry point*, so every decode-layer
capture lands as `_jit_l3_decode_layer_<timestamp>/` and every swimlane inside
it as `merged_swimlane_<timestamp>.json`. Three variants of the same layer are
then three identical-looking directories, and a week later nobody can tell the
SWA capture from the HCA one. Before handing a trace over:

- **Rename the directory to name the case**, keeping the timestamp:
  `traces/hca_layer3_<timestamp>/`. Nothing inside a capture references its own
  directory name, so this is safe -- but a renamed directory is no longer usable
  as `--runtime-dir`, so rename only captures kept for reading.
- **Never rename the files themselves -- symlink them.** The harness filenames
  are a contract: `swimlane_converter`, `--chip-swimlane-records-json`, and the
  VS Code swimlane viewer all open `chip_swimlane_records.json` by that exact
  name. Leave the real files alone and add readable aliases beside them
  (`<case>_rank<N>_swimlane.json` -> `merged_swimlane_<ts>.json`,
  `<case>_rank<N>_chip_records.json` -> `chip_swimlane_records.json`).
- **Keep both viewer entry points.** The merged swimlane is the Perfetto file;
  `chip_swimlane_records.json` is what the VS Code viewer reads directly. Ship
  the pair, not just the merged one.
- **Symlink the fastest rank**, because that is the rank the report quotes and
  the only one whose span is mostly kernel time. Point a `fastest_rank` link at
  its `d0/` directory, and a `<case>_fastest_swimlane.json` plus
  `<case>_fastest_chip_records.json` at its two traces, so the reader opens the
  right file without first working out which rank to trust. Identify it by span
  (`max(ts + dur) - min(ts)` over the `X` events), not by rank index.
- **Leave a `README.md` beside the set** giving the platform, the revision, the
  command that regenerates it, and the per-rank spans.
