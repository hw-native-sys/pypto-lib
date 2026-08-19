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

Where the numbers come from — see
[`docs/debug-and-tune/performance-tuning.md`](../../docs/debug-and-tune/performance-tuning.md):

| Metric | Source | Quote |
| --- | --- | --- |
| Wall time | `PYPTO_BENCH=1` → `[RUN] effective_us (N rounds) …` | `mean=` (daily CI's per-case number is exactly this field) |
| Core busy time | L2 swimlane per-task durations; PMU `*_busy_cycles` vs `pmu_total_cycles` | The bottleneck pipe's ratio |

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
- Capture a per-rank L2 swimlane and measure from the first real compute task
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
