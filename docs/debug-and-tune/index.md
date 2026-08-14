# Debug and Tune

Use these guides after a kernel can compile and run. Start with the symptom,
then move from the broadest evidence to the narrowest:

1. Reproduce correctness failures with fixed inputs.
2. Identify whether the problem is compilation, runtime scheduling, numerical
   precision, or performance.
3. Capture task-level evidence before instrumenting an individual kernel.
4. Change one independent variable at a time and revalidate correctness.

## Choose a guide

| Goal | Start here |
|------|------------|
| Diagnose compile errors, runtime failures, hangs, or missing dependencies | [Debugging](debugging.md) |
| Diagnose numerical drift and choose comparison thresholds | [Precision Tuning](precision-tuning.md) |
| Measure end-to-end time and inspect the task schedule | [Performance Tuning](performance-tuning.md) |
| Maintain comparable performance history across source revisions | [Performance Tracking](performance-tracking.md) |
| Choose matmul row, N, and K tiles | [Cube Tile Tuning](cube-tile-tuning.md) |
| Inspect one generated kernel in the operator simulator | [In-Core Simulator Profiling](incore-simulator-profiling.md) |
| Partition phases inside a multi-core CCE extern kernel on real hardware | [CCE In-Core Profiling](cce-incore-profiling.md) |

## Evidence hierarchy

Prefer evidence that changes the program least:

1. Existing compile reports and validation output.
2. Repeated device benchmarks and L2 swimlanes.
3. PMU counters and simulator traces for one kernel.
4. On-device instrumentation added to an extern kernel.

Simulator traces are cycle-accurate for the generated standalone case, but
they do not reproduce the full multi-core schedule or necessarily use real
control inputs. On-device timestamps preserve the real workload, but the
instrumentation changes the kernel ABI and adds overhead. Use the two methods
for different questions rather than treating either as a universal timing
source.

## Keep experiments reproducible

- Save or replay one input set when comparing precision or performance.
- Use the same platform, device topology, runtime configuration, and benchmark
  round count for every candidate.
- Report the median of repeated measurements and retain the raw samples.
- Keep generated traces and build products under `build_output/`; do not commit
  them.
- Remove diagnostic instrumentation and rerun validation before submitting a
  production change.
