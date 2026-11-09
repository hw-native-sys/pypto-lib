---
name: hw-util
description: Analyze hardware utilization for one SPMD kernel execution on Ascend 950, A2/910B, and A3/910C. Calculate GEMM MFU/MBU or theoretical forward FlashAttention MFU for GQA, MQA, MLA and sparse attention; FA MBU is deferred. Resolve execution spans and device peaks from swimlanes, CANN traces or supplied timings. Use for kernel utilization, MFU, MBU, BMU, GEMM or attention throughput analysis.
---

# SPMD Kernel Hardware Utilization

Default to one SPMD kernel invocation on one logical device. Resolve how that
device maps to physical DIEs and the published hardware specification. Analyze all
its block instances over the interval from the earliest execution start to
the latest execution end. Do not assume aligned starts or finishes. Report
multiple invocations separately, or explicitly label a requested aggregate.
The supplied formulas and calculators cover GEMM MFU/MBU and forward FA MFU.
For FlashAttention, read [the attention reference](references/attention.md) and
use `scripts/calculate_fa.py`; FA MBU is not calculated. For another kernel
type, derive useful work, memory traffic and applicable execution-unit peak
before computing utilization.

Analyze existing artifacts first. Do not launch a device benchmark, change a
kernel, or update reference repositories merely to calculate utilization.
Use [perf](../perf/SKILL.md) when a measurement is requested. Follow the
[benchmarking rules](../../rules/benchmarking.md) for measured comparisons.

## Establish the numerator and denominator

- Resolve the requested operator using source, generated orchestration,
  `name_map.json`, `dispatch_program.json`, and `deps.json` where available.
  Same-named GEMMs may implement different projections or layers.
- Determine effective **per-device** M/N/K, operand computation types, output
  type, scale granularity, and TP/EP layout. Distinguish allocated capacity,
  padded rows, and active tokens. A generated debug runner may replace dynamic
  dimensions with 1; that is not evidence of the captured runtime shape.
- Resolve total block count, work per block, participating physical AIC/AIV
  cores, observed concurrent instances, and any execution waves. Block count
  alone does not prove simultaneous execution. Whole-matrix `2MNK` already
  includes the work of all blocks; do not multiply it by parallelism again.
- Use capture metadata and generated code before current source, which may
  have changed since compilation. For external kernels whose source was not
  snapshotted, disclose that limitation when it matters.
- Match the peak to the chip SKU, core count, precision, and dense/sparse mode.
  Use Cube peak for a Cube GEMM, not Cube+Vector total. Establish whether the
  denominator covers the selected logical device or a whole physical card;
  a logical device is not universally equivalent to one physical DIE. Use all
  available cores within that scope even when fewer participate; a participating-
  core normalization is a separate metric. Do not mistake the trace timestamp
  clock for the Cube compute clock.
- Prefer a supplied whitepaper and local reference checkout. For Ascend 950,
  read [the 950 reference](references/ascend950.md). For A2/910B or A3/910C,
  read [the A2/A3 reference](references/ascend910.md), including the dual-DIE
  accounting and bandwidth evidence rules. Identify source revisions when
  conclusions depend on reference code.
- If the artifact omits shape or quantization, use independent config/source
  evidence or ask for the missing values. Continue locating events while waiting;
  mark any interim calculations as conditional rather than silently guessing.

## Extract time without mixing scopes

### PyPTO chip swimlane

Use `simpler_setup.tools.swimlane_converter.read_perf_data()` from the selected
runtime checkout. Do not decode or join raw cycle streams by hand. Select the
correct rank, dispatch, task ID, and `run_epoch`; never merge repeated runs into
one execution window. Verify raw AICore and decoded physical row counts match;
when scheduler rows exist, also check their count. For each task/epoch, compare
physical rows with `block_num * active kernel_ids slots`, including MIX lanes.

Level 1 suffices for execution-span analysis; dispatch/finish analysis needs
level 2 or higher. Do not require or capture level 4 just to compute throughput.
Use [critical-path](../critical-path/SKILL.md) for requested scheduler attribution.

```text
kernel span = max(block.end_time_us) - min(block.start_time_us)
```

This span includes staggered starts, gaps between execution waves, and tail
imbalance within the invocation. Do not replace it with mean block duration
or summed busy time. Include only instances belonging to the selected run.

Do not sum parallel core durations or divide whole-matrix FLOPs by one selected
core's duration. Include all output-producing stages for the named scope; if
quantization or an epilogue is excluded, say so. For MIX tasks, distinguish an
AIC-only window from completion of the full operator.

Merged-trace rectangles can start at `receive_time_us` and include local setup.
Inspect `kernel-duration-us` and `local_setup_us`; do not assume the visible
rectangle duration is kernel execution time. Report both boundaries if useful.

If explicitly requested, calculate the hypothetical aligned-start/aligned-end
case with the mean core duration. Label it **average-core normalized**, not an
observed latency or a predicted speedup. Equal block work should be verified;
unequal work makes this normalization less informative.

### CANN profiler JSON

Accept a top-level event list or `traceEvents`. Inspect a bounded sample and
metadata before printing events. Hardware `ph="X"` durations are whole-kernel
measurements, not per-core durations. Exclude duplicate host/overlap-analysis
lanes. Preserve large timestamps with `Decimal` when computing offsets.

Resolve events using model ID, stream, task ID, step, neighboring operations,
and source call order. Do not pool events by operator name alone. Verify the
selected event against model code; neighboring operations alone do not prove M/N/K.

For one invocation, use that invocation's execution duration. Only when the
user requests a summary across invocations, use median timing unless another
statistic is requested, and label metrics computed from that median. Separate
execution graphs with different token counts. Report sample count and whether
statistics pool layers or repeat one task. No silent warmup/outlier removal.

## Calculate

MFU expands to Model FLOPs Utilization. For a single kernel, call the analogous
metric **effective compute utilization** and state the adopted MFU convention.
Its numerator must contain theoretical useful work only. Hardware-required
padding, masked work, tile alignment and redundant execution never increase
the numerator, even when they consume actual execution time.

For FA, use [the attention counting rules](references/attention.md): ordinary
GQA/MQA/MLA counts mask-valid pairs; sparse attention counts all semantic selected
entries without an additional mask discount, excluding hardware-added entries
and repeated work. Only calculate FA MFU. The following byte formulas and
`scripts/calculate.py` apply to GEMM.

### GEMM MFU and MBU

MBU expands to Memory Bandwidth Utilization; if the user says BMU, confirm the
intended metric from context rather than assuming a universally standard acronym.

```text
F = 2*M*N*K                              # one multiply-add counts as 2 FLOPs
TFLOPS = F / (t_us * 1e6)
effective compute utilization (%) = 100 * TFLOPS / peak_TFLOPS

Q_read = M*K*a_bytes + K*N*b_bytes
# Scaled GEMMs only:
Q_read_MX = Q_read + M*ceil(K/group)*a_scale_bytes + N*ceil(K/group)*b_scale_bytes
Q_write = M*N*out_bytes
effective read TB/s = Q_read / (t_us * 1e6)
effective write TB/s = Q_write / (t_us * 1e6)
effective bandwidth / peak (%) = 100 * max(read_TB/s, write_TB/s) / peak_TB_s
actual HBM MBU (%) = 100 * max(measured_HBM_read_bytes, measured_HBM_write_bytes) / (t_us * 1e6 * peak_TB_s)
```

Use `Q_read` for GEMMs without scale inputs. For scaled GEMMs, substitute
`Q_read_MX` in the effective read-rate formula to include scale traffic.

Calculate read and write utilization separately over the same execution window,
then use the higher value as GEMM MBU. Apply the same bandwidth denominator to
each direction; do not add their rates for utilization. Report both rates and
keep the sum of bytes only as a traffic diagnostic. For A2/A3, use the shared
1.6-TB/s bandwidth specified in [the A2/A3 reference](references/ascend910.md),
without multiplying it by the DIE count.

Use valid model dimensions for effective FLOPs, not padding or redundant work.
Native scaled matmul still uses `2MNK`; include scale traffic in bytes and scale
handling in time, and use the matching native compute peak. A BF16 dequantize-
then-matmul implementation requires a different compute-path interpretation.

The byte formula assumes each operand is read once and output written once,
per-row/per-column K-group scales, no bias, and no accumulation input C. Adapt
for other recipes, packing, extra stages, or fused operations. For independent
batched GEMMs, sum FLOPs and account for broadcast operands without blindly
multiplying all bytes; for grouped GEMM use actual group sizes, not capacity.

Algorithmic bytes are **not measured HBM traffic**: L2 hits reduce HBM reads,
repeated loads and access granularity may increase them. Report an effective
bandwidth ratio when only shapes are available; do not relabel it as measured
MBU. Keep HBM, L2, and on-chip bandwidth domains separate. Do not clamp ratios
over 100%; investigate cache reuse, timing, bytes, or the chosen peak.

The helper covers one dense GEMM and prints JSON. This illustrative MXFP8
example assumes a 32-AIC 950DT and a hypothetical 100-us execution span;
the dimensions and timing are not a captured workload or performance baseline:

```bash
python .claude/skills/hw-util/scripts/calculate.py \
  --m 1024 --n 1024 --k 1024 --time-us 100 \
  --time-kind kernel-span --peak-tflops 865 --peak-tb-s 4 \
  --a-bytes 1 --b-bytes 1 --out-bytes 2 --scale-group 32 --scale-bytes 1
```

Supply `--hbm-read-bytes` and `--hbm-write-bytes` together for measured traffic
covering the same device and time window, with an explicit `--hbm-scope`.
Use zero for a measured zero-traffic direction; an unknown direction is not zero.
The former combined `--hbm-bytes` input is replaced by these directional counts:
a combined count alone cannot determine their maximum.

The supported attribution scopes are:

- `kernel-attributed`: the measurement identifies traffic belonging to the
  selected kernel.
- `exclusive-window`: the measurement window contains only that kernel's
  traffic; other kernels, transfers, and background traffic have been excluded.
- `device-wide`: attribution or exclusivity is not established. Report the
  measured ratio as device-wide utilization during this window, not the selected
  kernel's utilization.

Only the first two scopes populate `actual_hbm_mbu_pct`. The `device-wide`
scope instead populates `device_hbm_mbu_pct`, leaving the per-kernel field null.
Both use the maximum of the measured read and write utilization. Directional
rates and utilization are also reported and inherit the same attribution scope.
Matching timestamps alone does not attribute traffic: concurrent branches can
contribute to the same device-wide counter. Scope is an evidence-backed input,
not something the calculator can infer from a byte count.

Fractional element bytes support logical packed FP4 estimates, not
alignment/padding overhead. For complex traffic use a separately derived sum.
Omit `--peak-tb-s` when bandwidth is unverified: the helper still reports MFU
and effective TB/s, with bandwidth utilization fields set to null. Do not
invent a bandwidth denominator to complete a report.

## Interpret and report

Return operator identity and source evidence, effective shape/dtype, block count,
participating cores and observed parallelism, earliest start, latest end, span
and timing statistic, FLOPs/byte assumptions, measured-traffic attribution,
peaks, throughput, and utilization. For GEMM, also report separate read/write
bytes and bandwidth; for FA, report the semantic pair count and theoretical
QK/PV FLOPs. Omit MBU and bandwidth fields entirely from FA reports.
Mark unavailable parallelism or endpoint evidence as unknown; a whole-kernel
duration alone cannot establish it. Preserve trace locators so the user can
find the event. Keep generated CSV/JSON under `build_output/` when useful;
do not commit captures.

For concurrent branches, report individual block counts and actual overlap.
Partial core usage by one branch does not establish that the chip was idle.
Use the union span and summed work for a combined metric. Sum reads across the
selected branches and sum writes separately before taking their maximum; do not
sum branch maxima or percentages with different timing windows.
Busy-core counts and PMU pipe-busy ratios are not useful-FLOP or bandwidth
utilization.

There is no universal passing MFU threshold. For a stated read-once cold-memory
model with overlapping reads and writes,
`max(F/peak_compute, Q_read/peak_bandwidth, Q_write/peak_bandwidth)` is an
optimistic time lower bound, not a prediction. Include scales in `Q_read`.
A small-M GEMM may be bandwidth-limited well below 100% MFU.
Judge improvements by comparable whole-kernel or model latency, retaining cache,
profiler, quantization, layout-conversion, and sampling differences. Do not compare
a mean-core hypothesis directly with a CANN whole-kernel duration as a speedup.

Match the user's language. Put display equations in standalone paragraphs,
outside list items; include plain-text formulas if their CLI cannot render math.
