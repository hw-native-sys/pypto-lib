# On-Device InCore Timestamp Profiling for Multi-Core AscendC Extern Kernels

Use on-device timestamps when an L2 swimlane identifies a slow fused task but
cannot show which phase inside the task is responsible. This technique is most
useful for mixed AIC/AIV extern kernels, cross-core barriers, and producer to
consumer fusion on real inputs.

This is diagnostic instrumentation. It changes the extern ABI, consumes
registers, writes an extra GM output, and adds a cache writeback at the end of
the task. Remove it before reporting production performance or submitting the
kernel unless permanent telemetry is an explicit requirement.

For single-core instruction and pipe analysis, use the simulator workflow in
the [`incore-profiling` skill](../.claude/skills/incore-profiling/SKILL.md).
For task-level scheduling, start with the L2 swimlane described in
[`performance-tuning.md`](performance-tuning.md).

## Define event semantics before adding probes

A timestamp records when the scalar read executes. It does not wait for
previous asynchronous MTE, vector, or cube work to complete. For every event,
write down whether it represents:

- work being issued;
- an existing pipeline drain completing;
- the last active worker completing a phase;
- arrival at a collective; or
- release from a collective.

Prefer existing semantic boundaries. Adding a `PipeBarrier`, `dsb`, or another
synchronization instruction only to make a timestamp easier to interpret also
changes the behavior being measured. When synchronization itself is under
test, vary it as a separate experimental factor.

Use the public system-cycle API when the target supports it:

```cpp
static __aicore__ __attribute__((always_inline)) uint64_t read_cycle() {
  return static_cast<uint64_t>(AscendC::GetSystemCycle());
}
```

The underlying counter, its frequency, and whether samples are comparable
across AIC, AIV, and L2 records are architecture-dependent. Confirm those
properties on the target before taking cross-core minima or maxima. Keep all
analysis in integer cycles and convert to time only at the presentation layer;
do not copy a cycles-per-microsecond value from another platform.

## Capture the true entry and completion boundaries

If the total is meant to include the complete extern body, take the first
sample as the first executable statement in `kernel_entry`, before metadata
acquisition, tiling reads, or argument parsing:

```cpp
extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
  uint64_t entry_cycle = read_cycle();
  // Metadata acquisition and the kernel body follow.
}
```

Pass this value into an implementation helper rather than sampling again in
the helper. Likewise, place the final body sample after the work of interest,
but before the timestamp row is written. The L2 task end is required to account
for that writeback and the return tail.

## Give each core one cache line

Allocate a two-dimensional `INT64` output with one aligned data-cache line per
logical core. On a target with a 64-byte line, eight `uint64_t` values form one
row:

```text
timing[num_core_slots][8]
```

One owner per row avoids write races and false sharing. For a mixed kernel, a
typical collision-free mapping is:

```text
AIC slot = block_idx
AIV slot = num_aic + block_idx * aiv_per_aic + sub_block_idx
```

Verify the actual persistent-launch topology and the ranges of `block_idx` and
`sub_block_idx`; do not assume they equal an individual task's apparent launch
shape. If multiple invocations or layers can write the same output concurrently,
add an invocation dimension or a disjoint invocation offset. A single shared
matrix otherwise retains only the last writer or introduces races.

Keep timestamps in scalar registers while the measured body runs. Write them
once at the tail, then use the target-supported cache clean/writeback operation
so the host copy sees them:

```cpp
__gm__ uint64_t *row = timing + static_cast<uint64_t>(slot) * 8;
row[0] = entry_cycle;
row[1] = producer_done_cycle;
row[2] = collective_arrival_cycle;
row[3] = collective_release_cycle;
row[4] = body_done_cycle;
dcci(reinterpret_cast<__gm__ void *>(row),
     cache_line_t::SINGLE_CACHE_LINE,
     dcci_dst_t::CACHELINE_OUT);
```

The cache operation makes the DFX output visible; it is not part of the
production synchronization being measured. Direct scalar GM stores followed by
one cache-line writeback are sufficient on the verified C220/A3 setup. Recheck
API support and cache-line size when moving the pattern to another architecture
or CANN version. If one core writes more timestamps than fit in a cache line,
give it additional exclusive lines and write back every modified line.

## Thread the output through the extern ABI carefully

Declare the timing tensor as an output so the runtime copies it back to the
host. In pypto, adding this tensor can change both the return ABI and every
scalar's packed `args[]` position:

- extern arguments are packed as all tensors first, then all scalars;
- output-like parameter order affects extern return binding; and
- a new tensor inserted before the scalar shifts that scalar's packed index.

Inspect the generated orchestration's `add_input`, `add_output`, and
`add_scalar` order after adding the probe. Restore and recheck the original ABI
when removing it. See [`cce-extern-kernel-guide.md`](cce-extern-kernel-guide.md)
for the complete extern argument and return rules.

## Capture the actual device output

The golden harness can initialize the expected timing tensor to zero, but that
does not make the saved golden file a runtime dump. With `save_data=True`, files
under `data/out/` are golden outputs written before device execution. Reading
that file will therefore make working device stores appear to be all zero.

Capture the first argument passed to the timing output's custom comparator; it
is the actual D2H tensor:

```python
captured: dict[str, torch.Tensor] = {}


def capture_timing(
    actual: torch.Tensor,
    _expected: torch.Tensor,
    **_kwargs,
) -> tuple[bool, str]:
    captured["timing"] = actual.clone()
    return True, ""
```

Use this bypass only for the DFX tensor. Numerical outputs must retain their
normal comparison. Immediately validate the captured matrix:

- the expected core slots were written;
- required timestamps are nonzero;
- events within each participating row are monotonic; and
- inactive cores are excluded with an explicit mask.

Clear the output before every invocation, or store a generation ID, slot ID, or
magic value in each row. Nonzero monotonic timestamps alone cannot distinguish
the current invocation from a stale row left by an earlier run.

Persist raw tensors, L2 JSON, logs, and analysis scripts under `build/`, not
`/tmp`, so a long investigation survives cleanup without adding generated data
to the commit.

## Build one exact global partition

Do not add independently computed per-core phase maxima. Different maxima can
come from different cores, and early-arrival waiting can overlap work on other
cores. Instead, reduce absolute timestamps into consecutive global boundaries.

For a producer, collective, and consumer body, define:

```text
E = min(entry over all participating cores)
R = max(producer_done over active producer workers)
B = max(collective_arrival over all participants)
A = min(collective_release over all participants)
F = max(body_done over all participating cores)
```

The active-producer mask is part of the kernel's scheduling semantics. A worker
that finishes early remains in the set, while a core that never executes the
producer phase must not contribute to `R`.

The body partition is then:

```text
(R - E) + (B - R) + (A - B) + (F - A) = F - E
```

Check boundary monotonicity and this identity in integer cycles. The intervals
represent the global envelopes of the phases; they are not the sum of work
performed by all cores.

If `M0` and `M1` are the earliest L2 task start and latest L2 task end, a full
task partition also includes entry and writeback tails:

```text
(E - M0) + (R - E) + (B - R) + (A - B) +
(F - A) + (M1 - F) = M1 - M0
```

Only combine L2 and InCore absolute boundaries after confirming that both use
the same unit and absolute epoch. Validate the complete ordering:

```text
M0 <= E <= R <= B <= A <= F <= M1
```

Matching counter frequencies are insufficient if the epochs differ. If the
ordering fails, compare durations within each source rather than subtracting
timestamps across sources.

## Interpret a collective without double-counting

For per-core arrival `b[i]` and release `a[i]`, report these diagnostics:

```text
arrival skew          = max(b) - min(b)
first-release service = min(a) - max(b)
release skew          = max(a) - min(a)
full-release tail     = max(a) - max(b)
early-core residence  = max_i(a[i] - b[i])
```

`first-release service` is the closest boundary measurement of collective
service after the last participant arrives. `early-core residence` includes
arrival skew, so a large value does not imply that the collective primitive is
intrinsically slow.

When the consumer interval starts at `min(a)`, release skew is already covered
by the consumer's global envelope. Adding release skew again double-counts it.
Also inspect the target CANN implementation: a high-level collective may contain
an internal pipeline barrier, so the measured API interval includes that work.

## Compare a merged task with split tasks

Use consecutive boundaries for both sides of the comparison. For the merged
producer and consumer task:

```text
producer-equivalent = B - M0
gap                 = A - B
consumer-equivalent = M1 - A
total               = M1 - M0
```

`producer-equivalent` includes dispatch-to-entry and drain/arrival tails. It is
therefore a fair partition of the merged task total, not a claim about pure
producer instruction time.

For an ordered producer to consumer chain, let `P0/P1` be the earliest start and
latest end of the producer task, and `C0/C1` the corresponding consumer bounds:

```text
producer = P1 - P0
gap      = C0 - P1
consumer = C1 - C0
total    = C1 - P0
```

This three-term total covers the complete envelope when `P0 <= C0` and
`P1 <= C1`. A negative split gap indicates task overlap and must be preserved
rather than clamped. For arbitrary overlapping tasks, use the full envelope:

```text
total envelope = max(P1, C1) - min(P0, C0)
```

Run the merged and split cases alternately on the same device with identical
inputs and configuration. Report each paired run and its total delta. Do not
sum independently aggregated phase medians; those statistics need not equal the
median total.

## Account for probe overhead

Timestamping perturbs the task through cycle reads, register pressure, the extra
output/ABI, GM stores, and cache writeback. D2H copying and host analysis happen
after the measured task in the usual harness; they affect end-to-end harness
cost and possibly later scheduling, but not the completed `M0` to `M1` envelope.
The final InCore timestamp does not include its own row writeback, so the
`M1 - F` tail is the direct way to retain the device-side writeback cost in a
full task partition.

Measure two consecutive cycle reads repeatedly to estimate read cost and noise,
especially when studying a collective near one microsecond. Also measure
instrumented and uninstrumented L2 task envelopes separately. Instrumentation
can change scheduling in either direction, so confirm every final conclusion on
the uninstrumented kernel; do not mechanically subtract a single overhead
sample.

## Removal checklist

Before submitting a production kernel:

- remove entry and phase cycle reads;
- remove the per-core GM row stores and cache writeback;
- remove the timing tensor from the extern, callers, golden specs, and outputs;
- restore every tensors-first/scalars-last `args[]` index;
- remove capture-only comparators, CLI flags, and analysis printing;
- rebuild generated orchestration and inspect the restored ABI;
- rerun numerical validation on the uninstrumented kernel; and
- rerun performance measurements, because the DFX build is not the final binary.

Common mistakes are a late entry sample, sampling asynchronous issue instead of
completion, multiple cores sharing a cache line, reading the saved golden output,
including inactive cores in a phase boundary, reusing one matrix across concurrent
invocations, summing independent per-core maxima, omitting entry/writeback tails,
hard-coding timer frequency, and changing synchronization semantics while adding
the probe.
