# Performance Tracking

Performance tracking turns repeated device benchmarks into comparable source
history. A result is valid only when its source, benchmark contract, toolchain,
and device identity are all known. A commit SHA alone is not a performance
identity.

## Separate source and toolchain changes

Maintain independent time series for source changes and dependency changes:

| Lane | Source under test | Fixed resources | Cases |
|------|-------------------|-----------------|-------|
| `lib-main-attention` | Each selected `main` snapshot | One designated device | DeepSeek-V4 Flash CSA, HCA, and SWA |
| `dsv4-eplb-branch` | Observed tips of the maintained EPLB branch | One ordered eight-device set | MoE EP8, Decode Main, and Decode MTP |
| `toolchain-bridge` | One selected source anchor | Same resources as the owning lane | Cases affected by a dependency update |

The source lanes keep the complete PyPTO dependency chain fixed. A toolchain
upgrade is measured at the same source anchor before starting a new toolchain
epoch. Results from different lanes or epochs must not be connected by a
normal source delta.

## Performance identity

The comparison identity contains:

```text
lane
+ source tree
+ case contract
+ toolchain epoch
+ device epoch
+ lineage generation
```

The source commit and branch name remain useful provenance, but the source tree
identifies the content that ran. The case contract covers the command,
workload, sampling configuration, metric selection, and validation policy.

Record the complete toolchain dependency chain selected by PyPTO:

```text
PyPTO commit
  -> simpler commit
    -> PTO-ISA commit
  -> PTOAS version and artifact checksum
```

Also record Python, Torch, torch-npu, CANN, driver, firmware, relevant runtime
environment variables, and build provenance. Do not combine a PyPTO update
with a pypto-lib source comparison.

A device epoch includes the host, ordered logical-to-physical device mapping,
topology, driver, and firmware. Replacing a device or changing that mapping
starts a new epoch. An unavailable designated device defers the official run;
it does not authorize an automatic fallback into the same time series.

## Benchmark contracts

All maintained cases use five warmup rounds, 100 measured rounds, raw samples,
fixed deterministic fixtures, and fail-closed validation.

### Main attention

Run CSA, HCA, and SWA on `a2a3` with `start_pos=8192`, L2 swimlanes disabled,
and the suite's designated single device. The official value is the
`effective_us` median. Both `kv_cache` and `x_out` must pass numerical
validation before a value is accepted.

### EPLB branch

Run the three cases sequentially in one allocation of the configured ordered
eight-device set:

1. MoE EP8 with 16 experts per rank, eight tokens, balanced routing, and L2
   swimlanes disabled. Select the minimum of the eight per-rank medians and
   require `x_next` numerical validation to pass.
2. Decode Main / Compare3. Select the minimum of the eight per-rank medians.
   This is a runtime-only contract until the entry point has a numerical
   reference.
3. Decode MTP / Compare4. For each rank, select the median of slot 0 compute
   samples, then take the minimum rank median. Preserve slot 1 cleanup timing
   as a diagnostic, but exclude it from the official metric. All declared
   finite-output checks must pass.

The generic multi-rank headline is not a substitute for these case-specific
selection rules. Store all rank distributions and identify the selected rank
and device so a fastest-rank change remains visible.

## Source history

For a squash-merged `main`, each commit is a stable source snapshot. Process
commits oldest to newest and retain a window overlap so a missed scheduled run
can be recovered.

A long-lived development branch can be rebased or force-pushed. On every
observed tip, record the tip and tree commits, merge base, ahead/behind counts,
observation time, patch-stack digest, and lineage generation. Archive the tip
under a tracker-owned Git ref before queuing a benchmark.

- A fast-forward tip continues the current lineage.
- A non-fast-forward tip starts a new lineage generation.
- An identical tree may reuse an existing measurement through an explicit
  alias.
- A changed tree requires a new measurement.
- A non-fast-forward boundary uses a same-device old/new bridge rather than a
  normal previous-tip delta.

The tracker observes published branch tips; it never rebases the maintained
branch itself.

## Storage and finalization

Append one raw record immediately after each case completes. Raw JSONL and logs
are immutable evidence; retries add records instead of replacing them. A
separate finalization step folds attempts, prefers a contract-valid successful
measurement, orders source history, and computes compatible deltas.

Published measurements are frozen. An overlapping rerun of an already
published source snapshot must not silently change the value used as the next
snapshot's baseline. Publishing uses the complete measurement identity as its
idempotency key, not only the source SHA.

Treat an empty or partially valid result set as a failed collection. A
successful orchestration exit without the required case coverage must not
republish an older report as though a new measurement succeeded.

## Regression confirmation

Begin a new lane or epoch in shadow mode. Collect at least ten independent
suite attempts; the 100 samples inside one attempt measure runtime dispersion,
not host-to-host or attempt-to-attempt noise. Calculate an attempt-level median
and median absolute deviation for each case before setting gates.

When a candidate exceeds both its calibrated relative and absolute threshold,
confirm only the affected case in one allocation using:

```text
parent -> candidate -> candidate -> parent
```

Report a source regression only when the confirmation uses the same contract,
toolchain epoch, device epoch, and lineage generation. Precision or contract
failure is a test failure, not a performance number.

## Branch promotion

When the maintained branch lands on `main`, preserve the final branch tip and
the resulting main commit. Benchmark both under the same contract, toolchain,
and device allocation. Seal the branch lane, record the promotion relationship,
and start a new `main` lane at the landed commit. A metric-semantic change
requires a new contract version rather than a bridge to the old contract.
