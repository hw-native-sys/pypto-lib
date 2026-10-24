# Isolated prefill M2 evaluation

Tracking: [pypto-lib #1562](https://github.com/hw-native-sys/pypto-lib/issues/1562).
Current status: the staged-metadata compact path is enabled and passes
the bounded module checks below; historical intermittent precision failure
remains unresolved. No final full-device acceptance is claimed.

The user deferred remaining M1 work and resumed M2 on October 24. The reference
source is M0 fixtures plus the retained M1 complete-wave output view. Attention
and generic decode computation remain outside this evaluation.

## Candidate implementation

The dedicated prefill path has three connected responsibilities:

- Stable destination/expert packing exchanges a source/expert count matrix,
  INT8 token rows and FP32 scales/routing weights. Route maps remain at source.
  Payload transfers walk live counts; receiver scratch uses aligned expert slabs.
- Count-driven experts use M128 main tiles, M96/M64/M32/M16 tails, and an M192
  alternative for aligned 192-row slabs. W1/W3 share a dispatch; SwiGLU, row
  maximum and requantization share the activation stage. Existing INT8 scales,
  clipping, routing-weight placement and BF16 narrowing are preserved.
- Reverse puts read expert output slabs directly. Source combine reads reverse
  windows in original top-k order, accumulates with shared output in FP32 and
  rounds to BF16. No returned-output reorder or route-id transport is needed.

Production windows are wired through `prefill_layer.py` and `prefill_fwd.py`.
The prefill-only route window is removed. Full-wave input staging stays enabled;
the accepted M1 output view and guarded tail store remain. Existing wave
completion, all-rank retirement and signal-clear boundaries are preserved.
The hidden-staging TaskId anchors publication after the preceding wave retires.

The archived implementation needed two transport corrections before reuse:
the compact payload counter advances once per wave, so the layer fixture must
seed it with the wave epoch rather than the generic path's per-expert epoch;
count publication uses draining `put`. The selected candidate also gives each
FP32 scale row a full 64-byte stride. These are lib-side protocol/layout changes,
not proof of an upstream fault or an explanation of every observed mismatch.

The wave remains 128 local rows. Its maximum live-route capacity is
`EP * 128 * TOPK`, plus per-expert alignment padding in compute scratch. This
is count-driven execution, not a claim that all reserved buffers shrink:
source reverse windows grow to retain a worst-case lane for each expert rank.
No larger prefill wave, new router reduction math or shared-expert tile is
accepted in this evaluation.

## Validation boundaries

All device work uses the repository `.venv`, PyPTO `c42e22e`, simpler `d8f997f`,
PTOAS 0.67, PTO ISA `b1d460e`, CANN 9.0.0, Torch 2.10.0+cpu and a2a3 hardware.
MoE integration uses EP8/TP4, 16 experts/card and two TP4 groups containing
8192 tokens each. This remains a saturated component surrogate, distinct from
one global 8192-token request at EP16/TP4/DP4.

| Check | Evidence | Boundary |
| --- | --- | --- |
| Grouped experts | PASS against existing Torch routed-expert golden, original `2e-3` / 1% comparator | One card, 16 experts; counts 0, 1, 16, 17, 32, 33, 64, 65, 96, 97, 128, 129, 177, 192, 193, 257; not a tile performance sweep |
| Compact communication | PASS, exact outputs and counts | EP8, three consecutive epochs: all empty; ranks 0/7 active with 17/128 rows; all ranks active and all routes on one expert owner. Uses the actual dispatch/combine with an independent identity-expert oracle |
| Full-forward integration | 8K compile-only PASS in 65.89 seconds (default 512-token compile also passes) | EP16/TP4, 16 ranks, DP groups 1/2/3 inactive; no full device run or numerical acceptance |
| Hash/gated/partial integration | Hash/gated/partial candidate checks fail; all final M1 controls pass | Existing 8K and 513-token frozen M0 fixtures; no tolerance change |

The independent communication oracle does not validate expert arithmetic. The
expert oracle does not validate communication or production wave composition.
Neither substitutes for passing production integration.

## Performance evidence and failures

An initial same-allocation hash pair passes its original comparisons, with
100 measured rounds / 5 warmups and frozen M0 data. Fastest-rank median:
23.4507 to 22.5032 ms (4.04% reduction). Headline max-rank median:
36.5404 to 32.6263 ms. This pair uses the initial layer-zero candidate, not the
final corrected source, and is supporting evidence only. The subsequent
nonzero-layer run was interrupted after the counter-seeding defect was found.

A later source pair fails before timing: current M1 hash has 5.6200% bad
`ffn_out` elements versus the existing 5% budget; compact v2 has 8.5369% bad
`ffn_out` and 6.0131% bad `x_next` elements. Gated v2 has 4096 output NaNs;
the M1 gated control also fails, with finite errors over its budget. The
513-token M1 control passes while compact v2 has 5.1163% bad `ffn_out` elements.
These attempts provide no accepted performance comparison.

A separate 16-token router-normalization grouping experiment changes no
reduction order but fails with NaNs in both generic and compact 8K controls.
Its compact partial-wave case passes. It is not a demonstrated fix and is not
part of the selected candidate. Ownership of the integrated instability remains
unresolved; passing individual probes does not establish that all M2 code is
correct or that the baseline is repaired.

## Historical compact-v3 rejection

The final `compact_v3` comparison uses the original router grouping and all
three transport corrections described above. On the same EP8 allocation,
both original M1 controls pass hash and gated 8K checks. Both compact cases
fail before benchmarking: `ffn_out` contains 8192 NaNs and `x_next` contains
131072 NaNs; the frozen Torch outputs contain none. All preceding reported
HC/cache comparisons pass. This paired result does not establish that the
baseline is the cause of the compact failure. The first nonfinite producer
has not been isolated; shared/router arithmetic, expert inputs and production
wave composition still require diagnosis.

The final 513-token control also passes. The compact tail-wave case has
10.5219% bad `ffn_out` elements and 7.0712% bad `x_next` elements versus the
unchanged 5% budget. It therefore fails without relying on NaNs. All six final
cases ran to completion in task `task_20261024_002412_236830127479`; the driver's
zero exit means the sweep finished, not that every case passed. Individual
exit codes and comparisons determine acceptance.

| Final production case | M1 control | Compact v3 | Accepted M2 timing |
| --- | --- | --- | --- |
| Hash 8K | PASS; fastest 23.1571 ms, headline 29.0536 ms | FAIL, output NaNs | None |
| Gated 8K | PASS; fastest 24.2806 ms, headline 27.2741 ms | FAIL, output NaNs | None |
| Partial wave 513 | PASS; correctness only | FAIL, finite-error ratios above budget | None |

The control timings use 100 measured rounds / 5 warmups and the same frozen
M0 data and EP8 allocation. Raw rank medians and sample counts were checked.
They are module metrics and carry no candidate speedup credit.

At this rejection point no final M2 speedup was accepted, and compact v3 was not promoted to active
model source. The passing initial 4.04% hash pair is not substituted for this
failure. Candidate source passes Ruff, and full-forward compilation establishes
shape/caller integration only. Larger waves, router/shared-expert tuning and
M192 performance acceptance remain unfinished. M3 stays paused.

## Reproduction records

Local evidence root: `artifacts/dspark-prefill-m2-20261024/`.
`snapshots/base/` freezes the retained M1 source. `snapshots/compact_v3/` holds
the candidate without the router experiment; `environment/m2-candidate.patch`
is the reviewable delta against that base. Each measured/validated case has an
invocation, complete log, numerical results and an individual record. Source
hashes, raw samples, task mapping and failed preparations are retained.

Fixture preparation failures are excluded from model/toolchain diagnoses:
the transport probe initially passed TensorSpecs in the wrong positional order,
used unsupported direct INT8-to-FP32 casting and needed explicit manual-scope
authoring. The final probe corrects those errors and passes on hardware.

The complete device forward remains reserved for final acceptance. The rejected
compact-v3 snapshot remains archived; the following section supersedes its disposition.


## Metadata staging and current local implementation

The follow-up investigation preserves the pinned environment and all original
comparators. `prefill_moe_dispatch_pack` now loads token scales, route weights
and the padded route map into UB before packing. Scalar lookups use those UB
tiles. The pack task no longer rewrites the route map produced by the positions
task: it consumes that map as input, and the source combine keeps using it.
No router/shared-expert arithmetic, output rounding or acceptance threshold is
changed. This change includes that version and its prefill caller/window integration.

This is a bounded defensive change, not a demonstrated general cure for the
intermittent failure. Two ablations both pass hash 8K: original GM reads with
only the redundant write removed, and staged reads with the write retained.
Standalone metadata tests also pass both GM and UB reads, including ordered
reuse and concurrent four-byte MTE stores. The raw test was corrected to remove
an unused TLOAD before the GM read and still passes. These tests do not prove
a scalar-cache coherence defect or identify one unique causal change.

Heavy argument-dump instrumentation fails at runtime; lighter instrumentation
passes. An abnormal scale snapshot from the failed heavy run is a diagnostic
lead only, not evidence establishing the first bad producer. No instrumentation
is enabled for the accepted standard comparisons.

### Current numerical evidence

- Frozen M0 hash 8K, gated 8K and 513-token integration checks pass.
- A separate persistent-window stress run validates every dispatch with the
  existing comparators, including readback of resident outputs: all eight hash,
  eight gated and eight partial-wave dispatches pass. This differs from the
  standard benchmark, which validates its first dispatch only. Stress timings
  include validation work and are excluded from performance reporting.
- The original compact-v3 hash binary also passes all eight stress dispatches
  in that allocation. The historical failure therefore remains intermittent;
  the passing repaired path alone does not establish root-cause closure.
- A stronger communication oracle passes exactly over five consecutive epochs,
  covering empty ranks, uneven active counts, all owners/experts, varying token
  scales and route weights. Both output values and per-expert counts are
  compared. It uses the actual compact dispatch/combine and an identity expert,
  so it remains separate from the expert-arithmetic oracle above.
- EP16/TP4 full-forward compilation at one global 8192-token request, with DP
  groups 1/2/3 idle, passes in 68.21 seconds. This establishes caller and shape
  integration only; no complete device forward has been run in this update.
- Ruff and the existing DSpark/serving contract suites pass (63 tests). The
  published compute AST matches the validated repaired snapshot; publication
  only normalizes whitespace and adds header labels to the new kernel.

Fresh short/tail-wave checks at 1, 127, 129 and 511 tokens also pass; the
129-token case uses start position 1024 and all four use a new fixture seed.
These are recorded separately in the debug artifact tree. Prefix-position
coverage here exercises only the MoE fixture;
it does not establish Attention cache-prefix or prefill-to-decode correctness.

### Current module performance

All four standard cases use the same EP8 allocation, TP4, 16 experts/card,
synthetic frozen M0 inputs/weights, a2a3 and 100 measured rounds / 5 warmups.
Each case has eight raw rank streams with exactly 100 samples; raw medians
match the reported medians. Comparison is against the retained M1 path.

| Case | M1 fastest-rank median | Staged compact median | Reduction | M1 headline median | Staged headline median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hash 8K | 23.1680 ms | 22.3462 ms | 3.55% | 25.8459 ms | 26.3618 ms |
| Gated 8K | 24.3213 ms | 21.9928 ms | 9.57% | 28.0015 ms | 24.8689 ms |

The fastest-rank metric measures the module tuning boundary. Hash headline
latency increases by 2.00%; gated headline latency falls by 11.19%. Headline
includes rank start skew, so this does not establish a hash request-latency win.
These numbers describe the whole selected M2 path versus M1; they do not
isolate the speedup attributable to metadata staging itself. They also do not
predict the one-global-request EP16 full-forward result.

Evidence root: `artifacts/dspark-prefill-m2-debug-20261024/`.
`environment/repaired-candidate.patch` records the delta against retained M1;
`snapshots/coherent_pack_v3/` is the exact validated source. `acceptance/` holds
standard timing pairs, `stress/` holds per-dispatch correctness, and
`transport_varied/` holds the stronger exact oracle. Initial source hashes,
failed preparations, instrumentation runs, ablations and standalone probes
are preserved. M2 remains in progress: router/shared-expert tuning, wider
request/state coverage, final full-device acceptance remain open. M3 stays paused.
