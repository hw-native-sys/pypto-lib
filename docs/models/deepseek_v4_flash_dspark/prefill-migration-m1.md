# Isolated prefill M1 evaluation

Tracking: [pypto-lib #1562](https://github.com/hw-native-sys/pypto-lib/issues/1562).
Baseline: [M0](prefill-migration-m0.md). M2 resumes separately; M3 stays paused.
The M1 evaluation retains one bounded change; Attention acceptance remains open.

## Why the earlier migration was not an established win

The earlier patch combined M1, M2 and M3. Its Attention comparisons had lower
fastest-rank medians but higher headline medians; its MoE timing compared
historical allocations after an unchanged baseline replay failed correctness.
Those observations do not establish lower request latency or identify an
individual optimization's contribution. Isolated tests now show that some
proposed fusions actually regress DSpark, despite improving the MTP source case.
No final complete-prefill device measurement was made.

## Retained implementation

`moe.py::prefill_moe` writes complete 128-row waves directly into an `ffn_out`
slice. Partial waves retain fixed-capacity scratch and guarded copy-back.
A scope-local output alias retains the output across full-forward layer scopes.
The completion TaskId joins either the direct-output completion or tail store
before the existing publish/retire and communication-window reuse sequence.
Input staging, communication epochs, expert kernels, quantization, routing order,
public ABI and generic decode implementation remain unchanged.

Input-copy removal was tested separately but adds no measured incremental gain;
it is reverted, including its private dispatch API and new readiness task.
This M1 evaluation includes no compact M2 transport or M3 cache-preparation
change. The subsequent M2 implementation is recorded separately.

## Paired real-device measurements

All rows use a2a3, synthetic production-shape fixtures, frozen M0 inputs and
Torch golden, 100 measured rounds and 5 warmups. Each baseline and candidate
runs serially within one device allocation. Existing comparisons pass before
timing in the accepted rows; timed iterations do not independently validate
persistent state. The primary metric is the fastest rank's median. The headline
is the median of each round's maximum across ranks; it includes rank timing
skew and must not be substituted for a request-latency guarantee.

MoE uses EP8/TP4, 16 experts/card, with both TP4 groups populated by 8192 tokens.
This component surrogate differs from the global single-request EP16 full case.

| Isolated change | Baseline fast ms | Candidate fast ms | Reduction | Baseline headline ms | Candidate headline ms | Disposition |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MoE hash, output view | 23.7869 | 23.1312 | 2.76% | 28.0122 | 26.6882 | Retained |
| MoE gated, output view | 25.1191 | 24.6048 | 2.05% | 97.4307 | 29.9446 | Retained; large headline difference is not attributed to copy removal |
| MoE hash, add input view after output view | 23.1312 | 23.1775 | -0.20% | 26.6882 | 26.5644 | Rejected: no incremental primary-metric gain |
| HCA HC/RMSNorm fusion, TP4 | 22.9514 | 23.1492 | -0.86% | 42.2838 | 60.5424 | Rejected |
| CSA HC/RMSNorm fusion, TP4, experimental Q control | 24.1250 | 26.9106 | -11.55% | 49.1768 | 88.3366 | Rejected |
| SWA HC/RMSNorm fusion, TP4, experimental Q control | 11.9547 | 11.0478 | 7.58% | 77.2689 | 104.0945 | Experimental only; reverted |
| CSA query grouping, TP4, experimental Q control | 24.1250 | 23.5231 | 2.49% | 49.1768 | 39.4923 | Rejected: packed validation fails |
| SWA query grouping, TP4, experimental Q control | 11.9547 | 11.6923 | 2.19% | 77.2689 | 70.5067 | Experimental only; reverted |

The earlier independent hash output-view pair also passes: 23.7123 to
23.2330 ms (2.02%). It is supporting evidence, not pooled with the three-way run.
Large headline changes have not been causally assigned to host skew or a
specific scheduler blocker. In particular, faster minimum-rank time does not
prove that a persistent imbalance has been removed.

## M1 item dispositions

| Item | Current disposition |
| --- | --- |
| HC pre plus RMSNorm | HCA/CSA fusion rejected on performance. SWA fusion remains blocked by unstable Attention baseline; no active fusion |
| HC post grouping | Existing equivalent in DSpark `_hc_post`; no new speedup attribution |
| Complete-wave output copies | Implemented; hash/gated 8K and partial-wave checks pass |
| Complete-wave input copies | Rejected: no incremental gain over output-only and additional ordering changes |
| Query/RoPE copies | HCA already groups copies. CSA grouping fails packed correctness. SWA grouping remains experimental with unresolved baseline failures |
| Scratch reuse after last reader | No additional overlap enabled without a lifetime/dependency proof |
| Compact initial HC input | Deferred: identical streams and public ABI compatibility are not established for all callers |

## Correctness and instability boundaries

The output-only MoE path passes the original hash and gated 8K comparisons and
the 513-token partial-wave case (129 local rows: one full wave and one tail).
These checks use the existing comparators, including their original error
budgets. This is not exhaustive skewed-routing, idle-owner, multi-call, or
prefill-to-decode validation.

Baseline instability remains real: a fresh gated M0 build produces 4096 NaNs
in `ffn_out` and 65536 in `x_next`, while a later original-binary replay and
output-only candidate both pass. The original hash binary also fails on some
replays, including finite errors over the existing budget; a fresh hash baseline
passes and is used for its accepted comparison. Passing later checks does not
explain or erase earlier failures.

Original and freshly compiled M0 Attention cases produce nonfinite output while
cache comparisons pass. A SWA stage probe finds finite QR scales, followed by
nonfinite Q sums on rank 1 heads 17/19/21/23 before sparse attention. Changing
one aligned Q-head loop from pipelined to sequential permits the conditional
8K measurements above. SWA fusion also passes TP4 packed and nonzero-prefix
fixtures under that same experimental change. Standalone decode Q/KV, HCA
packed/prefix, and unchanged CSA packed/prefix controls pass.

However, a real level-4 SWA baseline capture with the same Q-loop experiment
still produces **65536 NaNs**. Therefore the experiment is not a reliable fix;
it and the dependent Attention candidates are reverted. Reduced standalone
Q-dequant tests pass both loop variants, so the integrated failure is not
minimized and no upstream ownership is confirmed. Failure evidence is retained in the module records. No tolerance or golden was loosened.

CSA grouping fails its packed case with 6608/524288 bad output elements
(1.2604%, existing budget 0.5%), with finite outputs and passing caches. The
unchanged CSA control with the same Q-loop experiment passes; grouping remains
rejected. Single-call prefix fixtures do not establish multi-call continuity.

The SWA CP CLI also accepts a swimlane option without forwarding it to its
harness configuration. The first two profiling attempts therefore generated no
chip records and are not traces. A profiling-only snapshot forwards the option
and produces real level-4 records. The failed numerical baseline prevents a
valid before/after performance attribution from that capture. The fusion capture
also fails (512 cache NaNs and 7864320 output NaNs). No profile-based
speedup or named resource-blocker claim is made.

## Integration and evidence

The retained source is M0 fixtures plus the output-view-only MoE change.
Full EP16/TP4/DP4 compilation uses one global 8192-token request and inactive
DP groups 1/2/3. It allocates no full golden snapshot and executes no NPU
forward. The final full device run remains deferred until migration acceptance.
The final compile passes in 51.27 seconds; build
`_jit_l3_prefill_fwd_20261023_212205_w2c3k3_q`. Ruff and whitespace checks pass;
63 existing DSpark/serving contract tests pass. Contract tests cover host
integration contracts, not NPU numerical computation. The exact retained patch,
source hashes and host-check output are stored with the evidence.

Local evidence root: `artifacts/dspark-prefill-m1-20261023/`.
Each module has `invocation.json`, `result.json`, `RECORD.md` and `run.log`;
benchmarks retain raw 100-sample rank arrays and frozen-data paths.
`snapshots/` contains exact source variants; `archive-combined/` preserves the
superseded M1/M2/M3 tree. `environment/selection.json` identifies retained and
rejected variants. No device-only or compile-only result is described as full
mathematical acceptance.

| Evidence | Queue task |
| --- | --- |
| HCA isolated fusion and packed/prefix | `task_20261023_203848_372307910506` |
| Hash three-way M0/output/input+output | `task_20261023_210152_39720006408` |
| Partial-wave three-way correctness | `task_20261023_210518_40253536495` |
| Fresh gated baseline failure | `task_20261023_210515_402459729210` |
| Passing gated baseline/output-only pair | `task_20261023_211601_41622515999` |
| Conditional SWA base/fusion/query | `task_20261023_210743_406430914742` |
| Conditional CSA base/fusion/query | `task_20261023_211011_41169375680` |
| Selected Attention secondary checks | `task_20261023_211559_41616301510` |
| CSA unchanged secondary controls | `task_20261023_211752_346716859` |
| Standalone decode Q/KV control | `task_20261023_211723_418484027734` |
| Actual level-4 SWA capture | `task_20261023_211850_3837310018` |

M1 remains partially accepted, with numerical Attention blockers. The user
rescheduled M2 and deferred remaining M1 experiments on October 24. The retained
MoE output-only change is the baseline for M2; no Attention workaround is enabled.

## Final scheduling experiment before deferral

An isolated four-rank resident HC case tests the original eight-token math with
48 persistent RMS workers, 48 persistent Sinkhorn workers, or both. All three
outputs pass the original comparisons. On the same four allocated a2a3 cards,
with one frozen golden, 100 measured rounds and 5 warmups, the fastest-rank
medians are 1740.5348 ms (baseline), 1752.2670 ms (RMS), 1751.9882 ms (Sinkhorn),
and 1814.4671 ms (both). None is retained. These are HC component boundaries,
not Attention module or full-request timings. A standalone level-4 HC trace
has only 5.64 us of critical-path data-wait gaps; it proves no named scheduler
blocker and does not establish an integrated Attention bottleneck.

Evidence: `artifacts/dspark-prefill-m1-hc-scheduling-20261023/`, including
per-case records, raw arrays, excluded fixture preparations, and the separately
validated level-4 trace. No model-source change from this experiment is active.
