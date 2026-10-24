# Prefill migration progress

Tracking: [pypto-lib #1562](https://github.com/hw-native-sys/pypto-lib/issues/1562).
The current work is **M2 only**. Remaining M1 experiments are stopped by the
user's decision; M3 stays paused. The earlier combined patch and measurements
are archived and are not the active implementation.
See the [M0 record](prefill-migration-m0.md) and [isolated M1 results](prefill-migration-m1.md).
Zigzag CP and its dependent changes remain excluded.

The repository `.venv` uses the M0 pins: PyPTO `c42e22e`, simpler `d8f997f`,
PTOAS `0.67`, PTO ISA `b1d460e`, CANN `9.0.0`, Torch `2.10.0+cpu`.
Activate it with `source .venv/activate-pypto.sh` from the repository root.
These are the main pins resolved for M0, not a newly resolved main revision.

The retained M1 delta removes complete-wave MoE output copy-back.
Hash and gated 8K checks and a 513-token partial-wave case pass. Same-allocation
100-round comparisons show approximately 2.8% and 2.0% reductions in the
fastest-rank median. These are module tuning metrics, not full-request latency.

M1 is **partially retained; remaining work deferred**. HCA/CSA HC fusion is rejected for measured
regressions. CSA query grouping fails packed-request validation and is reverted.
SWA fusion has a conditional timing gain, but the original Attention baseline
produces NaNs; a sequential Q-head-loop experiment does not reliably fix it.
Both the experiment and SWA fusion are absent from active source. No numerical
threshold was relaxed. Existing HC post grouping is reused without speedup credit.

The initial complete prefill remains the only full device measurement. The
retained MoE change is checked by full-forward compilation only; the final full
measurement waits until the migration and numerical acceptance are complete.

Evidence is recorded separately per module under
`artifacts/dspark-prefill-m1-20261023/`. The earlier combined state is preserved
in its `archive-combined/` directory and under
`artifacts/dspark-prefill-migration-20261023/`.

M2 includes a compact prefill implementation, starting from the retained
M1 output-only path. Dispatch stages token/route metadata in UB and consumes
the route map without rewriting it; expert slabs, fused activation/amax/quant
and direct reverse combine use the dedicated prefill path. Input staging stays
enabled. Attention and generic decode computation are unchanged.

Hash/gated 8K and 513-token checks pass. A separate persistent-window test
validates all eight dispatches for each case (24 passes); the original hash
binary also passes eight times, so the historical intermittent failure is not
proved fixed. Expanded five-epoch communication checks pass exactly with
varying owners/scales/weights. No numerical threshold is changed.

Same-allocation EP8/TP4, 16-expert/card standard measurements (100/5, frozen M0
data) show fastest-rank median reductions of 3.55% hash and 9.57% gated versus
retained M1. Hash headline rises 2.00%; gated headline falls 11.19%. These are
module results, not complete-request speedups. EP16 one-global-8K full-forward
compilation passes; the final full device measurement remains reserved.

This change includes the four M2 model files, with 63 existing contract tests
and Ruff passing. M2 remains in progress; root-cause
closure, broader state coverage and final acceptance remain open. See the
[M2 implementation and validation record](prefill-migration-m2.md). New evidence
is under `artifacts/dspark-prefill-m2-debug-20261024/`; rejected attempts remain
under `artifacts/dspark-prefill-m2-20261024/`.
