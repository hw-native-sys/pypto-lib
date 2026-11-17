# CPU reference dot proof and microbenchmark — artifact-only

MATH.md provides the bounded exactness proof; exact_dot.py is an isolated
candidate primitive. Original45 model sources and current long-running helper /
process remain unchanged. No NPU, checkpoint payload, full40 golden, large
weight allocation or full-model evaluator cache was used.

Actual K values for the four integer-dot consumers are1280/2304/5120. The helper
checks INT8-domain operands and K<=131071, so no accepted S32 intermediate can
overflow and every corresponding FP64 integer operation is exact. All original
FP32 scaling/cast ordering is preserved. Four frozen AST snippets and18 cases
plus strided/overflow/domain/shape/dtype controls pass bitwise S32/INT64 checks.
Source hashes and full Torch configuration are in results.json.

## Measured CPU scope

Primary fair-interface benchmark115357_417597622890:
M128,N64,K5120, Torch CPU16threads, warmup2 per variant,9 timed samples per
variant, alternating order, identical frozen operands in that process. The
candidate retains original operand-to-INT32 conversions before the checked
FP64 dot; domain scans and all extra conversions are included.

| CPU wall median | Seconds |
| --- | ---: |
| Original S32 expression |0.003902380936779082|
| Exact checked FP64 substitution |0.0008027340518310666|

This small dot measured4.861x faster. It is NOT a measured40-layer oracle speedup,
NPU speedup, native model result or full-model-cache claim. Shared-host load is
recorded; no isolated-host performance guarantee is made. Initial exploratory
115146 timed the raw-INT8 helper interface (~4.496x); that is preserved separately
and is not used as the matched proposed-interface headline.

## Concrete optional next CPU evaluator

A future independent CPU checkpoint-validation scope can copy the frozen model
reference modules into an isolated evaluator directory, verify canonical source
hashes and rewrite ONLY the four identified signed-INT32 matmul call sites to
this checked primitive. Require exact call-count/pattern fingerprints; do not
monkeypatch global torch.matmul or alter BF16/FP32 projections. Original model
source remains canonical. Record derived evaluator hashes and the mathematical
transformation/proof in CPU validation receipts.

Continue BOTH original unconditioned and actual-snapshot staged40-layer
references, direct pool/count gates, unchanged comparator thresholds and the
unconditioned logit metric. No candidate outputs become a golden. The checked
primitive must reject/fallback to original S32 for any unsupported domain/K;
there is no approximate FP32 replacement or overflow assumption.

Checkpoint v2 currently pins canonical imported module file paths/hashes.
An optional transformed evaluator therefore needs an explicit new artifact-only
adapter/receipt binding under a later assignment; it must not overwrite the
reviewer's frozen_source or silently bypass those provenance checks. Benchmark
the full OFFLINE CPU stage only after actual state checkpoints exist, releasing
all NPU owners first. Do not infer full-runtime speedup from this microbenchmark
or retest the currently running inline helper by patching its process.

No production dot/golden/kernel change is proposed in this unit. Main parks
with proof, bounded CPU data and a reviewable next evaluator proposal.
