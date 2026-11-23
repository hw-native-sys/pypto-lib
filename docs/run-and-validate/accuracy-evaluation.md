---
id: "validation-deepseek-v41-a3-precision"
kind: "validation"
status: "complete"
evidence_level: "experiment-derived"
reviewed_on: "2026-11-23"
related: ["diag-accuracy-state", "precision-tuning", "golden-harness"]
tags: ["accuracy", "numerical", "state", "distributed", "w8a8", "deepseek-v41"]
---

# DeepSeek V4.1 Flash A3 precision validation

This plan turns the DSV4.1 Flash A3 W8A8 experiment into a repeatable
correctness workflow. It follows the layered structure used by the
[`serving-blueprint`](https://github.com/co-agent-serving/serving-blueprint)
diagnostics and validation documents. The primary deployment contract is
TP4/DP4/EP16; TP8/EP16/DP2 is a separately recorded non-default experiment.
The model uses A3 W8A8 weights, FP32 scales, NZ-packed weights, and the
temporary Engram-disabled scope.

## Scope and acceptance language

The checks below answer different questions. A later check cannot promote an
earlier failure, and a successful EOS cannot promote numerical or quality
status.

| Status | Meaning |
| --- | --- |
| `PASS_NUMERICAL` | The named output passed its independent reference and threshold contract. |
| `PASS_STATEFUL` | Multi-call state, cache ownership, and returned state passed the named checks. |
| `PASS_DISTRIBUTED` | Rank ownership, counts, group reductions, and idle-rank behavior passed. |
| `PASS_REAL_BOUNDED` | A real checkpoint operation passed its bounded source, runtime, and cleanup contract. |
| `UNASSESSED_QUALITY` | Text was observed, but coherence, factuality, or original-model parity was not established. |
| `NOT_ACCEPTED_TRANSPORT` | CLI or bounded facts may be valid, but strict runtime/telemetry coverage is incomplete. |
| `KNOWN_PRECISION_GAP` | A source-level numerical contract is known to differ from the pinned reference. |

The report must retain every applicable status. `CLI=0`, `RunResult.passed`,
finite output, and natural EOS are useful facts, but none of them is a
whole-model precision verdict.

## Stages

Run the stages in order: source and ABI binding, CPU/reference comparison,
operator device correctness, stateful multi-call checks, distributed component
checks, bounded real-checkpoint execution, and finally prompt/data quality
review. Correctness must pass before performance is interpreted. If a capture is
incomplete, inputs or thresholds changed, or the reference is not independent,
the result is `inconclusive` rather than a pass.

## Short generation

Record prompt ID, rendered bytes, token IDs, generated IDs and text, finish
reason, EOS, usage, language, state continuation, batch, chunk, prefix, DP
votes, and speculative behavior. Short generation is a smoke test only; it
cannot replace a dataset score, an independent tensor comparison, or original
model parity.

## Dataset and prompt suite

Freeze dataset version, split, sample IDs, template, few-shot content, thinking
mode, sampling, EOS policy, maximum output, denominator, answer extractor,
semantic scorer, and thresholds. Preserve raw predictions, gold/prediction
pairs, per-question results, summaries, errors, and truncation. For the current
diagnosis suite, include the fixed factual, arithmetic, English, and Chinese
cases defined by the experiment plan; do not select only cases that terminate
within the old three-token budget.

## State and candidate identity

Each candidate source, runtime, weight, config, tokenizer, or verifier change
must be checked for reuse eligibility. A precision change re-runs affected
operator, state, short-generation, and dataset stages. Old artifact scores do
not substitute for evaluation of the new artifact. Each case starts from a
fresh state, and A→B→A replay must prove that one case cannot borrow another
case's cache or mutable buffers.

## Reproducible input contract

Before compiling or allocating a device, record:

- checkpoint, tokenizer, model source, generated artifact, runtime, and
  topology hashes;
- TP/DP/EP/world/rank-group mapping and local expert/head dimensions;
- exact prompt bytes, template, token IDs, token count, start position,
  maximum output budget, seed, sampling mode, and EOS IDs;
- input and output dtype/layout, dynamic axes, page/block metadata, and
  `In`/`Out`/`InOut` directions;
- the reference implementation, answer extractor, comparison tolerances,
  and the expected termination policy.

Plain completion and the checkpoint's shipped chat encoder are different test
contracts. Do not compare their output as if they were the same prompt. A
short prompt is a smoke test only. For text diagnostics, use fixed cases such
as a factual question in plain and chat form, a simple arithmetic question,
Chinese factual completion, English completion, and Chinese completion. Allow
natural EOS and enough tokens to expose repetition; do not teacher-force a
planned token sequence or force EOS.

## Layered validation matrix

| Layer | Reference and checks | Gate | Evidence and limits |
| --- | --- | --- | --- |
| Source and ABI | Bind all current source files, generated files, checkpoint metadata, dtypes, layouts, offsets, scales, NZ shapes, and parameter directions. | Exact hashes and manifests. | A source/golden hash proves identity, not numerical correctness. |
| CPU oracle | Run an independent Torch/host oracle with fixed inputs. Include finite extremes, signed zero, NaN/Inf controls where the contract permits them, and exact integer metadata. | Independent expected values; unchanged tolerances. | Never derive the golden from candidate output. |
| Operator native | Compare the original golden on device and require the harness result to pass, not merely process exit 0. Record absolute/relative error, rel-L2, cosine, percentiles, and bad-point counts. | Use the operator's original `atol`, `rtol`, and error-ratio budget. | `error_distribution` measures shape; it is not itself a pass gate. |
| Stateful multi-call | Start each call from an independent pristine state. Check prefill-to-decode handoff, returned state, A→B→A replay, poison of prior state, cache slot/page addressing, and before/after hashes. | All reference groups and state hashes pass; actual/reference capsules remain separate. | A later callback must not overwrite an earlier capsule. |
| Distributed component | Check TP group isolation, DP votes, EP destination/local-expert ownership, exact count matrices, idle-rank service, inactive output, and score-tail reductions. | Exact integer/count metadata and original numerical gate. | Synthetic component scope does not establish full E40 fit or all topologies. |
| Real checkpoint | Check source reads, one canonical upload, borrowed views, decode static reuse, cache `InOut`, runtime artifacts, cleanup, and actual greedy history. | `PASS_REAL_BOUNDED` only when all named fields and cleanup receipts pass. | Keep transport/telemetry status separate from bounded model facts. |
| Text quality | Review multiple fixed prompts for factual response, instruction following, language, repetition, and termination. | Semantic review with declared expectations; no automatic parity claim. | Without an original-model reference, mark quality and parity `UNASSESSED`. |

## Numerical contracts

Keep the kernel and golden operation order, accumulation dtype, and cast
boundary identical unless the contract explicitly says otherwise. For the A3
W8A8 path, verify:

- INT8 weight packing, per-channel FP32 scales, zero offsets, NZ transpose,
  and direct FP32-to-INT8 quantization;
- no accidental FP32→BF16→INT8 double rounding;
- explicit RNE (`CAST_RINT`) where the pinned reference uses torch-style
  narrowing; do not rely on the default ties-away cast;
- FP32 accumulation and residual boundaries, BF16 output stores, overflow,
  subnormal, NaN, and Inf behavior;
- exact output dtype before comparison, rather than silently casting one side;
- shape, finite-value, integer-index, routing, and page metadata checks before
  comparing floating-point tensors.

Use `ratio_allclose` or the operator's original comparator for acceptance, and
also save an error-distribution report with rel-L2, cosine, absolute/relative
percentiles, and fractions above declared thresholds. When a mismatch appears,
dump a small set of intermediate tensors in dependency order and identify the
first divergent stage. Do not change tolerances or golden arithmetic to make a
candidate pass.

## Stateful and distributed controls

The stateful check has three independent reference groups where applicable:
an unconditioned per-call reference, the original staged reference, and an
independent multi-call oracle. Each call stores its own initial/actual/reference
capsules and hashes. The DSV4.1 C2 retention evidence uses 19 checks per call,
three calls, three groups, persistent cache hashes, selection outputs, and
pool/count metadata; this is the preferred shape for a stateful receipt.

For MoE and TP controls, require all of the following before accepting the
component:

- exact INT32 route/count matrices, including zero rows and high local experts;
- actual and expected payloads in different storage with stable hashes before
  and after audit;
- inactive-rank output exactly zero where the contract requires it;
- EP idle-rank remote service and source/receive totals;
- TP group isolation and the FP32 score tail, not only an identity reduction;
- `RunResult.passed=True` for every case, plus normal image, lock, queue, and
  owner cleanup.

## Mandatory known precision gap

The experiment found a concrete source/reference discrepancy in HC post. The
pinned original A3 kernel uses `CAST_RINT` to narrow the FP32 residual to the
BF16 sublayer dtype, and the original reference performs `y.to(x.dtype)`. The
current A3 `hc.py` and its golden keep the result in FP32 after attention and
FFN. A CPU probe found 15,360 of 20,480 values different with maximum absolute
difference `0.00390625`; BF16 rounding of the current result reproduces the
original probe.

This is a `KNOWN_PRECISION_GAP`, not an upstream compiler finding. Because the
kernel and golden share the omission, their agreement does not prove original
model equivalence. The gap must be resolved or explicitly bound before claims
of original-model numerical parity, and the first divergent HC boundary should
be captured in a real activation differential. The logs do not establish that
this gap caused the observed repeated text.

Evidence: [`repetition_logic_review_1794069747/REPORT.md`](../../experiments/dsv41-flash-w8a8-a3/logs/repetition_logic_review_1794069747/REPORT.md),
[`hc_boundary_result.json`](../../experiments/dsv41-flash-w8a8-a3/logs/repetition_logic_review_1794069747/hc_boundary_result.json),
and the original-boundary references recorded in that report.

## Real-text diagnosis decision tree

When a prompt produces a short, repeated, or apparently irrelevant answer,
diagnose in this order:

1. Verify rendered template, BOS/EOS, token IDs, prompt length, position, and
   answer extraction against the checkpoint tokenizer.
2. Re-run the same fixed prompt with an independent CPU/tokenizer contract and
   compare prefill logits and first-token top-two margins.
3. Compare operator intermediates in dependency order: embedding/LM head,
   RoPE and attention masks, HC post boundaries, RMSNorm/router, W8A8
   dequantization, MoE routing/return, and TP/DP reductions.
4. Run A→B→A and poisoned-state controls to distinguish cache carryover from
   a deterministic numerical drift.
5. Check rank-local expert ownership, DP vote agreement, inactive-rank service,
   and returned cache/page metadata.
6. Only after the above, classify the result as a text-quality issue. Do not
   blame disabled Engram, sampling, or the runtime without a discriminating
   control.

## Failure taxonomy and report template

Classify the first failure as one of:

- **reference/input**: wrong template, IDs, expected answer, dtype, or shape;
- **model precision**: cast, rounding, accumulation, quantization, or HC
  boundary discrepancy;
- **state**: cache/page/slot/returned-state contamination or wrong position;
- **distributed**: rank ownership, route counts, TP/DP group, or reduction;
- **harness**: overwritten capsules, parser/device binding, or comparator
  lifecycle; preserve the runtime result but mark evidence durability failed;
- **transport/telemetry**: queue, cleanup, sampling, PSS, or process-history
  coverage; do not erase bounded numerical facts;
- **quality**: semantic/repetition behavior without a numerical root cause.

Every receipt should contain `status`, `scope`, `source_manifest`, `topology`,
`prompt` and `prompt_ids`, `reference_policy`, `thresholds`, `input/output
metadata`, `checks`, `actual/reference capsule hashes`, `generated_ids` and
text, `decode_calls`, `termination`, `DP/TP votes`, `runtime artifact hashes`,
cleanup/queue facts, timestamps, and explicit `unassessed` fields for original
parity, text quality, complete PSS/peak, and general shapes.

## Evidence from the A3 experiment

- Attention native cases passed the original `atol=0.02`, `rtol=0.02`, and
  `max_error_ratio=0.005` gates for core, packed, and window inputs.
- EP2/TP1 component cases passed exact count totals `12/6/6`, distinct actual
  and reference payloads, inactive output, and idle-rank service; TP2/EP4
  passed group isolation and the 257-column score tail.
- The TP8/EP16/DP2 real checkpoint run generated six IDs and genuine EOS after
  five decode calls. The independent audit recorded 4,363 passed checks and
  five strict telemetry unmet items; bounded facts passed but the whole case
  remained `NOT_ACCEPTED_TRANSPORT`.

These are bounded receipts, not a substitute for applying this plan to the
primary TP4/DP4/EP16 configuration and its multi-prompt quality suite.

## Related references

- [Precision Tuning](../debug-and-tune/precision-tuning.md)
- [Golden harness](golden-harness.md)
- [Compile/runtime workflow](compile-runtime-workflow.md)
- [DeepSeek V4.1 model documentation](../models/deepseek_v4_1_flash/index.md)
- [Serving Blueprint accuracy-state format](https://github.com/co-agent-serving/serving-blueprint/blob/main/diagnostics/accuracy-state.md)
- [Serving Blueprint accuracy-evaluation format](https://github.com/co-agent-serving/serving-blueprint/blob/main/validation/accuracy-evaluation.md)
- [Serving Blueprint numerical-equivalence format](https://github.com/co-agent-serving/serving-blueprint/blob/main/domains/09-kernels/numerical-equivalence.md)
- [Serving Blueprint PyPTO optimization gates](https://github.com/co-agent-serving/serving-blueprint/blob/main/validation/pypto-optimization.md)
