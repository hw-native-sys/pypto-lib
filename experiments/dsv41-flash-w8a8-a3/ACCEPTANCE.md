# Acceptance criteria

## Current acceptance ledger (2026-11-06)

Scope is goal.md decisions1–9, including genuine dynamic batch/S, paged state,
TP4/DP4/EP16 and the W8A8 checkpoint. Existing calibrated thresholds remain
unchanged; no new threshold agreement is inferred from this documentation sync.
COORDINATION.md supersedes the historical owner/lifecycle prose below: Main
implements and self-tests, an observer-owned independent reviewer/validator
executes acceptance on the frozen manifest, and observer records the decision.
Observer does not run tests. Every CPU/NPU check uses task-submit; no simulator
acceptance. Historical simulator/device results remain provenance only.

| Requirement / bounded repair | Current disposition | Evidence / limitation |
| --- | --- | --- |
| Environment | Rev3 pins retained for current accepted runs | ENVIRONMENT.md rev3 and independent reviewer execution records; rev1 below is historical |
| Generic indexer / source-pool consumption | Component-level ACCEPTED | TASKS observer acceptance; logs/observer_indexer_verify_1793930090 and observer_reindex_verify_1793934174; checkpoint2048 capacity still OPEN |
| F01 common MLA equation | Bounded ACCEPTED | logs/native_review_f01_1793935855 and observer_native_transition_1793935573; exact original torch_npu/tie parity unproved |
| F02 inverse attended-value RoPE | Bounded ACCEPTED | logs/native_accept_f02_1793946300/REVIEW.md;33 independent positive device cases/two numerical controls and CPU forward compiles |
| F03 compressed-KV basis/publication | Bounded ACCEPTED | logs/native_accept_cohort_1793951619/REVIEW.md; four source paths and eight numerical controls; F04 coordinates excluded |
| F08 MoE ties | Rank-local bounded ACCEPTED | Same independent cohort review; legal/unique IDs and unbiased weights; distributed lifecycle remains OPEN |
| F14 folded expert payload | Conditional synthetic ACCEPTED | Same independent cohort review; actual-checkpoint nonzero-offset incidence unproved |
| F04 coordinates / F10 capacity stride | ACTIVE, unaccepted | logs/cohort_f04_f10_1793952989; exclusive ownership in TASKS |
| Current complete prefill/decode | CPU compilation evidence only; device/numerical closure OPEN | Post-repair independent forward compiles exist; older prefill-green/decode-red runs are historical, not current-tree acceptance |
| Structural / layer / distributed requirements | Current affected-revision closure OPEN | Preserve historical controls and Main16-card evidence; ordinary suites, standalone TP and unequal/idle dynamic lifecycle remain open |
| Engineering / discovery / interfaces | OPEN | CI model-directory ownership missing; MODULES current overlay and INTERFACE caveats supersede stale planning rows |
| Real-weight generation / peak HBM / closeout | HELD / OPEN | Engram, generation binding/state/hashing, pool/state and semantic findings unresolved; no accepted real-checkpoint generation |

Exact commands, task IDs, allocations, exits, source hashes, reference provenance
and output verdicts are in each linked artifact directory and TASKS. A bounded
repair does not close the corresponding complete-model requirement. No global
acceptance or goal completion is claimed.

## Historical launch checklist and enduring criteria

The original launch table/status updates below are retained for provenance.
Their TBDs, rev1/simulator checks and old execution-owner statements are not the
current status; use the ledger above, latest TASKS and COORDINATION.md. Required
semantic, structural, lifecycle, generation and engineering criteria still apply.

## Required delivery gates

| Requirement | Source / artifact | Verification command | Required result |
| --- | --- | --- | --- |
| Reproducible execution environment | `ENVIRONMENT.md` rev 1 | `import pypto, torch; from golden import run`; `$PTOAS_ROOT/bin/ptoas --version`; `hello_world.py -p a2a3 --sim` | Selected integration pins, imports and smoke test are verified; deferred device steps remain visible. Status 2026-11-02: simulator evidence PASS; device steps deferred |
| Complete-model compiled prefill | TBD | TBD | Graph covers embedding, every layer, final norm, and head |
| Complete-model compiled decode | TBD | TBD | Same coverage; consumes and updates agreed persistent state |
| Prefill execution structure | TBD | TBD | Multi-token within-layer work; actual loop/shape evidence matches the frozen contract |
| Structural check negative control | TBD | TBD | Tokenwise complete-stack prefill is rejected, including renamed entry wrappers |
| First working paths | TBD | TBD | Attention and MoE/feed-forward structure checked before full-stack expansion |
| Engineering structure | TBD | TBD | Current ownership/call graph follows model rules; validation exercises delivered symbols and variants |
| Default validation and discovery | TBD | TBD | Bounded entries use existing harness/comparators and applicable CI discovery recognizes them |
| Prefill-to-decode handoff | TBD | TBD | Actual compiled entries pass state and position checks |
| Agreed shape and capacity limits | TBD | TBD | Supported boundaries pass; unsupported inputs follow the contract |
| Real-weight mapping | TBD | TBD | All included weights mapped; exclusions and transforms justified |
| Generation execution path | TBD | TBD | Real device invokes the delivered compiled model entries |
| Documentation and interface agreement | TBD | TBD | Delivered paths, arguments, limits, and commands match sources |
| Memory feasibility | TBD | TBD | Per-rank resident and peak bytes fit observed HBM with reserve |
| Distributed MoE, when in scope | TBD | TBD | Exact card count; distinct DP inputs; local/remote/skewed routing and combine match reference |
| Distributed lifecycle, when in scope | TBD | TBD | Repeated epochs, state isolation and unequal EOS/idle ranks complete safely |

Record entry symbols, source hashes, compile commands, runtime invocation paths,
and generated graph coverage. Placeholder files, aliases, Python per-operator
submission loops, and compile-only success do not pass the model delivery gates.

## Engineering gates

Use the [model rules](/mnt/data/pyptouser/yangyaodong/code/model-harness/rules/models.md) as the canonical criteria.
The owner records evidence in the existing `MODULES.md` and `TASKS.md`:

- Before implementation: complete the file-ownership and validation map.
- On the first working attention and MoE/feed-forward paths: inspect actual
  dependencies, call depth, phase shapes, and default validation; resolve structural
  findings before expanding the pattern to the full model. Check each distinct
  mixer as it becomes executable.
- At closeout: update the map, identify the exact symbols and variants called by
  validation and composition, and account for obsolete paths and their consumers.
  Execute the bounded default validation entries and applicable existing CI
  discovery checks, recording commands, exit codes and discovered entries.

An entry that validates another variant does not accept the delivered variant.
Validate every delivered variant; retain other implementations only with an
explicit consumer and their own acceptance evidence. Repository sources must
contain the permanent validation code. Frozen fixture data and logs may remain
in experiment artifacts. Reuse
existing harness/comparator capabilities and add only missing model semantics.
Ruff success, source hashes, or generation cannot substitute for these checks.
Document permitted exceptions under the existing rules with actual consumers
and responsibilities; unresolved violations keep the corresponding row open.

## Execution-structure gates

These gates are required even without performance targets. Freeze representative
supported prefill lengths greater than one and a decode fixture. Inspect the
actual source call paths and generated IR/task graphs after helper expansion.
Record the following for every fixture and phase:

- Source/build identity, entry symbol, command, and actual checker result.
- Loop nesting and bounds; which loop encloses the complete decoder stack.
- Logical token axes, token tiles, and effective projection, attention/recurrent,
  feed-forward, and routed/shared-expert operand shapes.
- State updates and dispatch/combine granularity, including worst-case MoE receives.
- Evidence paths and locations supporting each conclusion; unknown evidence fails.

Require token tiles greater than one in prefill projections and feed-forward
work. Allow tiled scheduling and necessary causal/recurrent scans. Reject an
outer prompt-token loop that executes the complete decoder stack per token;
numerical equivalence, readable generation, and a compiled loop do not waive this
gate. Two identical entry wrappers are not proof of phase specialization, while
shared helpers with verified phase-dependent behavior are allowed.
Count logical prompt tokens rather than heads or padding. Sparse expert groups
may be empty or contain one row; require a repeated-expert fixture to demonstrate
that the implementation can group multiple tokens for expert computation.

Use the existing validation entry and inspection tools, adding only necessary
model-specific checks, against the real compiled path and
a deliberately tokenwise full-stack negative control. The latter must be
rejected even after renaming the entry. Check generated shapes and nesting,
not only source spelling, kernel names, or self-reported booleans. Missing or
unrecognized evidence must not produce a pass. Save both actual results before
complete-model generation. A tokenwise-only exception requires an explicit user
scope decision and must remain visible in the final delivery status.

## Numerical gates

| Module / composition | Fixture | Output or state | Comparator / threshold | Command |
| --- | --- | --- | --- | --- |
| per-operator modules | owner fixtures (bounded) | projections/caches/state | MODULES.md validation map (existing `ratio_allclose` / `topk_pair_compare` / exact) | `python models/.../<module>.py -p a2a3 -d $TASK_DEVICE` |
| 12 mode entries (paged) | two-request multi-page plan | partial, Top-K/pool, published caches | partial 2e-2/1e-2/1e-2; caches 2e-2/1/64/1e-2; Top-K calibrated; pool set semantics | per mode file, one card |
| EP16 MoE | 16-rank two-token tile | routed | ratio_allclose 2e-2/2e-2/1e-2 | `prefill_moe.py`/`decode_moe.py` on 16 cards |
| layer suites | 7 cases each, 16 cards | x_hc_out, staged hooks, state | calibrated staged comparator table (`layer_reference.py`) | `prefill_layer.py`/`decode_layer.py` |
| complete entries | T=2 and multi-token / decode full | logits, final HC/pre-mix, per-layer progress | ratio_allclose on logits + staged comparisons | `prefill_fwd.py`/`decode_fwd.py` |

Check shape/dtype, finite values, integer metadata, modified and untouched state,
prefill-to-decode handoff, and all agreed boundary cases. Use representative
original-framework references; measurement-only reports do not count as passes.

## Real-weight generation cases

Calibrate the candidate case pool on the original model before freezing required
cases. Record exact messages, IDs, original outputs and runtime/dtype provenance.
If the baseline fails, apply the authorized original-only selection policy before
candidate execution or resolve the specification; do not freeze an unattainable
answer/budget expectation. Uncalibrated cases cannot silently become hard gates.

Status: the original-model baseline is **not available in this environment**
(the pinned vLLM-Ascend export ships no HF/PyTorch reference runtime and
`transformers` is not installed), so the generation cases are UNcalibrated and
are run as an explicit diagnostic scope: report exact ids/text/termination and
deterministic repeatability, with semantic relevance reviewed by the user. They
are not a hard token-parity gate.

| Case / messages / input tokens | Template and decoding settings | Budget / stop / truncation policy | Original result and evidence | Semantic expectation / review | Candidate command |
| --- | --- | --- | --- | --- | --- |
| "Who are you?" plain ids `[18387,477,440,33]`; chat-template variant `[0,128803,18387,477,440,33,128804,128822]` | chat-template proposal: plain completion (primary) + `deepseek_v41` special-token template (secondary); greedy, temperature 0 | 256 new tokens; stop on EOS or budget; truncation reported, not hidden | UNcalibrated (no original runtime); tokenizer ids cross-checked against the shipped `encoding.py` | coherent, prompt-relevant continuation; user review | `generate.py --prompt ... --max-new-tokens 256` on 16 cards |

Use deterministic greedy decoding and repeat a fixed case. Record token ids,
text, state progression, termination, revisions, topology, exit code, and logs.
Record the compiled prefill/decode entry and runtime artifact identities used.
Accept equivalent wording against the agreed meaning; keyword proxies do not
replace review. Keep factual capability, smoke, numerical gates and token parity
separate. On distributed runs record both ranks and cross-rank execution evidence.
Full-model golden is optional. Serving execution is outside this acceptance.

## Owner closeout

Map every agreed requirement to an actual result on the delivered revision in
`TASKS.md`. Rerun affected gates after changes. Retain failures and label optional
diagnostics separately. An unmet required row keeps the goal incomplete; continue
when implementable. After all required rows pass, save the native goal completion
result. Engineering acceptance, structural acceptance and its negative control
are required rows; module accuracy or generation cannot substitute for them.
Do not end with external parent acceptance pending.

## Status update (2026-11-04 23:22, main-agent)

The A5 single-program realignment unit stopped at its explicit STOP CONDITION:
the inline static-tile `moe_ep16` (with FFN-side HC and `rows` pinned to the
compiled token bound) still trips the pypto T_DYN `ForStmt` typecheck when
composed with a dynamic typed attention chain in one rank program. Controls
pass (chain alone, inline MoE alone, chain + FFN half without the exchange), so
the failure is attributable to the composition. With the already-filed runtime
lifecycle blocker for multi-program staged forwards, the complete-model
forward has no working architecture on the installed stack; complete-entry
device gates, generation and HBM remain open pending a user decision on a
pypto escalation or an alternative structure. Full evidence and the gate table
are in `TASKS.md` (2026-11-04 stop section) and `KNOWN_PYPTO_ISSUES.md`
(refinement 2 of the typecheck entry). No pypto sources were modified and no
commits were made.

## Status update (2026-11-05 00:25, main-agent)

The compiler blocker above is resolved model-side per the user decision
(kernel-loop alignment first). The 16 T_DYN `ForStmt` diagnostics were a
loop-structure issue in our own kernels: `gemm_w8a8` and `o_proj` used
sequential `pl.range` loops over the dynamic token extent. Restructured to
A5's provable form (one `pl.spmd` with a composite dynamic block count,
constant divisors, K reduction in `pl.pipeline`), the exact failing
chain + inline static-tile MoE composition now compiles. Device revalidation
after propagation is in flight (bounded operators, all 12 mode entries and the
16-card layer suites / EP16 drivers). The A5 single-program forward
realignment is unblocked; its continuation contract is in `TASKS.md`.
