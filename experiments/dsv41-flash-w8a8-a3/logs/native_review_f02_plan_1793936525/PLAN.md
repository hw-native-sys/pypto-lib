# Independent F02 validation plan

Status: PREPARED / PARKED. Preparation only; no CPU/NPU validation jobs submitted and no model writes. Main currently owns implementation and self-tests. After Main parks, observer will deliver the precise frozen scope and assign independent automatic-allocation acceptance to this reviewer. Observer coordinates/reviews evidence/records decisions; the old observer-personally-run requirement is superseded by the latest user instruction. F01 is already accepted; no F01 acceptance rerun is needed during this preparation.

## Independent reference and test equation

Authority remains decisions1–9 and official vllm-ascend v0.30.0rc2 revision8fde6d4e335fe300e2b949aab0e31659cce0811d. Primary local reference is the preserved upstream_rc2 archive from logs/logic_review_1793929657. Read directly:

- dsa_v41.py:291 and :711–716: inverse trailing-pair rotary on attended values at query-position cos/sin, before output projection; query tables, not compressed source coordinates.
- dsa_v41.py:432–438: pad inactive rows where needed, then call _forward_o_proj.
- dsa_v1.py:1731–1757 and :1759–1840: grouped BF16 Wo_a batchmatmul then flattened Wo_b; preserve intermediate activation dtype. Rank-local F02 does not validate optional output-projection all-to-all/reduce-scatter branches.
- tests/e2e/nightly/single_node/ops/singlecard_ops/test_inplace_partial_rotary_mul.py: original FP32 CPU rotary golden, with negate_sin=True for inverse, then activation-dtype narrowing.
- WEIGHT_MANIFEST.md:82–83: actual Wo_a/Wo_b weights are BF16 under this W8A8 checkpoint; no new output-projection INT8 policy.

Expected boundary is BF16 attended -> FP32 inverse rotary arithmetic -> BF16 rotated attended -> grouped Wo_a -> BF16 intermediate -> Wo_b -> BF16 partial output. Nonrotary lanes must remain unchanged. Verify this against final implemented code and actual original-source execution, rather than letting candidate kernel/golden agreement establish it.

## Frozen-handoff review

Require Main STOP, ended self-test jobs and no active source writer. Obtain Main final manifest/report and actual unit path. Save an independently measured current45-file manifest and verify byte equality with Main's final manifest before executing anything. Reject unexplained differences. Confirm only the authorized14 files changed; accepted mla_proj.py and indexer.py must retain F01/accepted hashes. Recheck the full45-file manifest after every acceptance batch and at closeout. If a writer resumes, stop acceptance and report the incomplete checkpoint.

Review shared inverse implementation, BF16 scratch ownership/alias safety, direction flag, exactly-once inverse placement, updated internal o_proj interface, all twelve phase kernel/golden paths including delegated decode C1 reuse, and both layer_reference._partial_rows call sites. Verify query table/token-row mappings, dynamic tails and public ABIs. Main-worker independent_oracle artifacts are useful supplied evidence but remain Main-owned work, not reviewer execution.

Reverify reference archive hash48c797546805b04ffde118c506f8902e37e19bf2cb5d1713097056a4c93e998c and exact tar equality for every upstream source actually executed, including newly relevant dsa_v1.py. Save file/AST hashes, toolchain/pin identities and exact final evidence-script hashes. Inspect generated inverse task -> BF16 scratch -> Wo_a input binding and dependency; verify no direct pre-inverse attended bypass. Compile evidence is not runtime proof.

## Reviewer-owned execution after handoff

Use a NEW sibling artifact directory logs/native_accept_f02_<measured_epoch>, never Main's output directory. Set PYPTO_PROG_BUILD_DIR and every helper output override to reviewer-owned paths. Inspect helper CLI/output behavior before use; write any acceptance adapters only under reviewer artifacts. Archive exact executed commands in commands.sh and task/result tables. Source activate-pypto.sh and explicitly prepend pinned PTOAS_ROOT/bin in queued jobs. CPU/compile jobs use task-submit --no-device; NPU jobs use task-submit --device auto and the allocated TASK_DEVICE. No physical-device assumptions, simulator, workers or threshold changes.

1. CPU: personally rerun pinned-original/analytic semantic checks, independently inspect or derive adjacent-pair complex inverse equations and sparse signed output projections. Use nonidentity per-token/per-pair phases, distinct heads/groups, asymmetric sparse Wo_a/Wo_b and nonzero outputs. Candidate golden, original-source CPU oracle and closed form must agree on deterministic fixtures. Keep original source execution and CPU substitutions clearly identified.
2. NPU core: personally run compiled shared o_proj on deterministic T1/T7/T17 including a tile-boundary tail, plus ordinary random coverage. Require exact deterministic checks where analytically/device defensible, retaining existing random gates. Verify one dynamic artifact reuse from actual build records. Check preserved attended input if aliasing is relevant to its contract.
3. NPU negative controls: run frozen missing-inverse and forward-instead-of-inverse artifacts against the same independent oracle and asymmetric fixtures. Each must fail numerically on expected output lanes; compile/runtime errors, zero outputs or runner exit0 alone do not establish a meaningful rejection. Record mismatch counts/max errors and unaffected outputs when available. Review an omitted BF16-rounding CPU control to establish sensitivity of the claimed cast boundary; do not invent unsupported device-bit parity claims.
4. NPU propagation: personally rerun all twelve phase executable defaults in fresh reviewer outputs, pending observer's precise frozen assignment. These check F02 propagation across actual consumers and delegated reuse, not complete F03/F04 parity. Run permanent MLA/indexer regression defaults if included in that assignment to protect the shared forward rotary change and accepted fixes; attribute these as regressions, not a repeat F01 acceptance decision.
5. CPU integration: personally compile both complete forwards if assigned; inspect actual generated inverse/output-projection bindings. A16-device layer run requires a concrete assigned integration question and appropriate automatic allocation; otherwise retain it as unrun/open rather than treating standalone phase checks as equivalent.

Commands will be finalized against the delivered helper interfaces, copied into reviewer commands.sh and recorded exactly when executed. No future task ID/device/exit is claimed now. For each actual queue task record submission, task ID, allocated device list, measured epochs, exit, build paths, source/script hashes and per-output gates. Preserve every unexpected failure and diagnostics before any justified rerun. Main self-test results and reviewer-run results get separate rows and summaries.

## Limits, outcome and recovery

Preserve F01 review's exploratory random exact-equality failure: CPU modeled dynamic quant arithmetic differs from candidate quantization at some half ties; existing random gate success is not exact torch_npu quantization parity. Original torch_npu rotary/BMM hardware parity and actual checkpoint generation are not proved by mathematical CPU stand-ins. State FP32 accumulation/BF16 narrowing assumptions and numerical-gate limits. F03/F04, distributed semantics, full-model publication/state/dynamic/TP/CI and generation remain open unless separately assigned and established.

Deliver REVIEW.md with ranked actionable defects or a bounded no-blocker disposition, final manifest actually tested, exact independent results, Main-evidence separation, generated-source hashes, outstanding coverage and resumption. Observer then records acceptance; reviewer does not close gates. Append real measured review-agent intervals under .coordination.lock. Park now; on follow-up read updated protocol/TASKS and frozen Main handoff before creating acceptance runners or queue jobs.
