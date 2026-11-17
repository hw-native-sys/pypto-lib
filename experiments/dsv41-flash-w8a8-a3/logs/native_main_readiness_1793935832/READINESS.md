# Native Main readiness: F02 output inverse RoPE

Status: MIGRATION READ-ONLY; awaiting explicit observer GO. No model writes,
worker launches or queue submissions occurred during preparation. F01 STOP in
TASKS at epoch 1793935814 is a claim for the observer to verify, not ownership
transfer or independent acceptance. The observer retains acceptance authority.
COORDINATION_STATE.md was absent when read; Main did not create this
observer-owned file. Historical CLI-only HANDOFF text is superseded by
COORDINATION.md and MAIN_AGENT_BRIEF.md.

## Assigned next coherent unit

Repair F02 only: inverse query-position RoPE of the BF16 attended heads before
Wo_a/Wo_b. Preserve the accepted F01 query magnitude and BF16-before-RoPE
boundary, reindex pool consumption, width/tail/padding, W8A8 policy,
DP4/TP4/EP16 topology, paged layouts, dynamic axes and existing numerical gates.
F03 source-compressed-KV rotation and F04 absolute C2 source coordinates remain
open. No claim of complete original attention semantics or full-model parity
can follow from F02 alone. Generation remains HELD.

Readiness inputs: current goal decisions 1–9, latest TASKS directives/checkpoint,
TIME_LOG tail, ENVIRONMENT rev3, HANDOFF tail, MAIN_AGENT_BRIEF, COORDINATION,
repository CLAUDE/model/problem rules, run-model-cases/task-submit skill,
model-family documentation, F01 REPORT and frozen logic REVIEW F01–F04.
The A3 implementation has no dedicated canonical model documentation page;
family page describes the distinct A5 performance target and is not authority
for changing this A3 topology or checkpoint quantization.

## Proposed exclusive file map after GO

Main exclusively owns the following model sources:

- rope.py: extend the existing BF16 trailing-pair implementation with a
  compile-time inverse direction while retaining forward behavior for current
  callers. Avoid a duplicate rotation implementation.
- o_proj.py: own the inverse rotation at the output-projection boundary before
  grouped Wo_a/Wo_b, and its owning golden and executable defaults. Add query
  cos/sin arguments to this internal operator; allocate distinct BF16 scratch
  so attention output aliases cannot corrupt concurrent consumers. Public
  phase/layer/forward ABIs already supply query tables and need no new ports.
- prefill_swa.py, prefill_c1a_full.py, prefill_c1a_reindex.py,
  prefill_c1a_reuse.py, prefill_c2a_full.py, prefill_c2a_reuse.py.
- decode_swa.py, decode_c1a_full.py, decode_c1a_reindex.py,
  decode_c2a_full.py, decode_c2a_reuse.py.
- layer_reference.py: pass query tables into both _partial_rows call sites.

These are 14 source files, with all kernel and golden callers adapted together.
decode_c1a_reuse.py delegates to the shared decode reuse compute; inspect its
propagation but no independent writer or unnecessary edit. Forward and layer
implementations compose these phase operators; inspect their generated bindings
and compile without changing them unless an actual consumer requires it.
mla_proj.py and indexer.py remain protected F01/accepted snapshots.

Main owns F02 integration artifacts, source snapshots/manifests, runner
integration, and locked TASKS/TIME_LOG appends. No other writer owns any of
these model files. Record the final unit artifact path and exact ownership in
TASKS under .coordination.lock BEFORE worker dispatch.

At most one native worker while reviewer is active. Worker exclusive scope:
only <F02-unit-artifacts>/independent_oracle/ (scripts, report, measured work
record IDs). No model/golden/threshold/ledger writes. Worker grounds the oracle
against pinned source, reports exact bytes/reference hashes, develops controls
and a reproducible CPU oracle runner, then parks. Main reviews the independence,
integrates appropriate permanent default semantic coverage, runs validation,
and imports each worker work record once with its source ID. Worker never
spawns agents. A worker is useful for a separately derived semantic oracle,
not merely editing disjoint call sites according to Main's implementation.

## Independent semantic validation

Authority is official rc2 revision 8fde6d4e335fe300e2b949aab0e31659cce0811d,
preserved under logs/logic_review_1793929657/upstream_rc2; archive SHA256
48c797546805b04ffde118c506f8902e37e19bf2cb5d1713097056a4c93e998c.
Upstream dsa_v41.py forward:711–716 invokes _apply_rotary(inverse=True),
then _project_output; _apply_rotary:291 passes direction and trailing head
range to DeviceOperator. Inspect the original operator's CPU golden and
_forward_o_proj implementation to establish pair order and cast boundaries.
AST-execute original methods where possible; identify CPU stand-ins honestly.
Never claim original torch_npu hardware execution from those stand-ins.

Use nonidentity per-token and per-pair angles, signed rotary/nonrotary lanes,
distinct heads/groups, asymmetric sparse Wo_a/Wo_b and nonzero outputs. A
closed-form paired-complex inverse with explicit BF16 rounding and sparse
signed projection oracle must not call candidate rope/output/attention
helpers. Compare candidate golden with original-source and analytic oracle;
then validate the actual compiled shared operator against that independent
oracle at T1, T7 and a tile-boundary tail such as T17. Reuse one compiled
dynamic artifact across token counts and verify unchanged nonrotary lanes.
Permanent o_proj defaults should preserve this meaningful semantic coverage.

Numerical controls: missing inverse (frozen original boundary) and forward
instead of inverse must fail output validation with the same asymmetric
weights and gates. Record failing lane counts/max error. Control success
requires numerical rejection; compilation/runtime failure is not success.
Preserve original failure artifacts and propagate unexpected exceptions.
Inspect generated inverse task dependency/output buffer binding into Wo_a.

Run all twelve actual phase executable defaults, including delegated decode
C1 reuse, against corrected owning goldens, with fresh build/output roots.
These establish propagation, not F03/F04 parity. Rerun F01 independent MLA
defaults and relevant indexer defaults to protect existing accepted behavior.
Compile both complete forwards and inspect shared o_proj/inverse bindings;
compile success alone cannot establish runtime publication/full-model parity.
A bounded representative layer run may strengthen integration evidence only
if needed and available; never close unrun full-layer/16-device gates.

All CPU/compile/lint checks use task-submit --no-device. All NPU checks use
task-submit --device auto with TASK_DEVICE, never simulator. Source
activate-pypto.sh and explicitly prepend pinned PTOAS_ROOT/bin. Verify rev3
PyPTO 38a8aa6 / simpler31f9e72 / PTOAS0.72 / ISA f83d4b7 and build-input pins
before execution. No duplicate observer acceptance jobs during migration.
No network fetching is needed while preserved official source is sufficient.

## Milestone handoff

At implemented/device-validated F02 stop all unit writers and own jobs. Record
source before/after hashes, exact reference bytes, command/task/allocation/exit
matrix, per-output results, negative controls, generated bindings and current
recovery state in REPORT/TASKS; send the same evidence to observer by native
message. Freeze that manifest for observer personal acceptance. Do not claim
observer acceptance. F03/F04 and all other unclosed gates remain open.
