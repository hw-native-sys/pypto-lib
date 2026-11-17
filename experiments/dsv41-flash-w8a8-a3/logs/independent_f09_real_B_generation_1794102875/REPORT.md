# SourceB real16 — independent actual postterminal audit

**Strict overall NOT_ACCEPTED: 4638 checks passed, 5 unmet.** Product CLI completed exit0; native queue and outer operation remain failed because process-memory telemetry was incomplete. The independent CPU auditor completed exit0. No status was relaxed and no NPU retry occurred.

## Exact case and one audit

Case `native_f09_real_B_generation_1794098661`, SAME native task **task_20261108_090706_320779619842**, Root ID **observer-20261108-sourceB-real16-shared-w8a8**, actual ordered devices0..15. Native terminal status is `completed (exit=1)`. External POST task **task_20261108_094741_97834220909** completed exit0; POST SHA **f204e88d7a38df18f5def6f18ef65b10a1eec885719a98365ae7161772a904cb**.

This NEW same-depth execution unit byteforked immutable preparation18, manifest **08f70c924d3fbabaf28e4b848ba9b2c75292352061976918d1b830d4ac594d39**. Inherited seals are input_*. Audit/helper criteria hashes remained exactly unchanged; only real handoff/outer command/output paths were adapted. Reviewer filled the handoff after outside-queue native+POST terminal captures under Root's NEW actual-audit GO.

ONE CPU audit **task_20261108_095507_99694925565**, no-device/max600/client900, completed **exit0**. `audit_command.txt` preserves the exact invocation using the same .venv-local Python directly, avoiding activation/runtime imports. Only the frozen pure-host standard-library tokenizer utility was used; no forward/model-compute/runtime import, NPU, header/payload read, golden, compile or old-case replay. No second audit/test was run.

## Unmet criteria and first cause

The five unmet checks are:

- native queue COMPLETE0: actual exit1.
- combined CLI/outer success condition: CLI0 and no launch/wait/monitor/post error remain true, but outer status is FAIL.
- strict telemetry COMPLETE: actual PARTIAL_OR_UNKNOWN, complete measurement false.
- every process sample known: some observations have unknown memory/inventory facts.
- whole PSS equals known-process sum: whole totals are null when coverage is incomplete.

There are **218 noncritical process/thread gaps**, critical telemetry errors empty, monitor errors empty, postcheck error null. First gap: epoch **1794100049.1395829**, PID **3209481**, birth **109996765**, read-smaps-rollup at `/proc/3209481/smaps_rollup`, **errno3 ProcessLookupError**, critical=false. Whole own-tree PSS peak is **null**; maximum known partial sum **12962226176 bytes** is not a complete peak. These gaps remain a strict case failure even though CLI0 and bounded model/transport checks succeeded. Old M02 real41/audit44 remain their own failed cases and were not relabeled.

## Verified bounded component facts

- Current SourceB45 **30b7106a5427c283917a7eeeec65aab598ab98bf5a70f1667285169c9c6d8eca**, gen5de1656a...e34c, A8153 math baseline, new plan **cd8b17741688eeb9ef39d8c0925dbb74a8f9a1cfcc0052db2e0a8c1aacb48cf7**, fresh704eb/767 signatures and Main helper16/code3 bindings passed unchanged. Historical HC ancestry stays separate from exact current F09 evidence; accepted171 remains its bounded synthetic scope, not an original realE40 reference.
- PREQUEUE, allocated, in-job POST and external POST exact source/tokenizer/library/header/HC+F09/signature/resource identities passed. Actual after-allocation Root scope and normal CLI matched. No tensor payload was reread by the auditor.
- One canonical prefill upload: **633997065216 H2D bytes**, **912 bases/57 arguments**, **164192 source reads / 745846365184 source bytes**, exact source map and all912 interval endpoints. Decode: **zero new static reads/bytes/upload/allocation/copy calls**. Durable borrowed coverage matched **912 bindings =112 C1 full-byte singleton views +800 same-shape aliases**, proper order/shapes/dtypes/bytes/zero offsets/equal positive addresses. These are pinned-owner/emitted alias facts, not independent device allocation-memory reads or full weight D2H parity.
- **One owner /16 native chip workers**, not two worker sets for two programs. Exactly one final-close/free pair at actual last phase decode, with source-bound hard public counters/reserves/host-disk-shm floors and no hidden cleanup error.
- Actual runtime artifacts are `build_output/_jit_prefill_fwd_20261108_090726_hzc1sumq` and `build_output/_jit_decode_fwd_20261108_090752_he0j5f9q`. Both actual100/96/E40/a2a3/static interfaces passed. Fresh SourceB HC witnesses matched actual source: each phase **PTO4 RINT-down/4 FP32-expand and C++2/2**, separately counted. Both actual metadata sets preserve six persistent cache InOut formals. CPU signature directories were not presented as executed artifacts.
- **15 actual decode calls**, finite all4DP selections, TP winner consistency, true prior generated-token inputs/histories, positions3..17/end18 and returned state diagnostics passed. Frozen checks validate all six persistent-cache fingerprint fields per step. Read-only JSON observation additionally confirms eight prefixes in every step, including x_hc/pre_mix.

## Fingerprint scope

Source generate.py223-225 converts each host tensor to **contiguous FLOAT32**, hashes those value bytes with SHA1, then keeps **16 hex characters**. These are diagnostic value-summary fingerprints, not hashes of original cache-storage bytes, saved raw tensors, or independent numerical/original-history parity. No missing raw payload was manufactured. `state_diagnostic_scope.json` preserves exact available fields and this limitation.

Final step15/position17 includes window `b65211cc901b9fcd`, C2 compressed `dd74c65d1afb16c2`, C2 state `34eda82525798c6a`, x_hc `ae85eea8c8eeb48b`. Eight available summaries do not establish complete original cache history or generalized F09 correctness.

## Actual output and quality

Generated IDs are exactly:

`[1014,1613,3305,1999,344,270,5281,294,270,17132,582,32568,4465,260,1014,57008]`

Raw suffix:

```text
 time...". What is the meaning of the phrase "once upon a time"?
```

Termination is max_new_tokens, 16 outputs/15 actual decode calls/end18. IDs/text match the prior M02 observation. This is a bounded observed equality, not causal proof, automatic coherent-text acceptance or full-original parity. Quality and complete peak-memory acceptance remain unassessed; scope is noEngram/noMTP, realE40/W8A8/TP4/DP4/EP16.

## Cleanup, identity reconciliation and closure

Frozen audit sources record **1159 PID/birth identities**, all absent; all16 ready workers have recorded births. The owner's1160-record set contains one additional identity, **PID3208241/birth109994965**, exactly matching saved observed_outer_parent.stat. Read-only reconciliation independently confirms all1160 matching recorded owner births absent now. These counts are bounded recorded evidence, not complete process history. Criteria were not changed to merge counts.

Raw native log confirms all16 lock releases. Outside-queue final occupancy is0/16 with no pending/running jobs. The audit process's own birth is separately absent after terminal exit0. Preparation18, source/helper/code criteria and all protected owner terminal inputs remained unchanged. `closure.json` and `identity_reconciliation.json` preserve exact receipts; no cleanup test or device run was used.

Protected generation SHA **9e30c0d0369af44d6ff3fbd764b6206854503d37b472b222e7d93532f6fac0ff**, outer SHA **7838a6389bb25ebb19fe4a2e0180dd0ddba8e92f476c27fd278e6f3fe190238f**, terminal facts **8b28275e7748a8f04f89399f8ff84705cfb45ba52af568848257ee103c934126**, known-owner identities **924db4ec2c9c047e16d99a6c9a5f20ff9d108cd21fac1ae8442a57c13b702063**, and raw native.log **3d4571f0ec5a91c69527216e7c8652abbfe6f5be3dbd3ed0793e5a2661c1fa75** stayed stable. `audit_results.json` lists all4638 passes/5 unmet checks and consumed source/evidence hashes; scoped_summary separates component facts from strict failure.

FINAL/PARK: overall NOT_ACCEPTED under unchanged telemetry policy. Shared writes0; Root imports once. No remaining audit/job, new native submission, model/runtime/helper change or policy relaxation. Root may accept only bounded component facts separately after reviewing this seal.
