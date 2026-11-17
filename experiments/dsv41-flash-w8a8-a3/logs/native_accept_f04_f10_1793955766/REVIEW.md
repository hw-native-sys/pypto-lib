# Independent F04/F10 frozen-cohort acceptance

Disposition: all assigned independent checks passed; no blocking bounded F04/F10 source or numerical defect found. Evidence supports F04 absolute source-row selection and the F10 shared-capacity receive-lane stride under the exercised equal/unequal/idle patterns. Recommend separate bounded acceptance to observer. This reviewer closes no gate, and no F09/full-model/checkpoint equivalence follows.

## Final snapshot actually tested

Main and worker were parked, with no model writers; Main's later Engram preparation remained read-only. Verified final REPORT.md SHA2560599b9df73320ce89b46e47ca708baac05e623abadb29ef5c41167bfee18211d, after.json SHA256bf635491416ffc893715041588683492547d3560ac9d7f1f07572bbfd652f089 and artifacts.json SHA256b353094a54267630798afc545248f4a538955caa2a0ff346b9cc276020427897. All45 current sources,264 Main evidence files and F10 worker manifest match. Twelve independently saved source manifests agree before/after each batch and at final collection. frozen_source/ preserves the45 reviewed sources.

Exactly8 sources changed: utils.py, prefill_c2a_full.py, decode_c2a_full.py, layer_reference.py, prefill_fwd.py, decode_fwd.py, generate.py and moe_exchange.py. Protected accepted MLA/o_proj/rope/indexer/paged_cache/router/loader bytes match. All Main compute-kernel ASTs remain unchanged. Independent final AST check restores only push_lane_base/compact_lane_base capacity names to tokens and proves the entire F10 module otherwise matches its old AST, excluding module-docstring prose. Final F10 SHA256951378f40c2ab7cb4f1573aeda72548bab017ed76c545641620e6e3502b4c780.

## F04 source/CPU and device findings

Authority is pinned official rc2 revision8fde6d4e335fe300e2b949aab0e31659cce0811d. Personally rerun provenance verifies archive SHA25648c797546805b04ffde118c506f8902e37e19bf2cb5d1713097056a4c93e998c and exact dsa_v41.py bytes (SHA2563505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd). Source1340–1349 copies `where(complete,input_positions-1,0)` for C2 source coordinates. The original complete mask also incorporates valid-token and skip-ring metadata; the bounded coordinate oracle supplies valid nonnegative parity-based rows and does not execute or accept the complete metadata lifecycle.

compressed_source_positions correctly selects current absolute C1 positions or emitting odd C2 position-1, zeroing nonemitting rows. Position1253 selects1252, independently of compressed row626, lens, slots or physical block addresses. Both actual source builders, shared layer metadata prefixes (prefill/decode C1/C2), both complete-forward metadata prefixes and actual generator helper pass row-tagged original-expression checks. These are six owning builder families, not merely a test of the helper. Prefix execution stops before irrelevant weight allocation; no full model/checkpoint was run. C1 current positions/profile selection and the SWA compute path remain unchanged. Generator helper correctness does not repair its unrelated ABI/profile/lifecycle issues.

Personally run frozen-old-builder control uses the accepted unchanged source kernel and independently derived original row reference. Both compressed_cache and index_cache are numerical FAILs, with explicit mismatch diagnostics; compile/runtime failures are not control success. Corrected high-position positive passes.

Continuation actually dispatches even1248/1250, captures returned device state_cache/compressed_cache/index_cache via comparator actual.clone(), and passes those actual clones into odd1249/1251 specs. Audited that closure/spec flow directly rather than trusting actual_state_carried alone. Both steps pass using identical `_jit_decode_c2a_full_attention_test_20261106_171025_7cm7ff90`. This is host handoff of actual device values, not persistent device-resident cache lifecycle or F09 full-forward acceptance.

## F10 source/CPU and actual16-device findings

Push and compact now use `(expert*16+source)*capacity` receive bases; allocations/return staging already reserve shared capacity. Active token counts still control valid payload/count processing. Only those two expressions change; routes, weights, public ABI, windows, epochs/signals and expert arithmetic are preserved.

CPU geometry independently enumerates old overlap for active1/capacity2 and mismatched unequal peer addresses; fixed lanes are disjoint. Acceptance additionally PERSONALLY executed actual shared moe_ep16 on all16 automatically allocated devices0–15. The old implementation is numerically rejected on49,992/163,840 routed elements (30.5127%), maxabs about1.41309, at unchanged .02/.02/1% gate. This is this review's schedule-specific observation, distinct from Main's37,776/163,840 result; do not substitute one count for the other.

Repaired equal1, unequal[0,1,2,1]*4 and idle[0]*8+[1]*8 pass. A second runner under the same independent16-device allocation executes all3 patterns sequentially in one process/one `_jit_dynamic_moe_20261106_170742_qhb235e1` compiled artifact. Its comparator first requires exact inactive-sentinel equality, then delegates the unchanged numerical band. All3 pass, including idle source ranks that still receive expert work. Thus six positive executions repeat three designs. The bounded harness calls the actual shared primitive; it is not a substitute exchange implementation. Its capacity2/runtime slices do not establish full-forward active-row propagation, general persistent buffer lifecycle or original-checkpoint equivalence.

## Independent execution and attribution

All checks here were personally submitted/executed by observer-owned reviewer /root/reviewer_native. Main/worker self-tests are not counted. Exact submissions are commands.sh and run.sh; raw logs/results/allocations/statuses are retained. CPU/compile uses --no-device; NPU uses --device auto, with --device-num16 for F10. Selected environment and pinned PTOAS are explicit. Host records PyPTO38a8aa6/simpler31f9e72/PTOAS0.72/ISA f83d4b7. No simulator or original checkpoint weights.

| Task | Independent result |
| --- | --- |
| task_20261106_170411_38011715330 | Initial frozen-source/evidence verification, exit0 |
| task_20261106_170510_380426026292 | All16 devices0–15, epochs1793955914–1793956109: old numerical rejection,3 separate positives and3 same-artifact replay positives; exit0 |
| task_20261106_170511_380473410668 | Original-coordinate/actual-builder and F10 geometry CPU checks, epochs1793955915–1793955923, exit0 |
| task_20261106_170511_380473527453 | Both forward CPU compiles, wrapper5915–5978, decode5919–5950/prefill5950–5977, exit0 |
| task_20261106_170601_381365323116 | Auto device0 after F10 released, epochs1793956114–1793956274:1 wrong-coordinate numerical control and3 positives (high-position+2 actual-state steps), exit0 |
| task_20261106_170601_381365512034 | Auto device1, epochs1793956113–1793956474:17 executables/37 numerical positives, exit0 |
| task_20261106_171729_41388227117 | Final source/264-evidence/worker/AST/results/binding collector, exit0 |

The37 downstream positives include all12 phase entries plus accepted MLA,o_proj,router,compressor,indexer defaults. Adding3 coordinate positives gives40 independent Main-scope executions and1 negative; F10 six positives/one old numerical rejection remain separately attributed. All expected negatives are real output numerical rejections. No unexpected numerical/compile/runtime failure occurred, and no foreground timeout occurred in this cohort; all actual task statuses completed0.

results.json records per-output verdicts, errors/counts, task-row epochs, allocations and work directories. generated.json hashes fresh decode/prefill complete-forward orchestration and compressed-cos/sin input bindings. F04 is a host table-selection fix; correct builder values plus compiled ports and device cases establish its exercised integration. Forward compile success/bindings do not establish full-model runtime publication.

## Remaining limits and recovery

Preserve original torch_npu metadata/reduction/quantization primitive and low-amplitude/tie parity limits, including earlier F01 exploratory random exact-equality failure. No original-checkpoint/full-model parity, general F09 lifecycle or generation is claimed. Engram F05/F06,2048 candidate capacity,F09,generator F11–F13,conditional checkpoint offset incidence,ordinary16-device layer suites,standalone TP,CI and composed full-model pool/state gates remain OPEN/HELD. The16-device execution here is the bounded F10 primitive, not a full layer/model gate.

No model/golden/threshold/Main-evidence/TASKS/checkpoint/PyPTO write, worker, commit or external filing. All own jobs ended and allocations released. Measured review-agent intervals append under the shared lock; parallel job durations are not summed. Reviewer PARKED. Observer records bounded F04/F10 disposition and assigns the next frozen scope; reread current protocol/directives and verify ownership/source hashes on resumption.
