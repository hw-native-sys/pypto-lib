# Independent noEngram acceptance — complete decode numerical failure

Do not accept the noEngram generation milestone. The independent native16-device decode1 retry completed hardware execution, then failed seven exported state/selection outputs. The independent bounded40-dense-layer prefill2 counterpart passes. No valid-prefix continuation, text, real weights or memory-fit acceptance follows.

Reviewer /root/reviewer_engram waited for Main's explicit final freeze and all-writers/jobs PARK message before imports/tests. Frozen unit generation_noengram_1793970065:45-source after.json SHA25619ce138488ed96a4230045a032cad426abbbaa333b65cd73af20338a819462de;45-evidence manifest SHA256ba9cb9af67295a5e8b7d98338152ccbd50cf4484e78d864b198e251f329a3e54;28-worker manifest SHA25615d1fed2282411d08c15374b3b7c85ac5423ed3c577512a59c55bafc52e80c85. Verified every entry before/after batches and saved all45 model sources in frozen_source/. Exactly generate.py,weight_loader.py,prefill_fwd.py,decode_fwd.py change from prior freeze; other41 protected sources are unchanged.

## High finding: actual complete decode does not publish required outputs

Trigger: actual noEngram40-layer decode, capacity1, bounded synthetic fixture with one expert layer reused across40 dense layers. Task_20261106_221305_24806427754, auto16 devices0–15, exits1 after95.27s native runtime. Normalized ExecutionMode.ONBOARD, generated metadata exactly a2a3, native Callable images/device runtime present. There is no halMemCtl/RunPreparationRefused in this run. The initial NATIVE_RESULT PASS line has no golden and is not acceptance; the subsequent owning comparison fails.

| Output | Actual numerical mismatch |
| --- | --- |
| state_cache_c2a |49,152/98,304 lanes fail (50%); first actual0 versus expected about-10.1025; max absolute20.0939 |
| pool_block_counts_c2a |48/48 counts fail; actual0 versus expected79 |
| pool_block_counts_c1a |16/80 counts fail; actual0 versus expected157 |
| topk_indices_c2a/c1a |Valid-lane masks differ from independent reference |
| pool_block_ids_c2a/c1a |Selected block pool sets differ |

Layer HC snapshots/final hidden, logits, WINDOW/compressed/index caches and top-k values pass the existing owning staged gates. Passing hidden/logits does not satisfy the exported state/selection contract. Exact errors and per-output verdicts are in native_decode/run.log:2351 onward and results.json. No threshold/golden change, runtime exception or simulated result is substituted for this numerical failure. This establishes a bounded complete-entry publication/state defect, not its isolated kernel/compiler/runtime root cause. Generated rank orchestration declares relevant external state/count tensors and contains inout bindings; that alone does not prove correct publication. A repair must reproduce and fix the exported values before valid-prefix continuation or real generation.

Independent prefill task_20261106_222143_25152446734, auto16 devices0–15, exit0: native runtime93.31s then every owning numerical comparison passes and COMPLETE40_NOENGRAM_NUMERICAL_PASS prefill is emitted. This is bounded prefill2 with40 dense-layer compute iterations and one reused synthetic24-expert shard per rank. Actual40-expert-layer checkpoint-layout ABI was compiled, not populated/executed with forty expert shards. These successful numbers are synthetic staged/per-layer acceptance, not real-checkpoint or strict unconditioned original-model parity.

## Driver/source and CPU review

Personal task_20261106_220426_244831324512 exit0 reruns copied driver/header helpers, permanent --contract-report --tokens2 --engram-mode disabled, both bounded expert-layer1 compiles and both actual expert-layer40 ABI compiles. Compiled metadata confirms the rank/layer/local-expert prefix[16,1,24] or[16,40,24] on routed weights, including prefill SSA-suffixed metadata names. No source changed during these runs.

Source and controls support the scoped F11/F12 repair: separate prefill tile and decode1 concrete TensorSpec/ScalarSpec contracts; ordered ABI/missing-nonweight rejection before large weight allocation; both query profiles bound, unscaled SWA0/1 and independent YaRN equations for compressed layers; complete-entry base/compressed overrides reject. Per-step builder refreshes token/request/position/window/compressed/index/state metadata and query/source RoPE while preserving cache objects. Positions126–129 cover parity/page boundaries on CPU. Handoff only carries six persistent WINDOW/compressed/index/state buffers; query top-k/pool/HC/mix scratch is rebuilt for decode1. Owning expert-axis annotations now match local24/rank layer24/host16-layer24 shapes. This CPU/source proof does not validate continued device state.

Actual runtime scalar engram_enabled follows engram_rotation throughout forward calls; default1 preserves enabled behavior and first path0 gates the compute. Inert arguments remain ordered-ABI-bound. Independent task_20261106_222612_255538531350 exit0 exercises actual materialize_rank with a loader stub: with_engram=False reaches layer_dense for Engram layers1/14 and never calls an Engram table. Source layer_dense condition skips Engram matrices. Results/output scope explicitly noEngram, original_model_parity=False. This is user decision10's temporary scope; it does not close eventual original Engram/hash gates.

The fresh preparation task_20261106_220921_2471096910 passes an explicitONBOARD native cube L2 check with exact output comparison before the decode retry. No admin/reset/permission/runtime action was taken. The prior Main preparation failure remains historical; current review now supplies the genuine numerical failure above.

## Memory/residency finding — real allocation stays held

Both header helpers and the independent CPU payload-denial check read274 safetensors headers without calling the reader's weight-payload method. Expert storage558396604416 bytes; dense/top17196936008; Engram221819890240 omitted for the first path. Header storage is not runtime HBM.

Real40-layer input metadata totals about635GB logical global bytes,39.69GB ideal rank bytes; Main's conservative per-device input upper bound is about124.9GB, before measured allocator/compiler/EP/cache/snapshot/transient overlap. Neither number establishes fit. DEEPSEEK_PER_LAYER_EXPERTS=1 compiles only; expanded synthetic storage must not be presented as forty real expert shards or measured peak.

Actionable source gap: generate.py:566/entry_ordered_specs stamps resident="stacked" and its prose claims upload once/reuse, but compile_entry only uses those specs in compile_only golden.run. Actual generate.py:1489/dispatch calls compiled(*ordered_plain_tensors,config=...), without invoking the harness resident path, preparing retained handles or replacing args with StackedDeviceTensor. The independent meta-tensor dispatch spy confirms all six expert arguments remain ordinary Tensor objects. Runtime one-shot execution only accepts host tensors; supported retained resident allocation belongs to a prepared DistributedWorker owner. Therefore the claimed upload-once rank residency is not implemented/proven by this driver path. Ordinary view transfers may still have narrower regions; this review does not infer that every full global buffer necessarily replicates. Actual transfer bytes/residency/overlap and peakHBM remain unmeasured.

The explicit allow_full_materialize guard currently blocks real weights, which is appropriate. Before enabling it, establish a supported real rank binding/ownership mechanism and measure a conservative peak/transient budget, including overlapping prefill/decode arguments and persistent caches. No full-weight allocation was attempted in this review.

## Continuation and remaining scope

Continuation was NOT launched because decode1 failed its numerical prerequisite. CONTINUATION_REVIEW.md records the d.MUTABLE_NAMES typo, clone/actual-returned-state requirements, staged-reference limits, same-artifact/protocol-window review and a meaningful published-row stale-metadata control. The prepared worker file is not accepted as runnable. No stale-metadata numerical rejection or128prefix->three decode claim is made.

Both owning full-forward comparator sets use actual previous-layer snapshots for most staged expectations; preserve that scope. A future true returned-state chain must show the exact returned buffers feed subsequent calls and separately distinguish conditional step references from strict cumulative equivalence. Same compiled path alone does not prove lifecycle/residency.

This review leaves F09 general unequal/idle/multi-request same-artifact behavior, historical missing original-backed hardware controls, F05/F07/F13, full checkpoint/primitive-tie parity, pool/state/TP/CI/offset incidence and text generation as observer-recorded OPEN/HELD. It does not recalculate the completed historical mode audit or accept Main's next unit. No source/golden/threshold/acceptance-decision changes, workers, commits, external filing, protected-runtime/checkpoint writes or full weights.

One reviewer-only collector mistake is retained: exact-name lookup for generated w1 metadata missed prefill's __ssa_v0 suffix and raised StopIteration. Corrected metadata-name normalization only; no model/test-gate change. Final hashes/results/commands/task statuses and measured review-agent ledger accompany this report. All own jobs ended; PARK after final observer notification.
