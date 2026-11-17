# Independent decode publication / prepared-owner acceptance

The cohort is NOT accepted. The ten return-direction changes restore the six previously failing selection outputs and return nonzero C2 state, but the repaired decode still fails its existing C2-state precision gate. The exact PreparedDispatch tiny proof fails FP32 scalar marshaling before its first chip execution. Both prerequisites for continuation therefore remain incomplete; no chain, text or full weights were run.

## Frozen provenance and resource discipline

Main and both workers were FINAL/PARKED with ended jobs before imports/submissions. The fresh resource check at epoch1794013132 showed0/16 occupied, no running/pending jobs, auto whitelist0–15. Historical16/16 occupancy and cancelled attempts were not reused as current evidence. Each native prerequisite was submitted once; no queue-timeout resubmission, old16-device baseline rerun, reset/admin/permission action or other-user-job action occurred.

Verified45-source manifest1e9496bfdeb177c13f00a2c813a3ecde3242b5bbb622ae44c5a1b5a14377ae43;65-evidence manifestcd3db5949916519398fcc7573a0de6f1ffa6769d03305bcc38b4ef6fabcb150f; worker manifests c6f865d130123c7fcff71203f5c6895713ae448e0661054717724fe309121566 and d7583979e9c61b8c0d60dd164005d9e80eac8d04bff8117eb7ae45751576e933. All source/evidence entries and both worker trees match before/after batches; own frozen_source/ saves45files. Only decode_fwd.py and generate.py changed; other43 source bytes are protected. No model/golden/threshold/acceptance-decision file was edited.

The retained independent old seven-output failure is task_20261106_221305_24806427754 in native_accept_noengram_1793972955. This unit's decode_before.py SHA256270acabd7cdb48d42ed906fcb878a6c8cfb9b38c3e43a7d0ec330c6cfa012f4a exactly matches that review's frozen decode source. It is the causal baseline; another old16-device run was unnecessary.

## Native repaired decode: publication improves, precision still fails

Personal task_20261107_090340_84319715825 allocated all16 devices0–15, normalized ONBOARD, generated a2a3, native Callable images/runtime and actual numerical output verdicts. Native execution103.64s completed without halMemCtl/RunPreparationRefused. Job exits1 at owning comparison, not preparation. The runner's initial PASS explicitly skips golden validation and is not acceptance.

All18 other owning output gates pass, including the newly checked published WINDOW/C1/C2 cache rows. All six previously failing selection outputs now PASS: C1/C2 top-k IDs, pool IDs and exact counts. Source count references remain79/157. C2 state now returns nonzero values close to the reference, unlike the old zero-return baseline.

The sole failure is state_cache_c2a at the unchanged atol=rtol1e-5,0.5%-outlier gate:49,152/98,304 lanes fail(50%); maximum absolute difference0.000860214. Example actual-10.102755, expected-10.102505, difference0.0002499 against tolerance0.000111. The prior baseline's maximum20.0939 zero-return mismatch is gone. This is a residual numerical discrepancy whose math/input/reduction/golden cause is not isolated here. No threshold relaxation, golden change, success inference from nonzero values or claim of original torch_npu parity is made. Native all-output decode acceptance remains OPEN.

The tested fixture is complete40-dense-layer decode1 with a single synthetic24-expert shard reused per rank across layers, noEngram. It is not real40-expert checkpoint execution or strict unconditioned40-layer equivalence. Most owning expectations are staged from actual preceding-layer snapshots; preserve that scope.

## Native exact-driver tiny proof: scalar marshaling defect

Personal task_20261107_090214_8386618278 allocated devices0,1 and reached prepare/static upload. At the first dispatch, generated tiny/.../orchestration/host_orch.py:20 calls TaskArgs.add_scalar(alpha__ssa_v0) with a0d FP32 Torch tensor and raises TypeError: only integer tensors of a single element can be converted to an index. No chip.run.runner_run, output comparison, repeated-state pass or PREPARED_RESIDENT_NATIVE_PASS follows. Native prepared reuse/retained handles/peak proof is NOT accepted from this attempt. Cleanup runs through owner.close and sampler.stop.

Concrete source boundary: generate.py:1563 unconditionally applies torch.as_tensor to scalar arguments; the public prepared runtime forwards scalar arguments as-is to the generated entry(distributed_runner.py around3196). The public TaskArgs.add_scalar binding documents Python int/float/bool or ctypes scalar inputs(task_interface.cpp:2722–2732). Independent CPU task_20261107_090440_8513569047 reproduces the same rejection for FP32 Torch scalar, while Python float, Python int and INT32 Torch scalar are accepted. Thus this is a local scalar-marshaling issue in the exact owner path, not an NPU permission or proven runtime defect. Actual generation currently has an INT32 engram_enabled scalar, so this generic FP32 failure does not by itself prove that its INT32 path fails; it does invalidate the assigned tiny residency/reuse prerequisite. No altered-scalar retry or source fix was performed.

## Independent CPU/source checks

Personal --no-device task_20261107_090121_83480917442 exits0. It reruns original-state/count/source scope, five positive/five missing-published-row controls, native-target decode1 compile, generated directions, fake-public-worker owner controls, handoff/source-target bounds, sampler lifecycle, full-weight guard, continuation clone/API controls and tiny ONBOARD compile.

The exact rank compute body is AST-identical after stripping only the ten changed mutation annotations. All other compute/golden functions remain AST-identical; the only comparator change adds published-row checks at unchanged standard atol.02/rtol1/64/1% thresholds. Original compressor_ref is AST-executed from the pinned rc2 source, verified against preserved archive bytes: asymmetric even/odd state writes match the owning golden exactly, and independent ceil(625/8)=79 and ceil(1251/8)=157 establish expected counts below cap2048. This source/CPU agreement neither explains nor accepts the residual hardware state error.

Generated HOST bindings for window_cache, C1/C2 compressed/index caches, state_cache_c2a, topk_indices/values and pool_block_ids/counts are all INOUT. Before the change the rank formal directions were INPUT despite top-level InOut. Existing producers/math are unchanged. This source/generated evidence plus restored six selection outputs supports the narrow return-boundary repair; no invented missing GPU dependency, new barrier/dummy or defer_wait is attributed as root cause.

PreparedDispatch now uses the supported compiled.prepare context, alloc_stacked_tensor handles for static leading-rank resources and stable shared host IO; fake-public-worker CPU checks establish intended one-upload/cumulative-state/replacement-metadata/cleanup behavior. Scalar native failure prevents upgrading that to native reuse proof. Protocol domains are fresh per dispatch(persistent=False); cache/state still travels shared-host H2D/D2H each call, not persistent device KV storage. Prefill/decode phase owners close sequentially and handles do not cross owner boundaries.

Handoff CPU controls preserve an exact used prefix when trimming unused page capacity; reject used WINDOW/source/target overflow and unknown state-batch shrink. Sampler _stop_event lifecycle and the unconditional host-materialization-overlap guard pass. Both default and explicit allow-full-materialize paths reject before tokenizer/materializer. Real host stack overlap, upload staging, peak HBM, transients and streaming ownership remain unproved; full weights stay HELD.

## Continuation disposition and recovery

The copied chain helper was independently read before launch. It uses the exact PreparedDispatch, one decode artifact/owner, true returned cache hashes, cloned nonstatic initial inputs, two-page prefix0..127 then128/129/130, original-prefix retention, and a stale128-metadata branch expecting129 with actual returned128 state. CPU isolation/prefix-corruption/API checks pass. Its prerequisite guards are not themselves acceptance evidence. The chain/control was NOT LAUNCHED because neither native prerequisite fully passes. No state-handoff/page-parity/stale-published-row numerical success, tiny peak or text claim is made.

Next bounded repairs: handle scalar ABI with supported scalar values and diagnose the unchanged C2-state numerical mismatch against actual stage inputs/original semantics. Freeze and independently recover both prerequisites before the reviewed chain/control. Do not repeatedly rerun the known-failing whole suite or silently relax gates. Existing generalF09 unequal/idle/multi-request, F05/F07/F13, historical missing original-backed hardware controls, full checkpoint/TP/CI/offset incidence and primitive/tie limits remain OPEN/HELD. Root observer records any bounded acceptance decisions.

All personal jobs ended. Results/commands/task statuses, source snapshots, original CPU outcomes and raw failures are retained in this review directory. No full weights, workers, commits, external filing or checkpoint/PyPTO/runtime writes. Reviewer PARK after measured ledger and final manifest.
