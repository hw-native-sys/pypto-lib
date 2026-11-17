# Independent empty-operation acceptance — noEngram, CPU/source only

Production source45 remains762dc4f7a64fe38ee7868bf5891411798be350ef6781ef35e4c58f7dfd2d89c7. This NEW review owns only independent_empty_operation_1794039524. No NPU job, actual prepare, real tensor allocation, checkpoint payload or forward ran. Old independent35/source/old31/postguard48 remain sealed. CPU-only outcomes are not native calibration or fit evidence.

## Disposition

The **calibration-only** A/B/C implementation is ready for a separately assigned native calibration, subject to root's resource/authorization decision. This review issues no NPU authorization. Actual calibration, initialized physical touch/upload/full40 forward peaks, real-weight fit and text remain OPEN; the production full-weight guard stays HELD.

The later separate large probe has a limited additional observation: its final fresh **prewarm** baseline is checked against calibrationA and aggregate prewarm, but its final after-free/before-close query is only recorded. Our independent fake changes that later query by66MiB against65MiB tolerance and the probe returns CPU_FAKE success. CalibrationC checks both initial and after-free snapshots and rejects this sort of regression. This does not affect the next calibrate-only route. Before a later large-probe GO, either add that final after-free comparison in a NEW separately reviewed source cohort or explicitly narrow the accepted final-baseline claim to the prewarm samples. Do not change frozen source or conflate this with actual device failure.

## Independently verified pins and artifact

- Main adapter empty_fit_operation_1794037635/empty_operation.py SHA8d3ad68504cef4f0527594a37daa9dd8ef90041c0b56b8e199af30c99a5b83aa.
- Main478-artifact manifest f55ce7d59678f0977d466e673e3c0ad0429b5c26cdddf6253bf059cb9630c556; all entries hashed independently. Sources manifest16a6db8a0bfd8a3fc26fec0a3f95ae051423342103dc2c2f9bfaf959813280a8.
- artifact_receipt.json SHAc2cc6c2b1db13a67e202b160e4abd4c1434a9fe9f6825be9e1e629f05234fe2f; artifact path signature_build/_jit_prefill_fwd_20261107_155011_6nkuzpv3; distributed_meta.json SHA3ef340dbe30e192bddd20ab2ec26bbe33b86453ffb0c6ce37d1512b847a50715.
- All283 generated files/actual57 static ABI shapes and dtypes verified; real E40 [world16,expert_layers40,localexperts24,...] w1/w2/w3,39,624,816,576B/rank, largest11,324,620,800B. Signature artifact size9,992,067B; no tensor data is implied. Existing compile raw logs/pins support the signature compilation; this reviewer did not redundantly recompile it.
- Accepted postguard core SHA1542c1848227a0326655dc809ede1c71bca1e3d71d67f77a11e604741b924865 is imported unchanged. Installed a2a3 TMR PROGRAM shared-bank contract/library24434f8879030ef1b1f31e5fc2ecce683e392d4f1baeb5bcee852fb183976e5c was queried again read-only on CPU through the actual native-factory setup before its mocked loader. Identity168 is not a bank count.

## Source/API and operation review

`main` defaults to refusal before native owner construction. Execution rechecks source45, pinned source artifact57 headers and every generated artifact file, exact16 distinct queue device IDs, operation/config/policy/source/artifact-receipt/library binding and time-limited root scope. Calibrate and large-probe authorization scopes differ. Native loader uses actual normalized `config.arch` (BackendType.Ascend910 enum), explicit ExecutionMode.ONBOARD, config.distributed_config and the pinned generated a2a3 artifact. Every prepare uses the SAME public normalized config with ring[0,0,4GiB,0] and persistent=False. Dtype mapping uses actual torch.int8/bfloat16/float32. Public queries/free/close match supported signatures. No dispatch/upload operation exists in the adapter; alloc_tensor uses init=None.

Calibration stages are Aprewarm/normalclose ->Bfresh prewarm, exactly1MiB I8 EMPTY/rank, allfree/query/normalclose ->Cfresh prewarm/query/normalclose. Largest/57-static allocation is absent from this mode. Fresh owner identity is checked, every handle is registered before core post-query validation, normal owner exit is required. Error paths save raw owner records, preserve primary query/schema/cleanup errors, and use null diagnostics for unavailable measurements. Process-tree snapshots include PID/startticks/cmdline; post-job independent PID/resource audit remains mandatory and is not supplied by a running parent's self-observation.

Root-approved tolerance semantics are implemented exactly: for each rank, max of nonnegative A->ALL B interval free drop or committed rise, INCLUDING tiny logical bytes, plus64MiB; >256MiB is STOP after normalB close, with no clamp/noC/no success receipt. C initial and after-free public states use that same tolerance againstA. This is bounded observation plus policy, not allocator-vs-ambient isolation, a leak proof or a universal peak guarantee.

Receipt validation separates CPU_FAKE_EMPTY_OPERATION and ACTUAL_DEVICE_EMPTY_OPERATION, recomputes the16 tolerances from raw perrank observations, checks3 distinct owner epochs/normalclose,0-16-0 tiny-handle/free/schema counts, original policy and null unknown peaks, raw A/C consistency, C bounds, exact binding and <=3600s age. Later large-probe mode additionally needs a fresh separate root authorization bound to the actual calibration receipt SHA. Receipt labels and owner IDs by themselves are not native execution proof: root must independently bind the actual queued job, immutable receipt bytes and actual lifecycle/PID audit before approval.

Reserve is17,954,081,472B/rank: N128 conservative IO7,609,876,736 +nominalcomm680,528,320 +fixed1GiB PLANNING allowance +untouched8GiB safety. This is checked with direct publicfree before/after allocation. Committed/ring/prewarm are not subtracted again. Fixed1GiB is explicitly conservative over-reservation, not measured isolated overhead or a cumulative cap. Full40 firsttouch/upload/forward peaks remain null; old MEASURED peak arrays and old core.authorize/main are never used. N128/two-page IO is conservative planning, not measured N3 sharding or current native usage.

## Independent execution evidence

All executed Python commands used `task-submit --no-device` and activate-pypto.sh. No Main test suite was reused as independent PASS.

1. task_20261107_162051_24467694384 COMPLETE0 ran own review.py. Independent source/478+old31/postguard48/old35 hash verification, actual57 metadata check, default refusal, positive calibration0-16-0 and derived65MiB tolerance PASS. >256MiB stops afterB; C regression, missingrank/stale/wrongoperation/fake->actual/unknownpeak0 all reject. Early rank7 query and malformed public schema preserve original error/null diagnostics and close/free. Separate fake largeprobe16-912-0 and final fresh prewarm regression controls pass. Late final after-free drift observation is preserved, not represented as accepted broad close proof.
2. task_20261107_162209_245073721350 exit1 was a reviewer-spy assumption error after valid mocked A/B/C: the test compared loader backend_type to string'a2a3', whereas the API expects normalized config.arch enum. Raw failure preserved in public_initial_failure.log. Corrected only the reviewer assertion, using fresh fake output and exact config.arch equality; no Main source change.
3. task_20261107_162244_245263731619 COMPLETE0 ran own public_factory_review.py. Actual production config/normalizer, actual Torch dtype map and actual installed contract query; from_dir+prepare MOCKED. SAME normalized ONBOARD4GiB config on3 prepare calls, onlyB16 emptyI8 handles, allfree/normalclose. Fake kind, crossscope, world32, changedsource and expired authorization reject. Physical device prepare count0. public_results.json holds these checks and resource snapshot.

Exact commands:

```bash
task-submit --no-device --max-time 240 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && python experiments/dsv41-flash-w8a8-a3/logs/independent_empty_operation_1794039524/review.py'
task-submit --no-device --max-time 240 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && python experiments/dsv41-flash-w8a8-a3/logs/independent_empty_operation_1794039524/public_factory_review.py'
```

Both scripts intentionally require fresh outputs; do not rerun into this sealed directory. A new assignment needs new outputs/collector.

## Next CALIBRATE-ONLY command — requires a separate root assignment

Candidate queued command (NO GO issued here):

```bash
task-submit --device auto --device-num 16 --max-time 1800 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && python experiments/dsv41-flash-w8a8-a3/logs/empty_fit_operation_1794037635/empty_operation.py --mode calibrate --execute --authorization <NEW_ROOT_ASSIGNED_CALIBRATE_AUTH> --artifact-receipt experiments/dsv41-flash-w8a8-a3/logs/empty_fit_operation_1794037635/artifact_receipt.json --output <NEW_REVIEWER_OWNED_CALIBRATION_DIR>'
```

The two placeholders are necessarily pending root assignment; neither currently exists as a valid actual receipt. After automatic allocation and BEFORE prepare, the assigned validator must create the auth using `binding(artifact_receipt, [int(x) for x in TASK_DEVICE.split(',')])`; do not assume physical0..15 from signature compile. Auth schema empty-operation-authorization/1, approvedtrue only under actual rootGO, scope empty-allocation-calibrate, execution_kind ACTUAL_DEVICE_EMPTY_OPERATION, future integer expiry<=3600s, exact current binding. Root may authorize a task-local creation step whose sole scope is this calibration. Do not manufacture actual auth by relabeling CPU fake data. **No --mode probe, large statics, upload or forward is included.**

Before submission and again inside the allocated job: rehash current45/adapter/478/artifactreceipt/283generated/core/old31/postguard48; verify actual16 automatic IDs and selected installed24434f runtime; assert no duplicate own queued job or live own NPU. Preserve rootassignment and exact command. Suggested admission POLICY for this small no-payload route is MemAvailable>=64GiB, output diskfree>=2GiB, /dev/shm free>=16GiB, with monitoring and firsterror STOP. These conservative values are a reviewable root policy proposal, NOT measured preparation peak or a sufficient-fit guarantee; root must accept/replace them explicitly. Artifact payload is only9,992,067B and tiny device payload is16MiB total; unknown prepare/transient overhead remains measured only after actual execution. In the independent CPU snapshot HOST1,722,754,654,208B/DISK9,526,936,862,720B/SHM1,079,343,157,248B were available, but refresh before/inside native job.

Every rank still must satisfy the actual publicfree guard, at least17,954,081,472B for empty A/C and that reserve+1MiB remaining before its tinyB allocation; after each allocation the reserve must remain. Record public total/free/committed, logical requested bytes, signed residual, timestamps, distinct owner identity, handle/free events and process-tree identities. Save receipts before ending; after task exit independently verify recorded own PID+starttick identities absent, queue COMPLETE0/locks released and unchanged source/artifact hashes. Do not query a closed owner or equate PID collection with zero leakage. Validate receipt bytes/tolerance/null peaks independently before even discussing a later largest/aggregate GO.

STOP on any pin/config/backend/scope/kind/rank mismatch, missing/invalid/stale receipt/query, insufficient host/disk/SHM/publicfree, pre/post guard violation, tolerance>256MiB, C regression, reused owner, allocation/schema/free/close failure or process/resource audit failure. Preserve first error/partial bytes, attempt only owned normal cleanup and end. No blind retry, ring enlargement, admin/reset/other-job interference or production guard release.

No new upstream defect is alleged. The late-final-query limitation is experiment-local scope/validation logic. This source review makes no fullfit/numerical/text claim.
