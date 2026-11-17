# Native16 empty reservation accepted by internal AND external checks — noEngram

ONE auto16 native job170442_307058025803 completed exit0. Independent external
CPU audit170742_308264929763 completed exit0. Three sequential distinct owners
allocated16 largest EMPTY buffers, then912 EMPTY handles (57resources/rank), then0
handles in the final fresh baseline owner. All928 allocated handles were freed;
all3 owners normally closed, all16 locks released and all50 recorded native
PID+starttick identities plus lock-wrapper3070688 were absent after job end.

This proves the requested empty allocation/contiguous-buffer feasibility at
11,324,620,800B perrank and logical57-resource reservation39,624,816,576B/rank.
It does NOT prove initialized weight firsttouch, checkpoint payload upload,
forward transients/numerics/fullfit/text or release the production hard guard.
All full40 firsttouch/upload/forward peaks remain JSONnull. No weights were
initialized and no model computation was dispatched. Normal preparation uploads
runtime callable code/images; it did not upload checkpoint tensor payload.

## External gate covers the previously recorded omission

Frozen Main8d3ad/core1542 and accepted actualcalib3bc729 remain unchanged.
NEW accept_reservation.py SHA
bac7748403064849ea05ee4c450061425d7e8194c738d9310696b4213fc32523
adds a separate acceptance verdict. It requires final_fresh.after_free_before_close
on ALL16 ranks to satisfy BOTH free-drop and committed-rise against BOTH actual
calibrationA_baseline and aggregate.prewarm at the ORIGINAL perrank tolerances.
It also checks source/artifact/policy/ordered-rank/calibration SHA, exact resource
and handle schemas, raw query structure, before/after remainingstatic+same reserve,
normalclose/free ownership and phase/raw-owner consistency. No tolerance was
changed, no old source patched and no internal status was relabeled.

Independent tiny CPU job170050_30608372498 COMPLETE0 read the old sealed70 raw
probe_late_final_drift case. Its internal CPU_FAKE success remains preserved,
but NEW external gate rejects66MiB>65MiB final after-free. Exact free/committed
tolerance boundaries PASS; +1B free drop and +1B committed rise each reject.
Missing ranks, wrong rank/source/calibration SHA, raw shape, null public query
and not-normal-close reject. Test copies live only in this NEW unit. See
validator_cpu_results.json and cpu_cases. No Main self-test was reused as our
independent proof.

Actual raw probe status remains EMPTY_RESERVATIONS_ONLY, while external_verdict.json
separately records PASS_EMPTY_RESERVATION_WITH_FINAL_AFTER_FREE_GATE. Actual
probe SHA is c4b62d80662c62058f4ba4a237e606e745b07d8506eb45c0e9ccb098aeec953e.
The post-job CPU audit reran this external checker against immutable actual raw
states and additionally audited native evidence/process cleanup/source pins.

## Root scope, pins and execution

Root explicitly corrected its initial shortened scope spelling to the frozen
literal empty-allocation-large-probe before submission. Task-local auth encoded
that scope, ACTUAL_DEVICE_EMPTY_OPERATION, accepted calibration SHA and exact
current binding; the correction is preserved in native_job.json. No adapter
change was used to accommodate authorization. mode=probe and operation=empty-allocation.

Accepted actual calibration SHA
3bc7291e174cccc8a8f0272d9786716a78bccb2ae8336c33fdfd4e10a16927f0
was1232.256829s old at allocated preflight, within3600s. Observed automatic
TASK_DEVICE orderedIDs0..15 matched its exact mapping before ANY prepare.
No compile-time physical mapping was assumed. No duplicate own native request
existed; prequeue snapshot showed0/16 occupied. Root formal policies were retained:
HOST>=64GiB/disk>=2GiB/SHM>=16GiB, refreshed before queue and inside allocation,
then monitored every2s and before each prepare. No resource-monitor violation/error.

All protected pins were independently checked before queue, inside allocated job,
post-job external audit and final seal: product45 manifest762dc4f7a64fe38ee7868bf5891411798be350ef6781ef35e4c58f7dfd2d89c7;
Main478f55ce7d59678f0977d466e673e3c0ad0429b5c26cdddf6253bf059cb9630c556;
adapter8d3ad68504cef4f0527594a37daa9dd8ef90041c0b56b8e199af30c99a5b83aa;
artifactreceiptc2cc6c2b1db13a67e202b160e4abd4c1434a9fe9f6825be9e1e629f05234fe2f;
all283 generated realE40/57 prefill128 files/metadata3ef340dbe30e192bddd20ab2ec26bbe33b86453ffb0c6ce37d1512b847a50715;
core1542c1848227a0326655dc809ede1c71bca1e3d71d67f77a11e604741b924865;
installed TMR24434f8879030ef1b1f31e5fc2ecce683e392d4f1baeb5bcee852fb183976e5c
shared-GMbank0 contract; accepted calibration38/review70/review35/core48/old31.
No protected file was rewritten. Identity capacity168 was not charged as168banks.

Public normalized ONBOARD/a2a3/ring[0,0,4GiB,0]/persistentFalse preparation used
the frozen real40 signature artifact. Native logs have48 chip-ready and48 callable
registration events (three per physical rank0..15), supported by raw owner/query
and queue evidence. Library/metadata/mode labels alone were not the acceptance.
Root controls and no forward/upload path were retained. Exact command:

```bash
task-submit --timeout 2400 --device auto --device-num 16 --max-time 1800 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && PYTHONDONTWRITEBYTECODE=1 python experiments/dsv41-flash-w8a8-a3/logs/native_empty_reservation_1794041910/run_reservation.py --execute'
```

Native runner interval1794042289.0948358->1794042382.8048997 is93.710064s,
including preparation, queries, allocator controls, close and external gate.
This is not kernel time or a benchmark claim. No unexpected failure, retry,
ring resize, administrator intervention or other-job action occurred.

## Actual memory observations and guard accounting

| Actual perrank public observation | Bytes |
| --- | ---: |
| Minimum largest-phase prewarm free |59,884,064,768|
| Minimum free after largest allocation |48,557,408,256|
| Minimum aggregate-phase prewarm free |59,885,076,480|
| Minimum free after last57th allocation |20,251,516,928|
| Maximum committed after last57th allocation |45,077,665,637|
| Minimum final fresh prewarm free |59,884,154,880|
| Minimum final after-free/before-close free |59,884,154,880|

The fixed reserve remains17,954,081,472B/rank: conservativeN128 hostIO7,609,876,736
+nominalcomm680,528,320 +fixed PLANNING1GiB +untouched safety8GiB. Initial aggregate
admission required57,578,898,048B/rank. After the last57th resource, the minimum
free exceeded the full reserve by2,297,435,456B. EVERY allocation's before query
covered remaining logical static target+reserve; EVERY after query covered the
decremented remaining target+same reserve. Exact shape/dtype/nbytes were checked
for all16 largest handles and912 aggregate handles, with rank/owner/free cookies.
Largest and aggregate were never resident simultaneously; normalclose preceded
the next owner. Final prewarm AND final after-free snapshots passed original
calibration tolerances against both required reference baselines.

No committed/prewarm/ring subtraction was added to direct publicfree. Fixed1GiB
is openly conservative over-reservation, not measured isolated sidecost or a
cumulative cap. N128 IO/nominalcomm remain planning allowances, not actualN3
device peaks or executed communication measurements. Public committed is the
allocator measure, not all driver/HCCL/VMM memory. Allocation success and these
snapshots do not certify all physical firsttouch or between-query transients.

The51 machine resource samples had minimum hostavailable1,719,528,116,224B,
diskfree9,526,328,688,640B and SHMfree1,079,341,514,752B. These are machine-wide
sampled minima, not isolated per-operation peaks. Raw times/free/total/committed/
requested/residual/handle/free/owner/PID data remain preserved in actual/, native.log,
resource_samples.jsonl, audit_results.json and native_outcome.json.

## Cleanup, evidence and remaining scope

Independent external CPU audit170742 verified exact16/912 schemas/bytes and
pre/post guards,3 ordered distinct owner lifetimes/all free cookies, actual Root
binding/calibration/pins, final extra gate,48ready/48registrations, native
COMPLETE0 and lock-release lines, queue snapshot0/16, all50 recorded process
identities absent and wrapper3070688 absent. Final collector verifies these
receipts/source hashes again, records all own jobs terminal and appends one real
review-agent time record under coordination lock. No close/PID check is described
as proof of zero leakage.

Only the empty reservation gate is completed here. Initialized real-weight upload,
host streaming/transformation staging peaks, physical model-weight firsttouch,
full40 forward/numerical fit and text still require separately authorized bounded
work and remain OPEN. Production full-weight guard remains HARD/unchanged. This
unit is sealed and reviewer PARKS; no further NPU job is scheduled.
