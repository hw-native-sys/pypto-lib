# Actual16 tiny empty calibration — noEngram; completed and externally audited

One root-authorized native CALIBRATE-only job `task_20261107_164303_25365942563`
completed exit0 and released all16 automatic locks. External independent CPU
audit `task_20261107_164529_255203212194` completed exit0. The actual A/B/C
handle counts were0/16/0, all16 tiny1MiB I8 handles were freed and all3 owners
normally closed. No largest allocation,57-resource ramp, weight tensor payload,
initialized weight upload, model forward or full-weight guard release occurred.
This is scoped native tiny-empty calibration, not real40 fit or text generation.

## Exact source/operation and queue evidence

Root's conditional prerequisite was satisfied FIRST: sibling decode capture
`task_20261107_163417_24812212428` COMPLETE0, immutable actual+initial manifest
424269b3042c1845802bee6ef0f14daddc08ffc6f32c6f5c043607e3ff03c33f,
normalclose0f89991d91f4b48e4f4a9b37891d43c5cde19a6df4c75de6516bd272a7f381fd,
all16 locks released and20 sibling ownPIDs independently absent. Our CPU
164235_25343695274 hashed/bound those stable receipts in decode_prerequisite.json
and refreshed machine resources before the calibration submission. No duplicate
own native calibration job existed (`task-submit --find` empty). Sibling later
CPU19 ran independently; no concurrent sibling NPU allocation was required.

Actual queued command is preserved exactly in native_job.json and PLAN.md;
automatic16, max1800/client2400, activated environment, explicit --mode calibrate
through the frozen adapter. Task-local auth was explicitly authorized by root
and bound actual TASK_DEVICE IDs0..15 after allocation; these were observed
assigned IDs, not inferred from compile-time placeholders. Normalized config is
ONBOARD/a2a3/public ring[0,0,4GiB,0], persistentFalse. Actual runtime logs contain
48 chip-process ready events and48 callable registrations, each rank0..15 three
times across the three prepared owners. Preparation uploads runtime callable
code/images as part of normal public setup; no model tensor payload was uploaded.
No model computation was dispatched.

Protected exact pins stayed unchanged in preflight, allocated job and external
audit: product45 manifest762dc4f7a64fe38ee7868bf5891411798be350ef6781ef35e4c58f7dfd2d89c7;
Main478f55ce7d59678f0977d466e673e3c0ad0429b5c26cdddf6253bf059cb9630c556;
adapter8d3ad68504cef4f0527594a37daa9dd8ef90041c0b56b8e199af30c99a5b83aa;
artifactreceiptc2cc6c2b1db13a67e202b160e4abd4c1434a9fe9f6825be9e1e629f05234fe2f;
actual283 generated realE40/57 prefill128 signature files, metadata3ef340dbe30e192bddd20ab2ec26bbe33b86453ffb0c6ce37d1512b847a50715;
core1542c1848227a0326655dc809ede1c71bca1e3d71d67f77a11e604741b924865;
installed TMR24434f8879030ef1b1f31e5fc2ecce683e392d4f1baeb5bcee852fb183976e5c
shared-GMbank0 contract. Old31/core48/review35/review70 manifests also unchanged.
Identity capacity168 was not treated as168 banks. No protected source/evidence
file was rewritten.

## Genuine measured results

Native runner interval was1794040989.2346792 ->1794041056.8482397,67.61356s.
This is the helper's recorded interval, not kernel time or an isolated performance
benchmark; preparation and control/cleanup are included.

| Measurement | Actual observation |
| --- | ---: |
| A prewarm committed, every rank |5,452,849,061B|
| Minimum A public free |59,883,593,728B|
| Minimum C prewarm public free |59,883,397,120B|
| B positive committed rise, every rank |1,048,576B|
| Maximum A->ALLB positive public-free drop |2,506,752B at rank12|
| Rank0 tolerance |68,214,784B|
| Rank12 tolerance |69,615,616B|
| Other14 rank tolerances |68,157,440B each|
| Resource-monitor samples |38|
| Minimum sampled machine MemAvailable |1,617,710,747,648B|
| Minimum sampled output-disk free |9,526,359,097,344B|
| Minimum sampled /dev/shm free |1,079,341,670,400B|

All rank-specific raw free/total/committed/time/requested/residual/handle/free/PID
details remain in actual/calibration.json, owner_records.json, A/B/C phase JSON,
B allocations JSONL, resource_samples.jsonl and native.log. No output tensor data
was read or saved; these files are telemetry/receipts. The CPU auditor independently
recomputed max positive A->ALLB free drop/committed rise INCLUDING tiny logical
bytes, then added the approved64MiB policy buffer. All values were below256MiB;
none was clamped. Both C initial and after-free snapshots met the SAME perrank
tolerance againstA. Public pre/post guards retained17,954,081,472B/rank reserve,
and B before-alloc queries also covered its remaining1MiB. Neither committed nor
ring/prewarm was subtracted again from free.

The1GiB fixed future side-cost allowance +untouched8GiB safety +N128 conservative
IO7,609,876,736B +nominalcomm680,528,320B remain PLANNING policy. This experiment
did not measure those planning values as physical peaks. Fixed1GiB deliberately
can overreserve paid side costs. Public-free drop includes driver/ambient effects;
it is not isolated allocator overhead. Machine resource minima include concurrent
host work; they are not this helper's isolated host/SHM/disk peak. Discrete query
and2s samples do not bound every between-query transient.

## External lifecycle/integrity audit

Native calibration SHA256 is
`3bc7291e174cccc8a8f0272d9786716a78bccb2ae8336c33fdfd4e10a16927f0`.
Independent CPU audit checked unchanged receipt SHA/bindings/policy/null peaks,
rank schemas/query timestamps, three ordered distinct owner lifetimes, exact
0-16-0 handle/free cookies, normalclose, device resource guards, tolerance raw
recomputation and C bounds. All50 recorded native PID+starttick identities were
absent after job end. Final collector additionally verifies the queue lock wrapper
2536729 absent. Native raw log records the16-device lock release; captured queue
snapshot shows0/16 devices occupied with no Pending/Running tasks at that instant.
No normalclose/PID snapshot is presented as a proof of zero leakage.

All own jobs before final collection completed0: CPU163406 preflight,
163714 syntax,164142 freshpreflight,164235 prereq binding; ONE native164303;
external CPU164529 audit. Native/source execution had no unexpected failure,
guard relaxation, larger-ring retry, administrator intervention or other-job action.
Final source/process/status/native48-event/ledger seal is in final_checks.json.

## Scope and next gate

This genuine new REAL40-signature/native preparation and tiny empty calibration
can be reviewed as an operation-specific receipt. It does NOT prove11.325GB
contiguity,39.625GB/rank static reservation, full57 residency, physical initialized
model-weight firsttouch, checkpoint upload/transformation peaks, HCCL/VMM forward
transients, model numerical fit or text. All full40 firsttouch/upload/forward
peak fields remain JSONnull, not zero. Hard production weight guard remains HELD.

Any later largest/57 empty job needs a separate root authorization, exact same
source/artifact/policy/ordered devices and a nonstale (<=3600s) accepted calibration
receipt. It is not authorized by this report. The separately identified final
large-probe after-free-query omission remains outside this job. Root can require
an independent external acceptance checker on that already recorded snapshot
againstA and aggregate prewarm at the original tolerance before any later large
PASS; no such large checker or allocation was implemented/executed here.

Reviewer seals this NEW directory and PARKS. No open fullfit/text/parity/dynamic
gate is closed by the tiny calibration result.
