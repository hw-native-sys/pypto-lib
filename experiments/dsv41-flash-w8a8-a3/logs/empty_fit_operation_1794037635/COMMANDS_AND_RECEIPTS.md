# Two separate future assignments — no actual execution authorization supplied

Both modes have operation=empty-allocation and defaultREFUSE. They reuse the
accepted corrected postguard1542 for every pre/post public-free check and cleanup.
They NEVER call old core.authorize/core.main or fill the old unknown
additional_peak_excluding_io_comm_and_already_committed_bytes array.

## Current artifact and policy

artifact_receipt.json names a NEW current-source, signature-only real40/E40/57
prefill128 ONBOARD/a2a3 artifact. All57 generated static headers/dtypes match the
sealed real-size budget. It compiled without Tensor samples/fixtures/prepare /
dispatch. The actual selected PROGRAM TMR installed library24434f... shares GM
bank0. Its identity directory168 is not a heap multiplier. Native execution
rechecks the installed public contract before preparing any owner.

Policy perrank is FIXED PLANNING, not a measured forward/allocation peak:

| Reserve | Bytes |
| --- | ---: |
| OriginalN128 conservative full hostIO allowance |7,609,876,736|
| Original nominal comm |680,528,320|
| Fixed future empty sidecost planning reserve |1,073,741,824|
| Untouched safety |8,589,934,592|
| Total retained reserve |17,954,081,472|

The full hostIO allowance is conservative planning scope, NOT measured N3 device
IO/sharding. Direct public free ALREADY reflects committed/ring/prewarm; do not
subtract them again. Before and AFTER every allocation require free>=remaining
static target+this SAME reserve. The fixed1GiB can conservatively reserve again
for side costs already paid. It is not an isolated allocator measurement, an
upper bound on total cumulative overhead, or a fullreal peak guarantee.
Full40 physical-firsttouch/upload/forward peaks remain JSON null at all stages.

## Root-assigned authorization schema

No valid actual authorization or measured native calibration is supplied here.
CPU fake receipts/auth files are explicitly CPU_FAKE_EMPTY_OPERATION and cannot
pass actual execution kind checks. For future rootGO only, one authorization:

    schema: empty-operation-authorization/1
    approved: true (only from an actual separate root assignment)
    scope: empty-allocation-calibrate OR empty-allocation-large-probe
    execution_kind: ACTUAL_DEVICE_EMPTY_OPERATION
    expires_at_epoch: actual integer, future and at most3600s
    binding: empty_operation.binding(artifact_receipt, ACTUAL16TASK_DEVICEids)

Binding includes operation/source45/adapter/core/artifactreceipt/policy SHA,
EXACT16 distinct physical device IDs in rank order, a2a3/ONBOARD/ring4GiB and
selected runtime library SHA. Populate actual assigned TASK_DEVICE values before
any prepare; never assume compile-time0..15 are the assigned IDs. A reviewer
under explicit rootGO may write the new assigned auth receipt after queue
assignment and before preparation. Probe additionally requires
calibration_receipt_sha256 of its ACTUAL calibration. Calibration scope cannot
authorize probe; timestamp/source/artifact/policy/config/rank changes reject.

## Phase1: calibrate-only (future rootGO, not run in this unit)

CPU preflight can run now only in reviewer-owned fresh output copies:

    task-submit --no-device --max-time180 --run 'cd <REPO> && source activate-pypto.sh && bash <UNIT>/run_final_cpu.sh'
    task-submit --no-device --run 'cd <REPO> && source activate-pypto.sh && python <UNIT>/empty_operation.py --mode calibrate --plan'

The self-tests intentionally refuse existing own output directories; independent
review uses a fresh own copy, not overwriting this sealed evidence.

Only after separate rootGO/resource checks, actual calibration command:

    task-submit --device auto --device-num 16 --max-time <assigned> --run 'cd <REPO> && source activate-pypto.sh && python <UNIT>/empty_operation.py --mode calibrate --execute --authorization <assigned-calibrate.json> --artifact-receipt <UNIT>/artifact_receipt.json --output <fresh-owned-calibration-dir>'

It performs ONLY A/B/C, no largest/57 ramp, dispatch or upload:

A: fresh public prepare/prewarm, public free/total/committed queries on all16,
normalclose. B: fresh prepare/query, one1MiB INT8 EMPTY(init=None) perrank,
publicbefore/after queries, freeall, query, normalclose. C: fresh prepare/query /
normalclose. One normalized public config is passed to each prepare. Every
public snapshot has rank/time/free/total/committed/logical-outstanding bytes;
handles/frees/owner epochs and own OS process-tree identities are retained.

For each rank, compare A prewarm with ALL B query snapshots: maximum positive
free drop or committed rise plus64MiB is the observed tolerance. Tiny requested
bytes contribute conservatively; this is not allocator-vs-ambient isolation.
If any result>256MiB STOP AFTER B CLOSE, no clamp/noC/no success receipt. C's
before/after snapshots must satisfy that same established tolerance againstA.
Write actual calibration.json only after all3 normalclose results. Raw requested,
committed and signed free-residual-minus-requested stay separate. Missing /
invalid raw measurements never become0; failure diagnostics are null where
unavailable. Errors preserve first cause and owner_records even without receipt.

## Phase2: separately authorized large empty probe

Only after independent ACTUAL calibration acceptance and a NEW separate rootGO:

    task-submit --device auto --device-num 16 --max-time <assigned> --run 'cd <REPO> && source activate-pypto.sh && python <UNIT>/empty_operation.py --mode probe --execute --authorization <assigned-probe.json> --artifact-receipt <UNIT>/artifact_receipt.json --calibration <actual-calibration>/calibration.json --output <fresh-owned-probe-dir>'

The calibration must be operation/source/artifact/policy/config/EXACTrank matched,
<=3600s old, all16 raw observations and tiny-handle/free/normalclose/tolerance
consistent, ACTUAL not CPU_FAKE. Reordered/different devices require a new
explicitly reviewed calibration/mapping, not silent reuse.

Largest11,324,620,800B perrank ->freeall/CLOSE ->FRESHprepare baseline ->57empty
resources39,624,816,576B/rank (912handles) ->freeall/CLOSE ->FINALFRESHprepare /
baseline/CLOSE. Compare fresh baselines to the established A/largest/aggregate
baselines at the calibrated tolerances. Largest and aggregate allocations never
coexist. Every allocation uses the independently accepted pre+post guard. No
uninitialized forward exists. Final normalclose/fresh queries are lifecycle
observations, not a proof of zero leaked bytes or unobserved between-query peak.

STOP on any missing/fake/null/wrong scope/pin/config/rank/time/shape/measurement,
public query failure, insufficientfree BEFORE/AFTER allocation, tolerance>cap,
C/finalbaseline regression, wrongschema, reusedowner, cleanup/close failure or
source/artifact mutation. Preserve first error, attempt frees for every registered
handle and normalclose; never resize/relax/relabel. At terminal job, independent
collector must recheck recorded own PIDs/resources and source hashes. Public
queries are never made against dead owners; per-owner process snapshots are raw
observations and do not assert that trackers were already gone while parent lived.

Even an actual large probe pass proves empty reservation/contiguous feasibility
only. Physical initialized touch, payload upload/transform/transient peaks,
full40 computation/numerics, realweights/text and fullweight guard remain open.
