# F04/F10 cohort: absolute C2 source coordinates and disjoint EP lanes

Main repairs F04 table construction; the native F10 worker exclusively repairs
moe_exchange.py. The prior F01/F02/F03/F08/conditional F14 acceptance remains
bounded as recorded. This cohort preserves compute equations, public ABI,
request/physical page mappings, quantization, TP4/DP4/EP16 and all numerical
thresholds. Main/worker checks are self-validation; observer-owned sibling
reviewer execution supplies independent acceptance after the final freeze.

## F04 source coordinates

The shared host helper `compressed_source_positions(position_ids, ratio)`
returns current positions for C1 and absolute pair origins for C2: an emitting
odd position p maps to p-1, and nonemitting rows map to0. It does not inspect
compressed lengths, logical row IDs, cache slots or physical block tables.
All actual C2 builders now use it: prefill/decode source fixtures, shared layer
fixtures, both complete-forward fixtures, and the generator's compressed-source
table helper/callers. C1/SWA profile computation and accepted F01–F03 transform
order stay unchanged. No generator F11–F13 or Engram fix is included.

Independent authority is official rc2 revision
`8fde6d4e335fe300e2b949aab0e31659cce0811d`, original dsa_v41.py metadata
construction at1340–1349. `f04/check_coordinates.py` verifies exact preserved
source bytes against archive SHA256
`48c797546805b04ffde118c506f8902e37e19bf2cb5d1713097056a4c93e998c`,
AST-extracts and executes its original source-coordinate expression unchanged,
and checks the candidate helper against it. Position1253 produces source1252,
independent of compressed row626 or any physical address.

CPU tests exercise actual source builders, actual layer/forward metadata-builder
prefixes and the actual generator helper with row-tagged tables. Prefix execution
stops before irrelevant weight allocation; it executes the owning metadata
statements and preserves their profile selectors. Altering compressed lengths,
slots and page tables does not change coordinates. C1 current-position semantics
remain exact. This validates coordinate choice; existing generation profile/ABI
and metadata-lifecycle defects remain open.

The device negative control uses the actual frozen old decode builder's
compressed-address-derived rows with the unchanged accepted source kernel. The
golden derives correct source positions from the independently executed original
expression. Both compressed and index caches must reject numerically; a compile
or runtime failure cannot pass the control. The corrected high-position case
passes against this reference.

Actual continuation runs first process even positions1248/1250 for two requests,
capture device-returned state/cache tensors, and hand those actual values into
odd-start positions1249/1251. Both dispatches reuse one compiled artifact/work
directory. Thus source rows1248/1250 are consumed at runtime with actual
compressor state, rather than recomputed candidate golden state. This is a host
handoff of actual device state, not a claim of persistent device-resident cache
lifecycle or complete-model F09 support. Three positive coordinate executions
and one numerical negative run are recorded in f04/device/results.tsv.

F04 changes only host coordinate construction. All compute-kernel ASTs in
Main-owned files are unchanged. Source emission masks, physical addressing,
unrotated index input, source-rotated attention values and query-position inverse
output rotation remain preserved.

## F10 shared capacity stride

Worker `/root/main_native/f10_stride` owns only moe_exchange.py and its artifact
directory. The receive-lane bases for push and compact now use shared capacity
instead of each peer's active token count. Existing allocations and return
staging already use capacity. Active counts control payload/count processing,
while source/destination agree on reserved lane geometry. The worker AST check
confirms only those two expressions change within the real shared moe_ep16.

CPU geometry demonstrates the old capacity>active overlap. Actual16-device
old equal-active1/capacity2 execution is numerically rejected on37776/163840
routed elements (maxabs1.23242) at unchanged .02/.02/1% gate. Repaired equal1,
unequal `[0,1,2,1]*4`, and idle `[0]*8+[1]*8` patterns pass on all16 devices.
A second job replays the same three patterns in one process through one
compiled dynamic entry and requires exact inactive-sentinel preservation.
Six positive executions repeat three designs, not six distinct designs.

These tests call the actual shared primitive through a bounded capacity2
artifact harness. They do not change the full-forward active-row contract,
public model capacity, F09 routing/lifecycle or serving interface. F10 evidence
does not accept full-model unequal/idle DP behavior.

## Executed checks

All CPU/compile checks use task-submit --no-device; NPU checks use automatic
queue allocation (one card for Main coordinate/downstream checks,16 for the
worker primitive tests). Runners activate the existing environment and prepend
pinned PTOAS. No simulator acceptance, environment change or relaxed gate.

| Task | Result |
| --- | --- |
| task_20261106_162214_36114236461 | Main builder adaptation/scoped lint, exit0 |
| task_20261106_163044_366157020207 | Initial CPU coordinate assertion stopped on int32/int64 dtype mismatch, preserved |
| task_20261106_163258_36685976220 | Corrected original/actual-builder CPU checks, exit0 |
| task_20261106_163358_36719224972 | Main coordinate device0: old-builder numerical rejection, corrected positive, two actual-state continuation dispatches, exit0 |
| task_20261106_163607_368475031144 | Main downstream17 executables/37 numerical positives, device1, exit0 |
| task_20261106_163607_368475121633 | Main lint and both complete-forward compiles, exit0 |
| task_20261106_162544_362210947 | Worker actual16-device old numerical control plus equal/unequal/idle positives, exit0 |
| task_20261106_162849_364549219514 | Worker same-artifact16-device three-pattern replay/exact inactive sentinels, exit0 |
| task_20261106_163302_366908915185 | Worker final source/evidence collector, exit0 |
| task_20261106_164342_37365031819 | Main final scope/kernel-AST/case/worker-manifest collector and lint/syntax, exit0 |

The initial Main host assertion compared equal-valued int64 table indices with
the original int32 position expression; the test now compares table-index dtype
consistently. The worker's first artifact harness needed an explicit INDEX cast
for dynamic slice extent before PTOAS accepted it. Both failures are retained
own-test integration mistakes, not numerical controls or upstream defects.

Main's downstream default coverage includes all12 phase entries, accepted MLA,
output projection, routing, compressor and indexer. Its37 numerical cases plus
three coordinate positives total40 Main positive device executions; the worker's
six primitive executions are separately attributed. Both complete-forward
compiles establish integration/lowering, not full-model runtime publication.

Coordinate job epochs1793954042–1793954200,device0; downstream epochs
1793954171–1793954524,device1; both allocations released. Both forward compiles
run at epochs1793954175–1793954233. Main appended MODULES current-coverage
overlay with these actual default/artifact distinctions; no interface change.

Final45-source after.json SHA256
`bf635491416ffc893715041588683492547d3560ac9d7f1f07572bbfd652f089`.
Exactly8 sources change: the seven recorded Main host/table files and worker
moe_exchange.py. Other37 remain byte-identical; all F04 compute-kernel ASTs
remain unchanged. F10 source SHA256
`951378f40c2ab7cb4f1573aeda72548bab017ed76c545641620e6e3502b4c780`.
Both continuation dispatches name the identical compiled work directory, and
the second explicitly carries captured actual device state/cache inputs.

## Fresh reruns

From repository root, choose fresh absolute output directories:

```bash
task-submit --no-device --max-time 240 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && F04_EVIDENCE_DIR=/absolute/fresh/host python experiments/dsv41-flash-w8a8-a3/logs/cohort_f04_f10_1793952989/f04/check_coordinates.py host'
task-submit --device auto --max-time 900 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && bash experiments/dsv41-flash-w8a8-a3/logs/cohort_f04_f10_1793952989/f04/run_device.sh /absolute/fresh/coordinates'
task-submit --device auto --max-time 1800 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && bash experiments/dsv41-flash-w8a8-a3/logs/cohort_f04_f10_1793952989/run_downstream.sh /absolute/fresh/downstream'
task-submit --device auto --device-num 16 --max-time 900 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && bash experiments/dsv41-flash-w8a8-a3/logs/cohort_f04_f10_1793952989/worker_f10/run.sh /absolute/fresh/f10'
task-submit --device auto --device-num 16 --max-time 900 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && bash experiments/dsv41-flash-w8a8-a3/logs/cohort_f04_f10_1793952989/worker_f10/run_replay.sh /absolute/fresh/replay'
```

Worker CPU geometry supports F10_EVIDENCE_DIR; its report has exact commands.
The accepted F02 run_compile.sh accepts a fresh absolute directory for both
forwards. Never overwrite Main/worker evidence. Every source hash/output verdict
and actual queue allocation is in after.json/unit_summary.json and raw logs.

## Remaining scope

No original torch_npu metadata/quantization primitive, exact checkpoint or full
model equivalence is claimed. Earlier low-amplitude/tie quantization-parity
limits persist. Engram F05/F06,2048 candidate capacity,F09 full-forward lifecycle,
generator F11–F13,conditional checkpoint offset incidence,ordinary16-device
layer suites,standalone TP,CI and composed full-model decode pool/state gates
remain OPEN/HELD. Both full-model device gates need fresh validation after the
remaining repairs; historical PASSes cannot accept this tree. Generation stays
HELD. No protected-source/checkpoint write,commit,external filing or simulator
validation. Known issues inspected; no toolchain/issue status change.

Freeze one final45-source manifest after all writers/jobs park. Main/worker
self-tests are not independent acceptance. Await observer-owned sibling
reviewer execution and observer's next bounded assignment before further edits.

STOP: all Main/worker writers,tool sessions and queue jobs have ended; all own
allocations released. Cohort source is frozen on after.json. Worker record
f10_stride-1793953131-1793953986 was imported exactly once under the shared lock;
parallel durations are not summed as elapsed wall time. Recover from this
report,TASKS/TIME_LOG,unit_summary.json and the worker report. No further
implementation before independent sibling disposition and next assignment.
