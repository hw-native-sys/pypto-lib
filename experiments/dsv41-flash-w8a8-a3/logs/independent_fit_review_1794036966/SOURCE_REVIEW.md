# Independent fit source review — noEngram, CPU evidence only

Reviewed production source45 `762dc4f7a64fe38ee7868bf5891411798be350ef6781ef35e4c58f7dfd2d89c7` and Main generation unit `generation_entry_1794035858`. Production and all 31 sealed Main artifacts were independently hashed by CPU job `task_20261107_153727_96880910110` (COMPLETE, exit 0). No model, golden, checkpoint, library, sealed helper or old evidence was modified. No NPU job, tensor payload read, real tensor allocation, fixture, forward or text generation occurred. Original goal remains W8A8/TP4/DP4/EP16 on 16 real cards; temporary first text is explicitly noEngram.

## Findings requiring disposition

1. **Post-allocation guard missing (reproduced).** Frozen `generation_entry_1794035858/fit_probe.py:95-108` checks `free >= remaining + reserve` before alloc, then records after-state without testing it. An allocator side cost on the last allocation can consume reserved headroom and still return EMPTY_ALLOCATIONS_PASS. `review_cpu.py` independently reproduces single-largest fake final free50 < reserve100, and aggregate57/rank = 912 tiny fake handles final free80 < reserve100 on all16 ranks. These are integer mocks, NOT physical bytes evidence. Every handle is freed and owner close attempted. See `fake_post_alloc_reserve_overrun.json`, `fake_aggregate_last_alloc_overrun.json`, `cpu_results.json`. Main owns a separate NEW repair unit, not this reviewer directory. Required fix: record post-query first, then fail if `after.free < remaining + unchanged reserve`, including final allocation; preserve first error and cleanup. An after-query failure must also fail/cleanup. Validate at a nonzero rank and at the last resource.
2. **Calibration operation attribution is unresolved, creating a circular gate if interpreted as full40 forward peak.** `authorize()` requires an accepted bounded memory file; `main()` requires16 nonnegative integers named `additional_peak_excluding_io_comm_and_already_committed_bytes`, but neither describes their measured operation/interval nor verifies receipt contents. The predecessor REPORT explicitly includes “other uncommitted forward peaks” in this term. Full40 forward/upload has never run. Requiring that unknown full40 peak before an EMPTY allocation experiment would require completion of the target experiment before its reservation prerequisite. Filling the array with zero or oldE1 values would be false evidence. `vector()` syntactically accepts sixteen zeroes; the CPU test proves only this schema fact and supplies no allocation authorization.
3. **Largest-to-aggregate fresh baseline is useful but final aggregate CLOSE is not independently observed afterward.** `run_phase` records after-free/before-close and successful `__exit__`; `run_two_phases` compares a fresh aggregate owner with the first owner prewarm. It does not create a final fresh owner after aggregate. A final baseline receipt needs a separate fresh prewarm/normal-close phase (no allocations), with explicit observed tolerance. Successful close return is supported API lifecycle evidence, not a post-close HBM query on a dead owner.
4. **Old full IO allowance is N128/two-page planning scope.** The sealed readiness budget describes prefill128/decode-capacity1/two pages. Reusing its full host IO payload perrank is deliberately conservative; it is not the measured device IO for new exactN3. An actual native artifact must separately bind its prompt/pages/ABI and57 real statics. Do not relabel old IO bytes as current measured sharding.

## Verified accounting and API contract

| Per-rank quantity | Prefill planning | Decode planning |
| --- | ---: | ---: |
| Real57 static bytes | 39,624,816,576 | 39,624,816,576 |
| Largest single static | 11,324,620,800 | 11,324,620,800 |
| Conservative full host IO allowance | 7,609,876,736 | 247,111,680 |
| Generated nominal communication bytes | 680,528,320 | 4,663,744 |
| Static + IO + nominal communication | 47,915,221,632 | 39,876,592,000 |

These are schema/planning quantities, not measured device peaks. Target4GiB ring2 + three default256MiB rings = 5,100,273,664B = 4.75GiB per selected shared bank/rank. The selected a2a3 TMR PROGRAM contract shares GM bank0;168 is run identity capacity, not168 resident banks. `device_memory_info()` reports device-wide free/total, already accounting for prewarmed and committed memory. `committed_device_memory()` covers this worker allocator (tensors/pooled/runtime arenas); it is diagnostic and does not include every HCCL/VMM/other-process allocation. Never calculate `free - committed`, subtract ring again, or invent bank multipliers.

Public API source inspection:

- `DistributedCompiledProgram.prepare(config, persistent=False)` builds/loads/registers/inits and prewarms using that config. It needs no tensor samples. Config is used for prewarm; every later dispatch would need its own same config, but this experiment dispatches nothing.
- `DistributedWorker.alloc_tensor(shape,dtype,init=None,worker_id=rank)` uses malloc and a tracked DeviceTensor. In `alloc_device_tensor`, copy_to is conditional on `init is not None`. No initial data or weight upload follows from init=None. Runtime allocation may map/commit physical backing internally; public allocation success cannot establish payload first-touch correctness or upload peak.
- `free_tensor` releases the correct worker allocation; allocator caches may remain. `__exit__` calls normal `close`, closes owned tensors and native worker, propagating first teardown error. Query the fresh next owner, never a closed worker.
- `device_memory_info` errors propagate; no zero fallback. A failed query is STOP.

Installed exported C API contract and exact public API file hashes are independently recorded in `contract_results.json` by separate CPU-only job `task_20261107_153844_97183731254` (COMPLETE, exit0). Runtime library expected SHA is `24434f8879030ef1b1f31e5fc2ecce683e392d4f1baeb5bcee852fb183976e5c`. This verifies the installed binary contract without device preparation; it is not fresh device telemetry.

Current checked toolchain: PyPTO `38a8aa6da24c62bf0c6f6f9ef94deb513fbb4d2e`; simpler `31f9e72ee6832cd198b77df18c1c305e7fd83676`; PTO ISA `f83d4b7e4be0a4ca9805944ae2d9ce8c84e581f7`; PTOAS0.72. Main source manifest and31artifact hashes are in `cpu_results.json`; runtime API hashes are in `contract_results.json`.

## Meaning of each proposed measurement

| Stage | Honest measurable result | What remains unknown |
| --- | --- | --- |
| CPU metadata/header-only |57 actual ABI shapes/dtypes/bytes; installed backend pins | Every device peak |
| Native prepare/prewarm only |16 baseline free/total/committed snapshots; preparation lifetime and sampled minimum | Unobserved transient peaks; allocation/upload/forward |
| Tiny empty calibration (e.g.1MiB/rank) | Allocation/free snapshots and observed residual/cache/close variation for THAT operation |11.325GB contiguous success; aggregate39.625GB; larger-size allocator overhead |
| Largest empty once/rank | Public allocation/contiguous feasibility and observed side costs at11.325GB |57 simultaneous allocations; initialized data/upload/compute |
|57-resource empty ramp |912 handles, retained schema39.625GB/rank and observed free/committed at each query | Real checkpoint payload, compute, numerical fit and between-query peaks |
| Explicit payload first-touch/upload | A future separately authorized bounded initialized transfer's measured lifetime and physical backing behavior | Full40 forward peaks until actually executed |
| Real full40 forward | Its actual sampled/profiled peak and original numerical gates | Other prompts/shapes/longer history without evidence |

An old boundedE1 prewarm (~5.453GB committed/~59.886GB minfree) is only historical comparison and source of candidate calibration questions. Its static allocation is6,463,879,104B/rank,33,160,937,472B below real57. Its forward sample is not a measured full40-expert peak. CPU peakPSS/host serial source allowance5,362,417,664B does not become device headroom. Publicfree already deducts committed prewarm. Empty reservation does not read true weights or run any uninitialized compute.

## Minimum honest route — no NPU authorization issued

1. Finish independent new source/odd-N host/functional disposition, new post-allocation guard review and native REAL40 signature artifact proof. Signature-only compile uses `task-submit --no-device`, activated environment, zero tensor samples; no global dummy stacks. Pin generated a2a3/TMR metadata, all57 statics, library and source45. Existing E1 artifact metadata is insufficient.
2. Assign a separate **empty-only calibration scope**. Use explicit ONBOARD/public4GiB config on exactly16 automatic queue devices. Measure two sequential prepare/prewarm/normal-CLOSE cycles; optionally a tiny empty alloc/free cycle perrank between them if root authorizes it. Requery every rank, save raw intervals and cleanup receipts. Derive observed close baseline differences for this source/artifact/config; record ambient device variation. A post-close tolerance is a reviewed policy informed by those measurements, not an invented universal leak threshold.
3. Record a narrow empty-operation budget receipt: `operation=empty-allocation`, actual artifact/phase/shape/ring/source/library,16 baseline samples,16 measured tiny/reservation side-cost samples with explicit interval/sample method, separately named16 POSITIVE policy safety allowances, and16 observed normal-close differences plus assigned tolerances. Missing or failed measurements stay null/STOP. The future **unexecuted full40 first-touch/upload/forward peak remains null** in a separate field. Do not feed it to empty admission. Query samples yield observed minima, not a guaranteed instantaneous high-water mark.
4. A NEW narrowly scoped artifact adapter is needed if the frozen schema continues to demand full-forward peaks. It must accept only the reviewed empty-operation receipt and explicit root scope, retain the existing conservative IO/nominalcomm allowances and positive planning safety allowance, enforce both before/after free guards, retain no-compute/no-upload behavior, and keep default refusal. This is a change in experiment attribution, not permission to claim full-model fit or relax production. No adapter is implemented here and no pending/null value is coerced to zero. Tiny measured side costs are not guaranteed larger-allocation peaks: reserve an explicit positive POLICY allowance and stop at the first unexpected overhead/headroom violation.
5. Execute largest11,324,620,800B once on EACH rank, record queries, free all, normal-CLOSE. On success only, a fresh owner must meet the observed baseline/tolerance before57resources/rank (912handles); before AND after EACH allocation verify `publicfree >= remaining target static + conservativeIO + nominalcomm + outstanding empty-operation allowance + positive safety margin`. As logical target bytes decrease, allocations already committed stay represented in publicfree; never add them again. An observed side cost already deducted from free is not charged a second time as outstanding cost. After aggregate free/CLOSE, do one final fresh prewarm baseline/normal-CLOSE measurement.
6. Report ONLY empty reservation/contiguous feasibility. Physical first-touch/initialized upload and real checkpoint streaming hostIO/HCCL/VMM/forward peaks require later separately bounded measurements, admission and original numerical acceptance. Full-weight hard guard remains HELD throughout.

No full40 forward is a prerequisite to steps2–5. No resource quantity is declared measured unless the exact scoped operation actually ran. If existing preparation itself cannot fit safely under assigned positive policy allowance, STOP without trying largest; no opportunistic upload, larger ring or silent retry.

Root's subsequent proposed policy (not measurements or NPU authorization) is1GiB/rank empty-operation side-cost allowance plus8GiB/rank safety. With the existing conservative prefill allowance this requires57,578,898,048B/rank at initial admission. The new real40 baseline must satisfy it; historical E1 free cannot authorize it. Keep8GiB untouched safety constant. If1GiB is a TOTAL cumulative side-cost cap, already observed/committed side costs are in publicfree; track consumed versus outstanding explicitly and STOP when observed cumulative cost exceeds the cap. Do not reserve the consumed part again. Save baseline/current committed, logical live bytes and devicefree residual separately; allocator pooling and ambient/HCCL changes complicate attribution. If attribution cannot support subtraction, a constant1GiB reserve is conservatively larger but must be honestly described as over-reserving, not as exact non-duplicated accounting.

For proposed close tolerance, compute perrank nonnegative adverse observed drift (free reduction/committed increase) plus64MiB POLICY buffer. If this exceeds256MiB, STOP rather than clamp a larger observation into a passing256MiB threshold. Two prewarm cycles are finite observations, not universal cleanup or peak guarantees. Root owns final policy and any later GO.

## Commands and stops

Executed CPU commands (repository root, no devices):

```bash
task-submit --no-device --max-time 180 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && python experiments/dsv41-flash-w8a8-a3/logs/independent_fit_review_1794036966/review_cpu.py'
task-submit --no-device --max-time 180 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && python experiments/dsv41-flash-w8a8-a3/logs/independent_fit_review_1794036966/read_contract.py'
```

Future executable entry for current metadata only (not allocation):

```bash
task-submit --no-device --max-time 180 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && python experiments/dsv41-flash-w8a8-a3/logs/generation_entry_1794035858/fit_probe.py --plan --phase prefill'
```

The allocation command cannot honestly be filled in with existing receipts: there is NO approved empty calibration or independent real40 additional-peak receipt. After a narrow reviewed adapter and rootGO, command form is `task-submit --device auto --device-num 16 --max-time <assigned> --run 'cd <REPO> && source activate-pypto.sh && python <NEW_EMPTY_ONLY_ADAPTER> --execute --authorization <ROOT_EMPTY_SCOPE_AUTH> --measurements <EMPTY_OPERATION_RECEIPT> --artifact <REAL40_A2A3_ARTIFACT> --output <FRESH>'`. Do not run the frozen old helper with fabricated calibration to make this command executable.

STOP on changed source/library/backend/config, missing operation attribution/receipt, missing functional disposition, wrong device count, public query failure, before OR after headroom loss at any rank, allocation/schema failure, unexpected owner reuse, bad fresh baseline, cleanup failure or lost owner. Preserve first error/partial telemetry, attempt only owned normal cleanup, end job. No blind retry, ring resize, administrator/reset action, protected guard edit, checkpoint payload or other-job interference.

This review does not allege a new upstream compiler/runtime defect: the demonstrated bug is in experiment-local admission logic and is being repaired as such. No kernel/toolchain change or new upstream issue was required.

## Independent repair validation — completed

Main froze NEW `fit_post_guard_1794037055/fit_probe.py`, SHA `1542c1848227a0326655dc809ede1c71bca1e3d71d67f77a11e604741b924865`. Independent source diff inspection confirms exactly two added lines after existing telemetry/JSONL write, checking postallocation free against remaining target plus unchanged reserve. No calibration/authorization/config/cleanup implementation changed.

Independent CPU job `task_20261107_154503_99005219350` COMPLETE exit0 ran our own `check_candidate.py`, not Main's self-test. Last rank15 largest overrun and last rank15/resource57 (all912 tiny mock handles allocated) overrun both correctly FAIL, preserve after-query telemetry, free ALL16/912 handles and normally close. Exact headroom equality100==100 passes. An injected postallocation public query error correctly fails and frees all16 already registered handles/close. `candidate_results.json` and per-case JSON/JSONL preserve the actual checks. Production45 and old Main31 hashes remained unchanged. Candidate48-artifact/source pins are checked again by final collector.

```bash
task-submit --no-device --max-time 180 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && python experiments/dsv41-flash-w8a8-a3/logs/independent_fit_review_1794036966/check_candidate.py --candidate experiments/dsv41-flash-w8a8-a3/logs/fit_post_guard_1794037055/fit_probe.py --sha256 1542c1848227a0326655dc809ede1c71bca1e3d71d67f77a11e604741b924865'
```

Disposition: the narrow postallocation admission/cleanup fix passes independent CPU validation. Old falsePASS evidence and old sealed helper remain preserved. Calibration attribution, physical first-touch/upload/forward peaks, realweight fit, text generation and fullweight hard guard remain OPEN/HELD; root owns any later scope/authorization.

Final-collector first attempt `task_20261107_154614_9930505208` exited1 before ledger/seal writes: this host's task-submit CLI rejects even a nested `--status` query from inside its daemon. Raw failure is preserved in `finalize_initial_failure.log`. Correction moves read-only queue status queries to the client, stores their raw `.status` outputs, and lets the CPU-only final collector verify those receipts and source hashes. No guard/test/production implementation changed for this collection error.
