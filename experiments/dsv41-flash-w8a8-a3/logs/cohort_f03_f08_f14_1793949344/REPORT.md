# Parallel repair cohort: F03 compressed values, F08 routing ties, F14 offset folding

This cohort repairs three separate contracts while preserving accepted F01/F02
and indexer/reindex behavior. Main owns F03 and shared integration; two native
workers own only moe_gate.py and weight_loader.py respectively. Both workers
parked before the integrated validation sweeps. All recorded writer scopes and
unique measured worker records are in TASKS/TIME_LOG. Main/worker validation is
self-evidence; the observer assigns an independent sibling reviewer/validator
for acceptance after this cohort freezes.

## F03: keep index latent raw and publish rotated attention values

All four prefill/decode C1/C2 source chains now use the normalized unrotated
compressor output for index-key projection. A separate BF16 tensor receives
forward source-position RoPE, is snapshotted and published into compressed
attention cache. The original compressor output stays live for both consumers.
Both source goldens and both composed/staged layer-reference publication paths
apply the same required transform. The shared rotary kernels themselves are
unchanged; rope.py adds their owning Torch golden. Public source/phase/layer/
forward signatures, slot mappings, emission masks, W8A8 policy and topology
remain unchanged. F02 still inversely rotates attended values at query position
before grouped Wo_a/BF16/Wo_b.

Pinned official rc2 `8fde6d4e335fe300e2b949aab0e31659cce0811d` remains the
authority. The preserved archive SHA256 is
`48c797546805b04ffde118c506f8902e37e19bf2cb5d1713097056a4c93e998c`.
F03 oracle verifies the exact original dsa_v41.py and rotary CPU-golden bytes
against that archive, then AST-executes `_write_compressed_source` and
`_apply_rotary` with explicit compressor/index-record/cache stand-ins. Original
source updates index K before it rotates latent values, then publishes long KV
at source slots. Both C1/C2 execution paths match a separate complex-multiply
oracle exactly at T1/T7/T17; captured index input remains exactly unrotated.
A separate compressor arithmetic equation matches the four owning goldens
exactly on the test fixtures. The original flow test starts with supplied
normalized BF16 latents; it does not execute original compressor NPU primitives.

Strong varying source angles distinguish this fix. Device controls exercise
actual frozen pre-repair source kernels for all four paths and require numerical
compressed-cache rejection. Four repaired kernels are checked against the
source-backed publication oracle. Four additional wrong-basis mutants rotate
the latent before index-key projection and must fail index-cache validation.
Compile/runtime exceptions never count as successful negative controls.

The previous whole-cache comparison could dilute a wrong64-lane rotary tail
among mostly untouched physical pages. A shared paged-cache comparator retains
that full-cache gate and also applies the same tolerance/outlier fraction to
the unique published physical rows. All four source defaults apply it to
compressed and index caches. CPU control demonstrates that the old whole-cache
gate admits the diluted failure while the strengthened check rejects it; exact
positive values pass. This strengthens coverage without relaxing thresholds.

**F04 remains open.** F03 consumes the source tables supplied by the existing
ABI. It does not fix compressed-address versus absolute C2 group-origin rotary
coordinates, including odd-start continuation. Existing default coordinate
fixtures are not proof of F04 semantics. The independent F03 controls supply
their own nonidentity rows to isolate basis/publication behavior.

## F08: select and suppress one tied expert

Worker `/root/main_native/f08_router` exclusively repaired moe_gate.py.
The router selects the smallest tied expert ID using a max reduction of reversed
indices, gathers exactly that expert's original score, and suppresses only that
ID for the next slot. Biased sqrt-softplus selection and normalization of the
selected unbiased scores with route scale1.5 remain unchanged. No route may be
dropped, duplicated or out of range.

Worker report/source-backed CPU oracle ground this in original rc2 routing
semantics. The actual frozen old kernel emits invalid73536 on zero/equal ties;
partial ties produce `[9,10,1,35,73481,73536]`. Three numerical before-controls
reject these. Repaired randomT16/zeroT1/equalT7/partialT17 defaults pass, plus
three repeated explicit tie checks. These seven positive executions include
repeated fixtures, not seven distinct designs. Original-source CPU and comparator
controls admit legal alternative tie orders and reject invalid/duplicate IDs,
missed biased winners, wrong weights and biased-score normalization.

Main integrated the owning tie-aware comparators into per-rank staged layer
validation. It derives scores from actual FFN-normalized rows and computes
expected weights from the actual legal selected IDs. It preserves the existing
numerical weight band and permits legal ties without forcing Torch's arbitrary
tie IDs. CPU integration checks cover two legal tie orders, invalid IDs and
wrong weights. Distributed execution and inactive-row lifecycle are not proved
by these rank-local tests; F09/F10 remain open.

## F14: retain folded routed-expert payload

Worker `/root/main_native/f14_loader` exclusively repaired weight_loader.py.
The shared expert loader retains `_fold_offset`'s returned adjusted payload
before transpose/NZ packing. The same-file routed validation reference now
independently expects q-offset; scales/layout contracts are unchanged.

Actual pre-repair expert loads with representable nonzero offsets mismatch
18432 NZ bytes over nine matrices; nine zero-offset matrices are exact.
After repair,18 sampled zero/nonzero matrices,72 expert-shard matrices and
72 scales match independent payload/layout expectations; invalid fractional
and overflow folds remain rejected, both INT8 endpoints work, and the owning
validator reports zero errors. Worker and Main fresh CPU runs pass.

This is a conditional loader defect. No checkpoint tensor was read in these
synthetic checks, no full-weight load occurred, and no checkpoint was written.
Historical sampled offsets were zero. Current-checkpoint nonzero-offset
incidence remains unestablished; no current-checkpoint corruption is claimed.

## Executed checks and retained failures

All CPU/compile checks use task-submit --no-device; NPU checks use task-submit
--device auto and queue-provided TASK_DEVICE. Runners activate the selected
environment and explicitly prepend pinned PTOAS. No environment/dependency
change or simulator acceptance. Existing rev3/original-primitive and
low-amplitude/tie quantization-parity limitations persist.

| Task | Scope and result |
| --- | --- |
| task_20261106_152322_332092917320 | F08 worker device0:3 before controls,4 defaults+3 repeated tie positives, exit0 |
| task_20261106_152452_33276698135 | F08 original-source CPU/reference/comparator checks and lint, exit0 |
| task_20261106_152301_33192521558 | F14 worker final before/after CPU rerun, exit0 |
| task_20261106_153255_33554383726 | F03 original-source publication/rotary and compressor CPU equations, exit0 |
| task_20261106_153629_337320314713 | F03 device1:4 before controls,4 repaired source checks,4 rotated-index controls; exact rows in f03/controls/results.tsv |
| task_20261106_153629_337320218413 | Integrated downstream device0:17 executables,37 positive numerical cases, exit0 |
| task_20261106_153814_33863502857 | CPU cache/staged-router controls, lint and both complete forwards compile, exit0 |
| task_20261106_153922_340062716824 | Main fresh F14 before/after CPU rerun and scoped lint, exit0 |
| task_20261106_154741_346167924278 | Additional staged biased-tie/original-weight CPU controls, exit0 |
| task_20261106_154805_346379432248 | Final source/case/binding collector, model lint, runner syntax, exit0 |

Main's downstream37 positives comprise3 MLA,4 output projection,4 routing,
2 compressor,6 indexer and18 phase cases across all12 phase entries.
The additional four source-backed F03 checks produce41 Main positive device
cases. Eight F03 negative controls are separate numerical rejections. F08 worker
before-controls and CPU F14 before checks retain their own distinct evidence;
repeated worker/Main positives are not claimed as unique test designs.

Final F03 control run exited0 on device1, epochs1793950594–1793951212;
all12 executable exits0. Downstream exited0 on device0,
epochs1793950595–1793950954; all17 executable exits0. Both allocations released.
Both forward compiles exited0, epochs1793950705–1793950762.
Final CPU staged checks also cover equal biased scores with different original
scores and different legal expert subsets; biased normalization is rejected.

Retained implementation/check failures: F08 worker first used unsupported
scalar-left subtraction before correcting its selector; Main first inserted an
unnecessary raw-latent snapshot whose returned dynamic dimension conflicted
with typed source tables; removing that copy restored direct raw-compressor
consumption by both branches. Main's first CPU comparator-control call omitted
required actual/expected context dictionaries; the corrected helper passes.
All are own-code mistakes, not successful negative controls or upstream defects.
No threshold adjustment or protected-source edit was used to obtain passes.

Final source scope and per-output verdicts are in after.json/unit_summary.json.
Expected scope is nine files: four full-source phases, rope.py, paged_cache.py,
layer_reference.py, moe_gate.py and weight_loader.py. Both rotary kernel ASTs
remain identical to accepted F02; mla_proj.py/indexer.py/o_proj.py remain byte
identical. Other accepted indexer/reindex repairs are preserved.

Final45-source manifest after.json SHA256
`0bba0d6b19f91da52bf5c23e466cbf44d7edbc326e5dadeb25f3ecd43d7ad6f2`.
F08 moe_gate.py SHA256
`040019c27390f3bc5355af1b1964c19dc3ec347a6e719fe795adedc047b4f872`;
F14 weight_loader.py SHA256
`46d42683c2450ebb49947df2bd823dc3659d66cf5ba4c5ce12d0f92faf51aa9e`.
All nine changed-source hashes and every per-output verdict are in after.json
and unit_summary.json. generated_evidence.json and generated_excerpts.txt
show distinct raw-latent index projection inputs and rotated attention-latent
publication bindings in component and complete-forward compile artifacts.

## Reproduce in fresh output directories

From repository root, choose fresh absolute output directories:

```bash
task-submit --device auto --max-time 1800 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && bash experiments/dsv41-flash-w8a8-a3/logs/cohort_f03_f08_f14_1793949344/f03/run_controls.sh /absolute/fresh/f03'
task-submit --device auto --max-time 1800 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && bash experiments/dsv41-flash-w8a8-a3/logs/cohort_f03_f08_f14_1793949344/run_downstream.sh /absolute/fresh/downstream'
task-submit --device auto --max-time 600 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && bash experiments/dsv41-flash-w8a8-a3/logs/cohort_f03_f08_f14_1793949344/worker_f08/run.sh /absolute/fresh/f08'
task-submit --no-device --max-time 240 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && bash experiments/dsv41-flash-w8a8-a3/logs/cohort_f03_f08_f14_1793949344/worker_f14/run_checks.sh /absolute/fresh/f14'
```

For fresh CPU F03 oracle results set F03_EVIDENCE_DIR, then invoke
f03/check_sources.py host after activation. For cache/staged-route CPU checks
set COHORT_EVIDENCE_DIR and invoke check_integration.py. Both-forward compile
runner from the accepted F02 unit accepts a fresh absolute output directory:
`attention_f02_1793936494/run_compile.sh /absolute/fresh/compile` under the
same experiment logs parent. Worker F08 host.py writes next to its script;
copy its artifact directory for fresh independent CPU output, or inspect the
preserved source-backed CPU evidence. Never overwrite Main/worker artifacts.

## Remaining gates and recovery

This cohort does not establish original torch_npu primitive arithmetic,
original checkpoint/full-model parity, or distributed acceptance. No16-device
layer/full-model job or real-weight generation is claimed. Complete-forward
compiles show propagation/bindings; they do not establish runtime publication.

F04 absolute source coordinates is the next attention dependency. Engram
F05/F06, remaining2048-block capacity, F09/F10 routing/dynamic lifecycle,
generation F11–F13, current-checkpoint offset incidence, ordinary16-device layer
suites, standalone TP, CI ownership and composed decode pool/state diagnosis
remain OPEN. Both full-model device gates need fresh validation after remaining
semantic repairs; historical PASSes do not accept this tree. Generation stays
HELD. Known issue ledger inspected; no toolchain/known-issue status changed.

Freeze one final45-source manifest only after every writer and own job parks.
Main-owned workers/checks cannot replace independent sibling acceptance.
Observer records the acceptance decision from that sibling's evidence before
assigning another bounded unit. No commits, external filing, checkpoint/PyPTO
writes or continuation into F04 under this cohort.

STOP: final cohort source manifest frozen after all Main/worker jobs ended.
No own device allocation, background implementation writer or worker job remains.
Worker epoch records were imported exactly once under the shared lock; parallel
durations are not summed as elapsed wall time. Resume from this report,
TASKS/TIME_LOG, after.json, unit_summary.json and worker reports. Await observer's
independent sibling review/acceptance and next explicit bounded assignment.
