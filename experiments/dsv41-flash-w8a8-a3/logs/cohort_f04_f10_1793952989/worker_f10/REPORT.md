# F10: shared EP16 receive-lane capacity stride

The shared exchange now addresses every `(local expert, source rank)` receive
lane using the compile-time reserved capacity. Both push and compact use the
same stride. Active token counts control counts and payload lengths without
changing the common geometry. Existing receive/slab allocations and return
staging already use capacity; return destinations use the source route ID
`token*TOPK+k` and are independent of another rank's active count.

Only moe_exchange.py changed: two expressions and module capacity prose.
Public signatures, windows, signal/epoch protocol, expert kernels, route
weights and ordinary static callers are unchanged. Source frozen SHA256
`951378f40c2ab7cb4f1573aeda72548bab017ed76c545641620e6e3502b4c780`.
This worker owns no other model or shared ledger file and makes no whole-tree
stability assertion while Main repairs F04.

## Reproduction and meaning

With active=1/capacity=2, old auxiliary puts write two rows at starts0,1,2,...;
adjacent lanes overlap by one row. A source with active1 and destination with
active2 also calculate different addresses for the same `(expert,source)` lane.
geometry.py independently enumerates the overlap and verifies that capacity
stride gives disjoint lanes and matching peer addresses. This CPU proof alone
does not predict the NPU's particular overwrite schedule.

Actual16-device old-kernel execution produces a numerical routed-output
rejection:37776/163840 points exceed the unchanged .02/.02 band, maxabs1.23242.
The negative runner requires a numerical routed mismatch; compile/runtime
exceptions cannot satisfy it. This is observed corruption for this fixture,
not a claim that every possible overwrite order must corrupt all inputs.

The artifact harness calls the actual shared moe_ep16 with fixed capacity2
distributed windows and per-rank runtime slices. It does not reproduce the
exchange in a substitute implementation. All16 ranks participate; routes target
three experts each on ranks0 and1, exercising local/remote routes, skewed
destinations, zero-receive ranks and idle sources that still receive expert
work. Expected values use the existing per-expert numerical reference and
host route-list dispatch/combine; inactive rows retain a nonzero sentinel.

## Device results

task_20261106_162544_362210947,16 logical devices0–15, exit0:

| Case | Active counts | Result |
| --- | --- | --- |
| old-code control | all1,capacity2 | required numerical routed rejection |
| repaired equal | all1,capacity2 | PASS |
| repaired unequal | `[0,1,2,1]` repeated4 | PASS |
| repaired idle | eight0 followed by eight1 | PASS |

The three repaired executions are distinct count patterns. Additional
task_20261106_162849_364549219514,devices0–15,exit0, repeats all three count
patterns sequentially in ONE process with ONE compiled dynamic artifact.
Compile times2.32s/.03s/.03s and one `_jit_dynamic_moe` build root establish
reuse; runtime counts come from the active_counts tensor. Its comparator
requires exact sentinel preservation on every inactive row, then delegates
active/full numerical checking to the unchanged ratio comparator. All three
pass, including idle source ranks0/1 that still perform destination work.
These are six positive executions of three designs, not six distinct designs.
Both16-device allocations were released normally.

CPU geometry/scoped ruff task_20261106_162544_362210817571 exits0. Initial
artifact compile failed because its runtime slice extent was INT32 where
shape lowering expected INDEX; the corrected harness casts the extent to
INDEX and compile task in compile2_queue.log passes. This is a retained
worker harness mistake, not a successful negative control or upstream defect.
No model workaround was added for it.

## Reproduce without overwriting evidence

From repository root:

```bash
task-submit --device auto --device-num 16 --max-time 1500 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && bash experiments/dsv41-flash-w8a8-a3/logs/cohort_f04_f10_1793952989/worker_f10/run.sh /absolute/fresh/f10'
task-submit --device auto --device-num 16 --max-time 400 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && bash experiments/dsv41-flash-w8a8-a3/logs/cohort_f04_f10_1793952989/worker_f10/run_replay.sh /absolute/fresh/replay'
task-submit --no-device --max-time 180 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && F10_EVIDENCE_DIR=/absolute/fresh/cpu python experiments/dsv41-flash-w8a8-a3/logs/cohort_f04_f10_1793952989/worker_f10/geometry.py'
```

run.sh preserves the actual old exchange snapshot and three individual cases;
run_replay.sh adds strict inactive preservation and same-artifact replay.
The scripts activate the selected environment and pinned PTOAS, use queue IDs
from TASK_DEVICE and need all16 logical devices. No simulator or checkpoint
weights are involved. Code generation and CPU checks use --no-device.

## Limits and recovery

This proves the bounded shared exchange capacity-stride correction under
runtime equal/unequal/idle counts. It does not fix or close F09 full-forward
runtime active-row propagation, inactive cache/state isolation or general
serving lifecycle, nor establish complete-model or checkpoint equivalence.
F04 and all other Main-owned changes are outside this worker's acceptance.
Worker self-tests are not independent sibling acceptance.

No commits, external filing, checkpoint/PyPTO writes, other model edits,
threshold relaxation or shared ledger edits. All own jobs ended, devices
released, and worker parks with source snapshots, raw failure/success logs,
geometry results, generated builds and uniquely identified work record.
Main imports that record once and integrates with its final frozen cohort.
