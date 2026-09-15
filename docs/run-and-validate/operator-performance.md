# Operator performance tracking

The `dsv4-operators` suite measures ten fixed A2/A3 workloads using the environment
installed by CI's `setup-ci-job` action.

| Case | Parallelism | Fixed workload |
| --- | --- | --- |
| MTP CSA, HCA, SWA | Single device | B=4, S=2, T=8, start position 8192 |
| MTP MoE | EP8 | 16 experts/rank, 8 tokens/rank, layer 0, balanced routes |
| MTP LM-head | TP4, DP2 | 8 active logit rows/rank, projection |
| DSpark CSA, HCA, SWA | TP4 | B=16/rank, S=8, T=128/rank, start position 256 |
| DSpark MoE | EP8 | 16 experts/rank, 128 tokens/rank, layer 0, balanced routes |
| DSpark LM-head | TP4, DP1 | 128 active logit rows/rank, projection |

The MTP MoE fixture uses the EPLB expert placement convention. This does not
measure the load-balancing algorithm. DSpark's standalone decode MoE exposes EP,
not expert-internal TP: its EP8 case uses the token scale of the TP4 model.
It must not be described as an EP8-by-TP4 expert implementation. MTP Attention
retains the existing single-device workload; EPLB is not an Attention switch.
LM-head includes hidden dispatch, projection, logits assembly and signal cleanup,
but excludes sampling and the preceding model layers.

DSpark's Attention CLI uses the length of `--start-pos` to select the local
batch. The manifest passes sixteen copies of `256`, not one scalar or sixty-four
values. It checks the resulting `[4, 128, 4, 4096]` output specification before
compilation. Ring settings are explicit: MTP uses task window/dependency pool
262144 and heap 2 GiB; DSpark uses 16384 and 1 GiB respectively.

## Fixed even devices

The [suite configuration](../../tools/perf/suites/dsv4_operators.json) specifies
single-device `4`, TP4 `0,2,4,6`, and eight-device `0,2,4,6,8,10,12,14` workloads.
MTP LM-head uses the first four and last four devices as its two TP4 groups.
The embedded `device_profile` defaults to epoch `a2a3-even-v1`. A CI-generated
`--device-profile` can supply the host name and topology epoch.

The CI job acquires the complete ordered eight-device allocation once, then
runs the ten cases serially on their fixed subsets. Use this submission prefix:

```text
task-submit --device 0,2,4,6,8,10,12,14
```

Append the queue timeout and `--run` command. Do not combine this explicit list
with `--device-num` or use `auto`: automatic pools may exclude the even devices.
Exact requests wait for those devices through the shared queue lock.

The runner verifies `TASKQUEUE_INSIDE`, the complete `TASK_DEVICE`, the selected
subsets and the model's actual `RunConfig.device_ids`. It rejects odd, duplicate
or reordered IDs and nonempty Ascend visible-device masks. Host IDs must be
unmasked. It records host name, character-device major/minor IDs, kernel version
and device epoch, and saves `npu-smi` inventory. Trace PID ordering does not prove
a mapping to device IDs. Change the epoch when hardware or topology changes.

Inspect the workloads without importing PyPTO, Torch or a device runtime:

```bash
python -S tools/perf/run_operator_suite.py --dry-run
```

Inside the acquired allocation, source CI's `activate.sh`, forward its resolved
`PYPTO_SRC`, `PTOAS_ROOT` and `PTO_ISA_COMMIT`, and run:

```bash
python tools/perf/run_operator_suite.py \
  --device-profile "$PERF_DEVICE_PROFILE" --output-dir "$PERF_RUN_DIR" \
  --date 2026-09-14
```

Use a new output directory for every attempt. The activation script supplies
CANN, the bundle and the job venv.

## CI environment

The existing `setup-ci-job` action owns installation, pin resolution and import
validation. The suite records lib, PyPTO, runtime and ISA revisions, PTOAS version,
bundle, Python/Torch/NumPy versions and CANN/driver metadata hashes for historical
comparisons. It does not install packages or duplicate CI's wheel validation.

## Measurement and comparisons

Every case starts a fresh process with Python/NumPy/Torch seed 1807, five warmups
and 100 measured rounds in one benchmark loop. The model's golden, comparison
functions and tolerances are preserved. The observer hashes actual input and
golden tensors and records the public `RunResult.bench` dispatch grid.

The `dsv4-operators-whole-dispatch-v1` metric sums **all** operator dispatch
Effective times per rank and round, computes each rank's median, and reports the
minimum rank median. Communication and signal cleanup are included. Raw samples,
per-dispatch timings, the maximum rank median and rank spread remain available.
This follows the [performance guide](../debug-and-tune/performance-tuning.md)'s
operator tuning convention; it is not end-to-end model step latency.

This establishes a new baseline. Historical Daily CI means, default EP2/TP2
workloads and the earlier robot's compute-slot figures are different contracts
and cannot be spliced into this series.

`--baseline` accepts an earlier run. `--week-baseline` requires exactly seven
logical dates earlier. Each selected baseline's run ID appears in the report.
CI supplies a Beijing logical date derived from workflow creation, so a queue
crossing midnight does not change the measurement's date. Positive changes mean
slower; failed, missing or incompatible measurements produce N/A.

CI history compares identical workloads, actual input/golden fingerprints and
device identity. Python/Torch/NumPy, bundle and CANN/driver changes also start a
new comparable series. Compiler, runtime, ISA, assembler and lib revisions may
change: the percentage then describes the **complete CI software stack**, with
changed components listed. It does not attribute a change to a kernel alone.

## Failure handling and storage

Failed correctness never yields a performance value. Missing ranks or rounds,
zero/nonfinite samples, duplicate invocation IDs, unstable callable slots,
flattened timing and missing fingerprints invalidate the measurement. Other
passing rows remain visible. Device faults and case timeouts stop the allocation;
remaining cases are marked unrun. An incomplete suite exits nonzero.

Results are checkpointed into `suite-result.json` and `report.md`. Each attempted
case retains `raw-result.json` and its log. A recognized fault or unresolved queue
completion must not cause an automatic retry. Every run attempts the fixed ten
cases unless a device fault or timeout stops the allocation.

Runs hash inputs and golden in memory without saving tensor snapshots.
Successful private builds are removed after reporting.
CI should upload only reports, JSON and logs, then remove the invocation's private
build directory. It must never upload model weights or build trees as history.

CPU checks are available through `python -m pytest tests/perf`. Device acceptance
of all ten cases at 5+100 rounds and complete nightly reports remain required
before replacing the existing performance collection. This suite alone does not
provide a measured baseline or install a schedule.
