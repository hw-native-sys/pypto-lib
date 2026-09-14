# Operator performance tracking

The `dsv4-operators` suite provides ten fixed A2/A3 workloads for an external
performance bot. It does not schedule itself or update a baseline automatically.
Its manifest, device checks, result validation and dry-run use only the Python
standard library. Running a model additionally requires a compatible PyPTO,
runtime, PTO ISA, PTOAS, CANN and Torch environment.

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

Copy [the device profile example](../../tools/perf/suites/even_devices.example.json)
to deployment configuration and replace `device_epoch` with an actual host and
topology version. Keep a profile fixed across both toolchains and all retries.
The example uses device 4 for single-device cases, devices `0,2,4,6` for TP4,
and `0,2,4,6,8,10,12,14` for eight-device cases. MTP LM-head's TP groups are the
first four and last four devices of that ordered list.

Inspect the plan without importing PyPTO, Torch or any device runtime:

```bash
python tools/perf/run_operator_suite.py \
  --device-profile tools/perf/suites/even_devices.example.json --dry-run
```

The bot should acquire the exact eight-device list once for a variant and run
the cases serially inside it. Single-device and TP4 cases use their fixed
subsets; the original `TASK_DEVICE` allocation remains intact. With the example
profile, the submission prefix is:

```text
task-submit --device 0,2,4,6,8,10,12,14
```

Append the deployment's queue timeout and `--run` command. Inside that task,
activate the chosen environment, set `PYPTO_ROOT` and `PTO_ISA_ROOT`, and invoke:

```bash
python tools/perf/run_operator_suite.py \
  --device-profile "$PERF_DEVICE_PROFILE" \
  --output-dir "$PERF_RUN_DIR" --variant candidate --date 2026-09-14
```

`PERF_DEVICE_PROFILE` must point to the deployment copy, and `PERF_RUN_DIR` must
be a new directory. Do not combine an explicit list with `--device-num`: that
option belongs to automatic allocation. An auto pool may select different
cards or exclude the even devices entirely. Exact requests wait for those cards
and remain subject to queue locking and concurrency limits.

The runner verifies `TASKQUEUE_INSIDE`, the complete ordered `TASK_DEVICE`, each
case's subset, and the model's actual `RunConfig.device_ids`. It rejects odd,
duplicate or reordered IDs and nonempty Ascend visible-device masks. This first
version supports unmasked host device IDs only. It does not infer container
remapping or pretend trace PIDs are device IDs. The report records the ordered
host device nodes, host name, kernel version and device epoch; `npu-smi` inventory
is saved separately for inspection. A physical topology change requires a new
epoch. The queue lock remains necessary; passing `-d` alone provides no exclusion.

## Measurement and evidence

Each case is a fresh process with Python/NumPy/Torch seed 1807, five discarded
warmup rounds and 100 measured rounds in one benchmark loop. The model retains
its own golden, comparison functions and tolerances. The launcher observes its
single `golden.run` call, checks the workload and devices, hashes the actual
input and golden tensors, and writes the returned `RunResult.bench` dispatch
grid. It does not patch generated kernels or replace numerical references.

The `dsv4-operators-whole-dispatch-v1` metric sums **all** operator dispatch
Effective times in each rank's round, computes each rank's 100-round median,
then reports the minimum rank median. Required communication and signal cleanup
remain included. Per-dispatch samples, all rank medians, the maximum rank median
and rank spread are retained. This is an operator tuning metric, not end-to-end
model step latency. Trace PID keys identify runtime streams; their ordering
does not establish a PID-to-device mapping.

This is a new baseline contract. Do not splice it into historical mean-time,
EP2/TP2, different-batch, or compute-only series that excluded a separate cleanup
dispatch. In particular, the earlier EPLB robot's compute-slot figures need a
contract review before comparison. See the [performance guide](../debug-and-tune/performance-tuning.md).

Zero/nonfinite samples, wrong round or rank counts, repeated invocation IDs,
unstable callable slots, flattened timing, missing golden fingerprints and
failed correctness invalidate a metric. A single failed case retains the other
valid rows. A recognized device fault or case timeout stops the allocation and
marks remaining cases unrun. The bot must not retry a still-running queue task
or start another variant on a faulted allocation.

Every attempt has an immutable run ID and a new output directory. Results are
checkpointed into `suite-result.json` and `report.md`; each attempted case keeps
its log and `raw-result.json`. A complete pass requires all ten cases. `--model`
or `--case` is useful for investigation but produces an incomplete official
suite and a nonzero exit code.

`--baseline` selects an earlier result file explicitly. `--week-baseline` must
be exactly seven logical dates earlier. Reports include the selected baseline
run ID; dates default to Beijing time unless the scheduler supplies `--date`.
Positive percentages mean slower. Missing or incompatible evidence is N/A,
never zero or a carried-forward measurement. Historical comparisons require
identical toolchain, workload contract, device identity and actual fixture and
golden hashes; lib SHA may change. The `comparison(..., mode="toolchain")`
helper instead requires identical lib SHA for same-day Control/Candidate A/B.

Source checkouts must be clean. Preflight checks the PyPTO runtime gitlink,
runtime-owned ISA pin, recorded build ISA, selected PTOAS version, loaded module
locations and native library hashes. CANN, driver and Python/Torch/NumPy versions
also belong to the comparison identity. The deployment must additionally check
its native-build provenance against the selected runtime source before enqueueing;
an ISA build stamp alone cannot prove which runtime source produced a binary.

## Storage and rollout

The default uses an in-memory golden and hashes rather than full tensor
snapshots. Each case builds under its private working directory. Successful
builds beneath that case's `build_output` are removed after the report is
written; logs and raw samples remain. Failed builds are retained for diagnosis.
`--keep-builds` preserves successful builds too, and `--save-data` explicitly
retains full input/reference snapshots and their builds. Use these options only
with a deployment retention policy and sufficient disk space. No cache directory
outside this invocation's private output tree is deleted.

The external bot owns scheduling, toolchain activation, exact queue submission,
native-build provenance, baseline selection, result retention and notification.
It should run Control and Candidate serially on the same profile and retain both
run IDs. A failed setup, timeout or missing result must remain visible in its
notification rather than reusing yesterday's success.

CPU validation is available through `python -m pytest tests/perf`. Before this
suite replaces Daily CI performance, complete device correctness and 5+100-round
acceptance for every case in both environments, inspect the dispatch boundaries,
and obtain complete reports on two consecutive nightly runs. No measured
baseline or device acceptance is supplied by adding this tooling. Daily CI's
correctness and performance collection remain active until that migration gate
has been satisfied.
