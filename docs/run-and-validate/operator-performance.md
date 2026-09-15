# Operator performance tracking

The `dsv4-operators` suite measures twenty fixed A2/A3 workloads using the environment
installed by CI's `setup-ci-job` action.

The CI implementation lives in `.github/scripts/`: `ci_operator_perf.py` handles
the queue and GitHub artifacts, `run_operator_suite.py` executes and reports the
fixed suite, and `run_operator_case.py` captures each model's golden and timing
result. `operator_perf_cases.json` defines all twenty workloads and their devices.

Each model has five operators at each of two scenario contexts: 8K (8192)
and 128K (131072) historical tokens, for 20 measured entries in total.

| Case | Parallelism | Fixed workload at each context |
| --- | --- | --- |
| MTP CSA, HCA, SWA | TP1 | B=4, S=2, T=8, start position 8192 or 131072 |
| MTP MoE | EP8 | 16 experts/rank, 8 tokens/rank, layer 0, balanced routes |
| MTP LM-head | TP1, DP1 | 8 active logit rows, projection |
| DSpark CSA, HCA, SWA | TP1 | B=16, S=8, T=128, start position 8192 or 131072 |
| DSpark MoE | EP8 | 16 experts/rank, 128 tokens/rank, layer 0, balanced routes |
| DSpark LM-head | TP1, DP1 | 128 active logit rows, projection |

The per-device workloads stay fixed across contexts. DSpark keeps B16/T128 on
one card; it does not move the former TP4 group's B64/T512 onto that card.
MTP Attention uses its existing single-device entry. TP1 does not change MoE's
EP8 expert parallelism. The MTP MoE fixture uses the EPLB expert placement
convention; neither model's balanced fixture measures the load-balancing
algorithm itself.

MoE and LM-head do not consume historical KV context. Their `scenario_context`
field labels the model scenario only; their inputs and token counts are identical
between the 8K and 128K entries. Each model's MoE and LM-head run once: the
128K row explicitly references its 8K row through `measurement_case`. The report
has 20 scenario rows from 16 executions: 14 single-device workloads and two EP8
workloads. Shared rows carry the same measurement ID, fingerprints, timings and
failure status; they are not independent samples. Sharing is rejected if any
execution setting differs after removing only the scenario label. Attention
passes the context as its real `--start-pos` input and always runs separately.
Case IDs end in `-8k` or `-128k`; history compares the same scenario and contract.

LM-head includes hidden dispatch, projection, logits assembly and signal cleanup,
but excludes sampling and the preceding model layers. MTP TP1 projects eight
active rows using a zero-padded 16-row matmul; the padded rows are never published
to another rank or returned as logits.

DSpark's Attention CLI uses the length of `--start-pos` to select the local
batch. The manifest passes sixteen copies of the selected context, not one
scalar. It checks the resulting `[128, 4, 4096]` output specification before
compilation. Ring settings are explicit: MTP uses task window/dependency pool
262144 and heap 2 GiB; DSpark uses 16384 and 1 GiB respectively.

## Fixed even devices

The [suite configuration](../../.github/scripts/operator_perf_cases.json) specifies
single-device `4` and eight-device `0,2,4,6,8,10,12,14` workloads.
Attention and LM-head use device 4; MoE uses all eight devices.
The embedded `device_profile` defaults to epoch `a2a3-even-v1`. A CI-generated
`--device-profile` can supply the host name and topology epoch.

Every unique measurement acquires and releases exactly its required cards:

```text
# Attention and LM-head
task-submit --device 4
# MoE EP8
task-submit --device 0,2,4,6,8,10,12,14
```

The suite submits these requests serially. A single-device case does not reserve
seven unused devices. Exact requests must not use `--device-num` or `auto`.

On the intended A3 host, dies are paired as `(0,1)`, `(2,3)`, and so on; each
pair communicates over SIO, with the even die attached directly to PCIe.
Fixed even IDs preserve that path and avoid mixing topology-dependent timing.
Local tuning has observed differences of roughly 20-25% between die selections;
this is workload- and host-specific, not a guaranteed penalty on every system.
Verify the mapping when provisioning another host. The
[CANN IPC documentation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/appdevgapi/aclcppdevg_03_2013.html)
describes SIO and HCCS links between dies. The other die in a package can still
contend for shared resources, so device-ID locking alone does not isolate a
benchmark from another tenant. CI uses the shared NPU pool; shared load can
affect measurements even while the selected device IDs are reserved.

The worker checks `TASKQUEUE_INSIDE`, an exact `TASK_DEVICE` match, and the model's
actual `RunConfig.device_ids`. Odd, duplicate or reordered IDs and nonempty Ascend
visible-device masks are rejected. Host IDs must be unmasked. The profile must
name the expected host; a missing or mismatched hostname fails before submission.
Host name, character-device major/minor IDs, kernel version, device epoch and
`npu-smi` inventory are recorded. Trace PID ordering does not prove a mapping to
device IDs. Change the epoch when hardware or topology changes.

Inspect the workloads without importing PyPTO, Torch or a device runtime:

```bash
python -S .github/scripts/run_operator_suite.py --dry-run
```

Outside any task allocation, source CI's `activate.sh` and run the suite with a
profile that includes the expected `hostname`:

```bash
python .github/scripts/run_operator_suite.py \
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

Every unique measurement starts a fresh process with Python/NumPy/Torch seed 1807, five warmups
and 100 measured rounds in one benchmark loop. The model's golden, comparison
functions and tolerances are preserved. The observer hashes actual input and
golden tensors and records the public `RunResult.bench` dispatch grid.

The `dsv4-operators-whole-dispatch-v2` metric sums **all** operator dispatch
Effective times per rank and round, computes each rank's median, and reports the
minimum rank median. Communication and signal cleanup are included. Raw samples,
per-dispatch timings, the maximum rank median and rank spread remain available.
This follows the [performance guide](../debug-and-tune/performance-tuning.md)'s
operator tuning convention; it is not end-to-end model step latency.

The report prints warmup/round counts, whether golden was replayed, per-rank
median spread, model wall time and shared-measurement references. `wall_seconds`
covers the model child (compilation, fixture/golden generation, validation and
benchmarking); `submission_wall_seconds` also includes queue wait, worker startup
and recovery. Shared wall times must not be summed twice. Golden is computed live;
replay is disabled and explicitly reported as `golden_replayed=false`.

This establishes a new baseline. The former ten-case suite has different case
IDs and is not accepted as history. The previous independently sampled twenty-row
contract also differs from v2 and cannot supply comparable deltas.
Historical Daily CI means, default EP2/TP2
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
passing rows remain visible. A correctness failure does not stop later workloads.

A wall-clock timeout is recorded as `timeout`, separately from a recognized
`device_fault` such as 507034 or AICORE_EXCEPTION. The worker terminates the whole
model process group, including children surviving their leader. Before releasing
the allocation after a timeout it requires two `npu-smi` observations showing
all selected physical IDs healthy and idle. A verified recovery permits the next
case in a new allocation. A device fault, failed cleanup, unknown recovery state
or unresolved queue completion stops further submissions without an automatic
retry or device reset. An unrecognized inventory format cannot establish recovery.
Already completed measurements remain visible; an incomplete suite exits nonzero.

Results are checkpointed into `suite-result.json` and `report.md`. Each unique
measurement retains `measurement.json`, raw results, model/queue logs and any
recovery observations. Queue return codes and errors are stored independently of
measurement status. A queue error after a completed, validated measurement fails
the overall execution without removing that measurement's timings.

The wrapper persists its actual suite exit code and failure phase in
`execution.json` before publishing. Finalization uses this evidence, not a
fabricated queue code derived from a GitHub step outcome. Publishing or wrapper
failures retain passing rows, fingerprints and metrics; interrupted rows remain
explicitly incomplete. A later artifact upload failure remains an Actions step
failure and must not relabel successful measurements as numerical failures.

Runs hash inputs and golden in memory without saving tensor snapshots. CI uploads
reports, JSON and bounded logs, then removes the invocation's private build
directory after a successful upload, including failed builds. Model weights and
build trees are excluded from history.

## Daily CI integration

[Daily CI](../../.github/workflows/daily_ci.yml) runs the operator suite in its
`operator-performance` job. It uses `setup-ci-job` with the bundle toolchain,
an isolated `operator-checkout` and a distinct build namespace. Per-case queue
payloads source that checkout's `activate.sh` and forward its resolved `PYPTO_SRC`,
`PTOAS_ROOT` and `PTO_ISA_COMMIT`. No separate toolchain install is introduced.

The job uses the existing `self-hosted`, `linux`, `arm64`, `npu` runner pool.
Before device setup, it records the assigned runner's hostname in
`OPERATOR_PERF_HOST`; the suite and workers verify that same hostname throughout
the run. An optional repository variable `OPERATOR_PERF_HOST` adds a pre-setup
check against an explicitly expected host. It does not control runner scheduling.
No new runner label or repository variable is required for the default path.
Set `OPERATOR_PERF_DEVICE_EPOCH` when the host topology changes (default
`a2a3-even-v1`). Recorded host or device identity changes make historical
comparisons unavailable. Device reservations are per case; the host remains
shared, so investigate contention before interpreting performance changes.

Scheduled runs and manual runs use the full Daily CI suite by default. To validate
only these performance cases, select `performance_only` in the manual dispatch,
or run the workflow on the desired branch with:

```bash
gh workflow run daily_ci.yml --ref <branch> -f performance_only=true
```

This skips correctness sweeps, E2E and their combined summary. The performance
job still publishes its own summary and artifact.

The Actions console prints setup/history phases and a progress snapshot every
30 seconds during execution. Snapshots show completed unique measurements, the
active case and devices, whether its worker has written a checkpoint, and the
last available model stage. A missing worker checkpoint can mean queue wait or
worker startup; a heartbeat alone does not establish device progress. Full logs
and measurement evidence remain in the final artifact.

Limits are provisional safety bounds, not device measurements:

| Repository variable | Default seconds | Meaning |
| --- | --- | --- |
| `OPERATOR_PERF_CASE_TIMEOUT` | 1800 | Model child wall time per unique measurement |
| `OPERATOR_PERF_QUEUE_TIMEOUT` | 3600 | Client completion wait, including queueing and execution |
| `OPERATOR_PERF_SUITE_TIMEOUT` | 10800 | Entire suite, including queue waits |

Each queue task receives the case limit plus 90 seconds for worker overhead,
process cleanup and recovery observations. `task-submit --timeout` includes both
queueing and execution; it is not an acquisition-only timeout. The completion
wait must exceed the case limit plus 90 seconds and is capped by the remaining
suite budget, reserving 30 seconds for client exit. If the client disconnects
before worker completion is known, stop submissions and let the workflow clean
up this job's queued tasks. The suite stops submitting when another full case
and its cleanup cannot fit; remaining rows carry an explicit budget reason.
Calibrate these settings using first-run wall times, especially cold compilation
and 128K fixture generation. The job keeps a separate 300-minute ceiling for
setup, execution, upload and cleanup; adjust that ceiling if increasing the
suite budget. Concurrent performance jobs wait without cancelling one another.

[The CI wrapper](../../.github/scripts/ci_operator_perf.py) reads the workflow's
creation date and retrieves main-branch Daily CI artifacts with an Actions read
token. That token is removed before invoking the device queue. The previous
baseline is the latest attempt from the most recent earlier logical date within
14 days; the weekly baseline is exactly seven days earlier. Same-day runs, other
branches and other workflows are excluded. The latest attempt is selected even
when it failed or is still running; an older success never replaces it. Missing,
expired, malformed or mismatched artifacts produce N/A with a reason. A history
lookup failure after resolving the current workflow date still allows today's
measurement to run without comparisons.

Each attempt uploads `operator-perf-<attempt>`, containing `suite-result.json`,
`report.md`, history selection, submission metadata and bounded case logs/raw
results. The result identity includes repository, workflow run ID, attempt and
lib SHA, which must match the historical artifact's owning run. No shared-disk
history or writable baseline branch is required. GitHub's artifact retention
policy applies, so keep artifacts for at least eight days for weekly reporting.
The existing upload action retries transient upload failures once.

The performance job publishes its own summary and artifact. The correctness
summary has no dependency on this job, so a slow performance run cannot delay
the correctness report. After queue cleanup, finalization preserves interrupted
checkpoints and publishes the final performance summary before artifact upload.
Full correctness sweeps disable benchmarking and retain status tables; A5
correctness and E2E remain covered, while this performance suite covers A2/A3 only.

The CI unit-test job installs NumPy and PyYAML alongside pytest and CPU Torch. Run the
performance and reporting checks with:

```bash
python -m pytest tests/test_operator_suite.py tests/test_ci_operator_perf.py \
  tests/test_daily_ci_summary.py
```

Before merging the replacement, validate MTP LM-head TP1 and MTP MoE EP8 with
16 experts/rank on the intended CI stack; the default TP2/EP2 correctness sweep
does not cover these specializations. Then complete the 16 unique measurements
at 5+100 rounds, verify all 20 report rows and obtain two complete nightly reports.
CPU tests and IR lowering do not constitute this device acceptance. No measured
baseline or device acceptance is supplied by this implementation.
