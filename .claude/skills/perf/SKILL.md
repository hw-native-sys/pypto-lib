---
name: perf
description: Run one pypto-lib case on an Ascend device and collect a benchmark with raw samples or an optional chip swimlane. Defaults to benchmarking V4-MTP CSA decode. Uses task-submit when available, otherwise Python directly. Use for running a performance case, capturing its swimlane, or measuring its device latency.
---

# Run a Performance Case

Run the selected case to completion, confirm its validation result, and return
the requested artifact or timing. A task ID alone is not the result.

## Invocation and defaults

```text
$perf [platform] [swimlane[=N]] [file.py] [case arguments...]

$perf
$perf a3 swimlane deepseek_v4_flash_mtp/decode_csa.py --start-pos 8192
$perf a5 swimlane=4 models/deepseek_v4_flash_mtp/decode_csa.py
$perf a3 models/example.py --batch-size 8 -d 0
$perf a5
```

Read the arguments in this order: platform, capture mode, target file, then
the target's own arguments. Accept `a3` and `a2a3` as aliases for the case's
`-p a2a3`; `a5` maps to `-p a5`. Bare `swimlane` means level 1;
`swimlane=N` selects an explicit level from 1 through 4. With no mode, run
the benchmark with raw samples and no swimlane capture.

| Setting | Default when omitted |
| --- | --- |
| Case | `models/deepseek_v4_flash_mtp/decode_csa.py` |
| Mode | Benchmark with raw samples (`PYPTO_BENCH=1`) |
| Platform | The configured device family: `a2a3` or `a5` |
| Devices | The case's required count, allocated by `task-submit` when available |
| Raw sampling | 100 rounds, 5 warmup rounds |

Once the filename is consumed, preserve all remaining arguments and their
order as literal case arguments; do not interpret their values as skill modes
or rewrite option names. An optional `--` separator before case arguments is
also accepted. Without a file, options beginning with `-` apply to the default
case. Do not forward skill-only tokens (`a3`, `swimlane=4`) to Python.
Reconcile explicit platform, device, and capture options in the case arguments
with the selected mode; do not emit duplicate conflicting flags. Ask only
when paths, arguments, platform, or device ownership cannot be resolved.

Read [Benchmarking rules](../../rules/benchmarking.md) and the relevant
[performance guide](../../../docs/debug-and-tune/performance-tuning.md)
sections for measurement and capture semantics.

## Resolve the case and environment

1. Find the repository root with `git rev-parse --show-toplevel`. Resolve an
   explicit path first, then a unique full-path suffix, then a unique basename
   from `rg --files`. Do not select arbitrarily among duplicate filenames.
2. Inspect the entry's parser, early `sys.argv` reads, `# ci:` markers, and
   platform-specific implementations. Check `--help` in the active environment.
   Parser acceptance alone is not proof of platform support. The default CSA
   entry takes one device, `-p a2a3|a5`, and `--enable-chip-swimlane N`.
3. Honor an explicit platform. Otherwise identify the device family from the
   machine's configured pool or `npu-smi info`; never infer it from CPU
   architecture. A simulator cannot provide device performance numbers.
4. Determine the device count from the requested EP/TP/world size and the
   target's distributed configuration. Use `# ci: devices=N` for its default
   case, not as a replacement for explicit world-size arguments. A supplied
   device list must match the requested workload.
5. Use the active Python environment and selected PyPTO checkout. Follow
   [setup-env](../setup-env/SKILL.md) when dependencies are missing. Verify
   PTOAS against that checkout's `toolchain/versions.env`; check the actual
   executable version, not the directory name. Do not assume a host-specific
   install path or select the newest release independently of the pin.
6. Run from the repository root with it on `PYTHONPATH`. For repeated eligible
   runs, follow [Save and Replay](../../../docs/run-and-validate/save-and-replay.md).

## Select the capture mode

**Benchmark (default):** set
`PYPTO_BENCH=1`, with the selected rounds and warmup explicitly
in `PYPTO_BENCH_ROUNDS` / `PYPTO_BENCH_WARMUP`. Keep raw samples in the log;
return summary statistics to the user. Do not enable swimlane capture unless
the user requests both. Report any changed sample counts and compare only
matching configurations.

**Swimlane:** explicitly set `PYPTO_BENCH=0` so
inherited benchmark settings do not repeat or perturb the capture. Add the
entry's swimlane flag with level 1, or the explicitly requested level. Keep
the case's numerical validation enabled.

Inspect the flag shape: an integer option needs a value; `nargs="?", const=1`
accepts a bare flag or `1`. A legacy `store_true` flag maps to runtime level 4
and cannot express level 1. Report that limitation and actual capture level;
do not pass an integer to a boolean flag. If the entry has no capture option,
report the missing interface rather than silently running without a trace.

## Execute

Check `command -v task-submit`. If present, use the queue; otherwise execute
Python directly. A queue error is not a reason to bypass its device locks.

### With task-submit

- Read `task-submit --help` and `task-submit --devices status` for this host.
  Use `--device auto --device-num N`, or an explicit device list when supplied
  by the user. Preserve the pool's whitelist, blacklist, and fixed sequences.
- Pass the allocated device list through the target's device argument using
  the job's `TASK_DEVICE`. Quote it so expansion happens inside the job, not
  in the submitting shell. Do not reserve one card for a multi-card case.
- Preserve the selected Python, CANN, PyPTO, and pinned `PTOAS_ROOT` environment
  in the job. In particular, an inherited `PTOAS_ROOT` takes precedence over
  `task-submit --ptoas`.

Benchmark submission shape for the default case (substitute the resolved paths and
platform; shell-quote each path and literal argument):

```bash
task-submit --device auto --device-num 1 --timeout 3600 --max-time 1800 \
  'cd <repo-root> && PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYPTO_BENCH=1 PYPTO_BENCH_ROUNDS=100 PYPTO_BENCH_WARMUP=5 <python> models/deepseek_v4_flash_mtp/decode_csa.py -p <platform> -d "$TASK_DEVICE"'
```

Submit asynchronously, without `--run` or `--wait`. Report the task ID as a
progress update, then use `task-submit --status <id>` at reasonable intervals
(typically 15–30 seconds) and `task-submit --log <id>` to collect the result.
Keep waiting while queued; do not cancel and resubmit to chase idle cards.
The execution timeout is in seconds; raise it for a known long case rather
than silently reducing benchmark rounds. On failure, collect diagnostics and
report the failure; do not retry unchanged runs automatically.

### Without task-submit

Run the case with Python in the active environment. Use the user's device
list, an existing session allocation, or the host's assigned `TASK_DEVICE`.
On a dedicated host, the entry's default device is usable when ownership is
established. On a shared host, low utilization alone does not establish
ownership; obtain the assigned IDs instead of claiming an arbitrary card.

For the default benchmark, after resolving `PERF_PLATFORM` and `PERF_DEVICES`:

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYPTO_BENCH=1 \
  PYPTO_BENCH_ROUNDS=100 PYPTO_BENCH_WARMUP=5 \
  python models/deepseek_v4_flash_mtp/decode_csa.py \
  -p "$PERF_PLATFORM" -d "$PERF_DEVICES"
```

Capture stdout and stderr in a unique log under `build_output/`. Preserve the
Python exit status when using `tee` (`set -o pipefail` in Bash). If the execution
tool returns a running session, keep monitoring that session to completion;
do not relaunch the case to retrieve its output. Shell-quote target arguments
as data in both execution routes.

## Collect and report

For either route, require successful execution and the entry's actual golden
comparison or forward-validation summary. Report validation-skipped runs as
such; they do not establish correctness. On failure, return the error and log
path instead of presenting partial data as a successful result.

Locate the work directory from this run's log. If it is absent, require an
unambiguous matching artifact created by this run; do not pick the newest
directory across concurrent jobs. Include the resolved case, platform, device
list, command, validation result, and log path in the result.

### Swimlane result

Return a clickable link to `dfx_outputs/chip_swimlane_records.json` and the
actual capture level. For L3 output, files live under `rankN/dK/`:

```bash
python .claude/skills/perf/scripts/fast_card.py <dfx_outputs-dir>
```

The helper sums dispatch spans for each rank and returns the fastest rank's
record files. Report its selection and spans, preserving every dispatch for
that rank. A capture span is diagnostic, not a benchmark median. Map the rank
to a device only with evidence from the run; otherwise label it unknown.
Keep matching `deps.json` and `name_map*.json` alongside the records for the
PyPTO3 Toolkit VSCode extension. Merged Perfetto traces remain secondary.

### Benchmark result

Save the complete log from either execution route and summarize it with:

```bash
python .claude/skills/perf/scripts/summarize_bench.py \
  --devices <device-csv-in-verified-rank-order> <run-log>
```

If rank-to-device mapping is unknown, pass `--devices unknown` and retain
runtime rank IDs. Keep separate benchmark blocks separate. Report rounds,
warmup, min/max/median/mean/sample-standard-deviation and the headline:
single-card median, or lowest per-rank median for multiple cards. For multiple
dispatches per round, use the harness's per-rank summary of round totals for
the headline; the raw table is per dispatch. Do not claim a round latency
from flattened or incomplete samples. Retain persistent rank imbalance in
the report. Return text and tables, not a chart.
