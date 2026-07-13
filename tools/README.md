# pypto-lib tools

## export_all_kernel_insight.py

Exports MindStudio Insight / msprof op-simulator traces for every generated
PTOAS InCore kernel in a `build_output/<case>` directory.

Use it directly on an existing build:

```bash
python tools/export_all_kernel_insight.py --build-dir build_output/Qwen3Decode_<timestamp>
```

For Qwen3-14B decode, the same export can be requested as part of the normal
case run:

```bash
python models/qwen3/14b/qwen3_14b_decode.py --max-seq --enable-l2-swimlane --enable-pmu 2 --export-kernel-insight
```

Output is written under the selected build directory as
`kernel_insight_all_funcs_<timestamp>/`. The file
`latest_all_funcs_kernel_insight_export_root.txt` points to the newest export.

## critical_path.py

Reconstruct the critical path of an L2-swimlane run and write a per-task
compute/stall report.

Run it on a `build_output/<case>` directory produced with
`--enable-l2-swimlane`:

```bash
python tools/critical_path.py build_output/_jit_l3_decode_fwd_<timestamp>
```

It auto-discovers every rank/device under the directory
(`dfx_outputs/rank*/d*/`), reconstructs two critical paths — the static CPM
latency floor (dependency-limited, unlimited cores) and the observed
as-executed path — and writes `critical_path_report.md` into the build
directory. For every task on the observed path the report lists its wall-clock
duration, the compute time it contributes, and the runtime-scheduling stall
(`data-wait` = waiting on a producer, `core-wait` = waiting for a core to free)
before it, each as a fraction of the makespan, plus a per-kernel-family summary.

Options: `--report NAME`, `--top N` (kernel-family rows), `--tol TICKS`,
`--stdout`.
