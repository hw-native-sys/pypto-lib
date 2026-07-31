---
name: incore-profiling
description: Profile generated PyPTO kernels in-core with the Ascend msprof operator simulator and export cycle-level Insight traces. Use when inspecting per-kernel timing, instruction or pipe behavior, comparing generated kernels, or producing cleaned Perfetto traces from a built case.
---

# Profile a Kernel In-Core

Read the canonical
[In-Core Simulator Profiling](../../../docs/debug-and-tune/incore-simulator-profiling.md)
guide before collecting a trace. It defines prerequisites, artifact semantics,
data-dependent caveats, troubleshooting, and reporting. Use this skill for
target resolution, safe tool invocation, workload wiring, and artifact
delivery.

## Bundled scripts

- `incore_profile.py`: discovers PTOAS functions, generates/builds standalone
  cases, invokes `msprof op simulator`, and records a manifest.
- `gen_profiling_case.py`: fallback `.pto`-driven testcase generator used by
  the profiler when no PTOAS source checkout is supplied.

Keep both scripts in this skill directory.

## Safety

- Prefer `--build-dir` over rebuilding a user case.
- Run `--list-funcs` before collection and profile only the relevant function
  when possible.
- Treat `build_output/` as generated local state; never commit traces.
- Do not modify source kernels merely to make a synthetic trace busy unless
  the user explicitly asks for a source change.
- Pass `--no-auto-msopprof` on the first real preflight. Auto-provisioning or
  `--msopprof` copies files into the selected CANN installation. Obtain user
  approval before any such external write.
- Do not mix a worker and injection library from unrelated CANN versions
  without explicitly reporting the fallback and validating the collection.
- Never interpret a near-empty or wrong-pipe trace as a fast kernel.

## Resolve the input

Accept either:

- an existing `build_output/<case>/` directory; or
- a case script plus its normal arguments when no suitable build exists.

Resolve the requested target explicitly (`a2a3` or `a5`). If the user supplies
neither a target nor evidence from the build, infer it from the case command
only when unambiguous; otherwise ask.

For an existing build, list functions without touching the toolchain:

```bash
python .claude/skills/incore-profiling/incore_profile.py \
  --build-dir build_output/<case> \
  --target <a2a3-or-a5> \
  --list-funcs \
  --no-auto-msopprof
```

Confirm that each selected `.cpp` has a sibling `.pto`.

## Collect

Run one function:

```bash
python .claude/skills/incore-profiling/incore_profile.py \
  --build-dir build_output/<case> \
  --target <a2a3-or-a5> \
  --func <function> \
  --no-auto-msopprof
```

Profile all discovered functions only when requested by omitting `--func`.
Use `--cann-set-env`, `--soc-version`, `--aicore-arch`, or
`--pto-isa-root` when auto-discovery selects the wrong installation or SoC.

To build a case first:

```bash
python .claude/skills/incore-profiling/incore_profile.py \
  --case <case.py> \
  --target <a2a3-or-a5> \
  --build-output-root <repo-root>/build_output \
  --no-auto-msopprof \
  -- <case-arguments>
```

Use `--task-submit` only when the user requested or authorized the repository's
internal task queue workflow.

If the preflight reports a missing worker:

1. Prefer a complete matching CANN installation and rerun with its
   `set_env.sh`.
2. If copying a local worker is necessary, identify the exact source and
   destination and request approval.
3. After approval, use `--msopprof <path>` or the auto-provision option and
   verify that the companion injection library came from the same package.

## Validate the collection

Inspect the generated `manifest_export.csv` and `summary.txt`. A successful
export is not sufficient evidence that the intended workload executed.

Check:

- total cycles are plausible;
- CUBE cycles are nonzero for matmul kernels;
- VECTOR cycles are nonzero for vector work;
- mixed kernels contain both expected pipe classes; and
- the instruction CSV is not only scalar/synchronization work.

If control tensors or scalar tail arguments collapsed the workload:

1. Work only in the generated standalone case under the profiling output.
2. Locate controlling tensors and scalars from the `.pto`, signature, and
   generated orchestration `add_scalar` calls.
3. Replace the relevant `vN.bin` inputs and patch generated `main.cpp`.
4. Rebuild and recollect from the standalone case directory.
5. Record every wired value in the delivered summary.

## Clean the trace

```bash
python -m pypto.tools.clean_sim_trace \
  <OPPROF_directory> \
  -o build_output/incore_<kernel>_<source>_<timestamp>
```

Keep `raw_simulator/`, `instr_metrics.json`, and the cleaned JSON together.
Rename `trace.clean.json` to `<kernel>.clean.json` so multiple downloads remain
distinguishable. Add a `summary.txt` containing source, target, workload,
toolchain selection, and per-pipe totals.

## Reporting

Return:

- source case/build and selected generated function;
- target, CANN environment, camodel SoC, and any fallback;
- manifest status and final artifact directory;
- exact control inputs and scalar arguments used;
- total and per-pipeline cycles;
- evidence that the intended work executed; and
- limitations of the single-core synthetic case plus the device measurement
  needed to confirm the conclusion.
