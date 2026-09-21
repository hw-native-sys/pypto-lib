---
name: test-with-golden
description: Speed up iterative kernel testing by generating the golden reference ONCE (save_data=True) and replaying it on every later run via golden_data, skipping input generation and the torch golden recompute each iteration. Adds a --save-data flag to the kernel when it lacks one. Use for performance/timing tuning where the kernel's numerical result is unchanged between runs; NOT recommended for precision debugging.
---

# Test with a Frozen Golden

Read [Save and Replay Golden Data](../../../docs/run-and-validate/save-and-replay.md)
before editing or running the target. That page is the canonical API,
snapshot-layout, invalidation, and CLI-wiring reference.

## Workflow

1. Confirm that replay is appropriate. Use it for performance or timing work
   whose specs, inputs, golden logic, and intended numerics remain unchanged.
   Do not use it for a precision investigation that needs fixture, seed,
   dtype, rounding, quantization, or reference changes.
2. Inspect the target's `__main__` and its `run` call. Check
   whether both `--save-data` and `--golden-data` are already exposed.
3. If either flag is missing, add only the minimal argparse option and keyword
   forwarding shown in the canonical guide. Do not change the default run:
   saving must remain opt-in and replay must default to `None`.
4. Run the target once with its normal platform/device arguments plus
   `--save-data`. Treat a failed or validation-skipped run as unusable; do not
   promote its data directory as a snapshot.
5. Resolve the snapshot from `RunResult.work_dir/data` when the caller exposes
   the result. Otherwise inspect `build_output/` and require an unambiguous
   passing run with both `in/` and `out/`; never guess among concurrent runs.
6. Verify that the snapshot contains every file required by the current specs.
   Record the snapshot path and the source revision or working-tree state that
   produced it.
7. Replay with `--golden-data <work_dir>/data`. Require both input and output
   cache-hit messages and a passing validation result.
8. Reuse that exact path for later eligible iterations. Regenerate immediately
   when any invalidation condition in the guide becomes true.

## Safety and Scope

- Keep snapshots in `build_output/` or another untracked location. Do not stage
  `.pt` files.
- Do not delete an existing snapshot unless the user asks; report stale or
  superseded paths instead.
- Do not present replay as a correctness check when validation was skipped.
- Treat `runtime_dir` as a separate optimization with its own source-compatibility
  constraint; do not enable it merely because `golden_data` is in use.
- Keep temporary CLI edits scoped to the requested kernel. Retain them only
  when they are suitable as a supported interface; otherwise revert those
  edits after the iteration with the user's work preserved.

## Report

Report why replay was eligible, any CLI edits, the capture command, snapshot
path, replay command, cache-hit evidence, validation result, and the conditions
that will require regeneration.
