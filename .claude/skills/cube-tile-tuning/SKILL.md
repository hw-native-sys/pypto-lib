---
name: cube-tile-tuning
description: Tune cube/matmul row, N, and K tiles for a PyPTO kernel using analytic candidates, compiler buffer evidence, and repeated device sweeps. Use when optimizing cube throughput, resolving Mat/L1, Acc/L0C, or Vec/UB constraints, sizing grouped-GEMM occupancy, or trading one tile dimension for another.
---

# Tune Cube Tiles

Read the canonical
[Cube Tile Tuning](../../../docs/debug-and-tune/cube-tile-tuning.md) guide and
[Performance Tuning](../../../docs/performance-tuning.md) before changing a
kernel. Keep the technical method in those documents; use this skill to
orchestrate one concrete tuning run.

## Scope

- Tune only the kernel or model file requested by the user.
- Preserve numerical behavior and the existing harness.
- Keep the bundled scripts in this skill directory; they are internal
  candidate-ranking aids, not public APIs.
- Treat target-device compile output and repeated timing as authoritative.

## Bundled scripts

Run both from the repository root:

```bash
python .claude/skills/cube-tile-tuning/hint_l1_tile.py \
  --M <full-M> --N <full-N> --K <full-K> \
  --bytes-a <bytes> --bytes-b <bytes> --bytes-c <acc-bytes> \
  --b-trans --platform <a2a3-or-a5>

python .claude/skills/cube-tile-tuning/tile_budget.py \
  --M <row-tile> --N <n-fragment> --K <k-fragment> \
  --bytes-in <bytes> --bytes-out <acc-bytes> \
  --weights <live-weights> --accum <live-accumulators> \
  --platform <a2a3-or-a5>
```

Use `--b-trans` for the usual K-contiguous LLM weight layout. Also run the
hint tool with `--no-wave-considered` when total DMA and device-wave ranking
disagree. Read `--help` before using less common flags.

The hint script ranks directions; it does not know the exact compiler
allocation or the currently bound pipe. The budget script is an estimate.

## Workflow

1. **Resolve the target.**
   - Read the requested kernel, its golden function, and its existing tile
     constants.
   - Follow `docs/pypto-coding-style.md` for any kernel edit.
   - Identify all matmuls in the affected scope and give distinct tasks
     distinct knobs before sweeping.

2. **Capture a validated baseline.**
   - Run the existing correctness command.
   - Capture a benchmark and L2/PMU evidence appropriate to the suspected
     bottleneck.
   - Record the exact command, input/occupancy, platform, device, and raw
     timing samples.
   - Reuse fixed golden data only when the numerical computation is unchanged.

3. **Map constraints.**
   - Record full and per-call M/N/K, element sizes, live weights and
     accumulators, layout, and pipeline depth.
   - Inspect
     `build_output/<case>/report/memory_after_AllocateMemoryAddr.txt`.
   - Tune against `Mat`, `Acc`, and `Vec`. Never treat `Left`/`Right`
     (compiler-managed L0A/L0B staging) as independent DSL budgets.
   - Run both bundled scripts to rank and screen initial candidates.

4. **Plan a funnel.**
   - Round 1 changes one independent axis per candidate.
   - Round 2 combines compatible winners and probes neighbours.
   - Round 3+ pushes the useful axis to a reported constraint, then trades a
     second dimension if needed.
   - Keep a baseline and the previous winner in every round.

5. **Fan out safely.**
   - Run independent candidates concurrently only when isolated worktrees and
     devices are available.
   - Never patch the same shared file concurrently.
   - Seed every harness identically and match production row occupancy.
   - Keep rounds serial because one round's evidence selects the next.

6. **Classify every candidate.**
   - Numerical mismatch: reject and investigate; tile changes must preserve
     results.
   - `Mat buffer usage ... exceeds`: trade K down or shrink N/M.
   - Vec/UB allocation failure: change the vector fragment, not the cube
     fragment blindly.
   - Acc/L0C growth: inspect whether compiler output tiling and FIXPIPE drains
     erase the expected gain.
   - Infrastructure or device failure: rerun or mark invalid, never rank it.

7. **Select and verify.**
   - Use at least three repeated timings and compare medians.
   - Require a gain outside the observed noise band.
   - Prefer the simplest candidate among statistically equivalent results.
   - Rerun correctness and the repository's relevant lint/tests after the
     final edit.

## Reporting

Return:

- baseline command, median, raw samples, and correctness result;
- one table per round with tile values, compile status, median, and diagnosis;
- the winning M/N/K values and the constraint they sit under;
- memory-report evidence for `Mat`, `Acc`, and `Vec`;
- any candidates rejected for correctness or infrastructure reasons; and
- final validation commands and results.

Do not report an analytic ranking as a measured performance result.
