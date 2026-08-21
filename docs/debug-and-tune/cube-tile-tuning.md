# Cube Tile Tuning

Use this guide to choose the row (`M`), output (`N`), and reduction (`K`) tile
for a PyPTO matmul. It complements [Performance Tuning](performance-tuning.md):
first confirm that a cube scope is important to end-to-end time, then tune its
tile against the actual compiler memory report and repeated device
measurements.

The process has three layers:

1. Rank promising shapes analytically.
2. Reject or reshape candidates at their on-chip memory constraints.
3. Measure the survivors on the target device.

The final device build and benchmark are authoritative. A formula or search
utility is a way to narrow the search, not proof that a tile is optimal.

## Describe every matmul before changing it

For each cube scope, record:

- full `M × N × K`;
- element sizes for both operands and the accumulator;
- the row tile, N fragment, and K fragment used by one matmul call;
- the number of weight operands and accumulators live at the same time;
- whether the weight is stored K-contiguously (`[N, K]`, commonly used with
  `b_trans=True`) or N-contiguously (`[K, N]`);
- the pipeline depth and whether operands remain resident across loop
  iterations; and
- the production row occupancy, not only the shape in a small test fixture.

Give unrelated matmuls separate tile constants. One shared constant silently
lets the most constrained task cap every other task and makes a sweep difficult
to interpret.

## Read the compiler memory report correctly

Every build writes:

```text
build_output/<case>/report/memory_after_AllocateMemoryAddr.txt
```

The relevant spaces are:

| Report space | Hardware role | How to use it |
|--------------|---------------|---------------|
| `Mat` | L1 operand storage | A hard, user-visible constraint for the chosen M/N/K fragment |
| `Acc` | L0C accumulator | A performance boundary; plain matmul output can often be compiler-subtiled |
| `Vec` | Vector/UB working storage | A hard constraint for vector fragments and mixed consumers |
| `Left` / `Right` | L0A/L0B cube staging | Compiler-derived information, not a DSL-level tile budget |

`Left` and `Right` may be close to 100% because PyPTO subtiles an L1 fragment
through L0A/L0B. Do not shrink a DSL tile merely to reduce those percentages.
Changing the M/N/K fragment directly changes `Mat`, `Acc`, and sometimes
`Vec`; the compiler derives L0A/L0B staging underneath it.

Treat the numbers printed by the build as ground truth. Hand estimates omit
alignment, allocator padding, liveness overlap, and backend-specific
transformations.

## Model the three practical constraints

For a K-contiguous weight and a simple double-buffered matmul, useful first
estimates are:

```text
Acc bytes ≈ M_tile × N_tile × accumulator_bytes × live_accumulators

Mat bytes ≈
    (M_tile + N_tile × live_weights)
    × K_tile × input_bytes × buffering_factor

contiguous weight bytes = K_tile × input_bytes
```

### Mat/L1 wall

Growing `N` or `K` increases the weight fragment in `Mat`; multiple live
weights multiply that cost. When a wider N fragment exceeds L1, reduce K to
buy room, reduce the number of simultaneously live weights, or reduce N.
Expect the real allocation to be larger than the simple estimate.

### Acc/L0C boundary

Growing `M` or `N` increases the accumulator fragment. For a plain
`tile.matmul` result, PyPTO can often split an oversized output through its
L0-tiling pass. That makes L0C a soft capacity boundary, but not a free one:
each extra output tile adds an exposed FIXPIPE drain and may restream
operands.

Do not assume this fallback applies to every form. `tile.matmul_acc`, a vector
or PV left operand, and a mixed on-chip consumer may have stricter constraints
depending on the compiler version. Compile the exact candidate.

### Contiguous-transfer floor

On A2/A3, an innermost transfer below the 512-byte cache-line size can move a
full line while using only part of it. For a K-contiguous weight, keep:

```text
K_tile × input_bytes >= 512 bytes
```

when feasible. A wider N tile that forces K below this floor can lose more DMA
efficiency than it gains from fewer matmul fragments. Use the target's own
cache-line value rather than carrying this constant to another architecture
without verification.

## Tune the row tile first

For grouped or MoE GEMMs, the row tile controls how often a weight is
restreamed. If a typical group spans more rows than one row tile, each extra
row tile may load the same weight again. Make the benchmark's row distribution
match production, then try to cover a typical group in one row tile without
violating `Mat` or `Acc`.

A tile selected with an unrealistically small fixture can look fast while
performing poorly at production occupancy.

## Decouple conflicting fragments

Cube and vector work can require different output fragments:

- a cube N fragment is constrained primarily by L1 and L0C;
- a vector fragment is constrained by UB and its transfer pattern.

If one shared fragment forces both tasks to use the smaller value, separate
their tile constants. When the program structure permits it, split the scopes
through a GM intermediate so each task can use its own fragment. The split
itself may be performance-neutral or add traffic; its value is that it exposes
independent tuning axes. Confirm the end-to-end result rather than assuming
decoupling is always beneficial.

## Run a funnel-shaped sweep

Sweep in serial rounds, with independent candidates inside each round run in
parallel when enough isolated devices and worktrees are available.

### Round 1: isolate axes

Start from one validated baseline. Change one of the following per candidate:

- row/occupancy tile;
- N fragment;
- K fragment; or
- task grouping or pipeline depth.

This establishes which axes affect the bottleneck and in which direction.

### Round 2: combine and refine

Carry the best two or three candidates forward. Combine compatible wins and
probe adjacent shapes. Drop axes whose effect is inside the measurement noise.

### Round 3 and later: push to the next wall

Push the remaining useful axis until the build report or a compile failure
identifies a constraint. Trade a second dimension to clear that constraint,
for example K down to make room for N. Stop when a complete round does not
improve the running best beyond the noise band.

## Measurement discipline

- Use identical, seeded inputs for every candidate.
- Revalidate numerical output for every shape; tiling must not change results.
- Measure at least three times and compare medians, not the minimum.
- Use the chip swimlane and PMU data to confirm that the expected pipe changed.
- Record compile failures as evidence: a `Mat buffer usage ... exceeds` error
  identifies L1, while a UB allocation error usually belongs to the vector
  fragment.
- Keep the simplest configuration when multiple candidates are statistically
  indistinguishable.

## Decision checklist

Before landing a tile change, confirm:

- the benchmark represents production occupancy;
- each independent task has its own tile knobs;
- `Mat`, `Acc`, and `Vec` are within the exact build's constraints;
- `Left`/`Right` percentages were not used as DSL-level targets;
- K-contiguous transfers are not accidentally below the cache-line floor;
- the end-to-end median improves outside normal noise;
- all numerical checks still pass; and
- the code comment records the relevant constraint, not a transient benchmark
  story.
