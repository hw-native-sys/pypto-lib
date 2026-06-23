---
name: cube-tile-tuning
description: Tune cube/matmul tile sizes (row tile, N fragment, K fragment) for a PyPTO kernel — analytic hints, an on-chip buffer constraint model, and an empirical device sweep. Use when optimizing a matmul/cube's throughput, sizing the row / N / K tiles, resolving Mat (L1) / L0C / UB buffer overflows, or trading one tile dim for another.
---

# Cube (matmul) tile-size tuning

A repeatable method for choosing the cube tile of a PyPTO matmul scope so it runs as
fast as the bound pipe allows without overflowing an on-chip buffer. Three layers,
cheap to expensive: an **analytic** pass (rank candidates), a **constraint model**
(which buffer walls each candidate), and an **empirical** device sweep (the truth,
with run-to-run noise handled).

Pairs with `docs/performance-tuning.md` (L1/L0 tuning, swimlanes, buffer-occupancy
reports) and `docs/pypto-coding-style.md` §6 (mixed cube+vector vs. decoupled scopes).

## When to use

- A matmul/cube scope is the kernel bottleneck and you want a faster tile.
- You hit `Mat buffer usage (...) exceeds platform limit` or a UB/L0C overflow and
  need to know which dim to shrink (and which to grow to compensate).
- You're sizing the row/occupancy tile (the M tile per matmul) for a grouped/MoE GEMM.
- You decoupled a cube from its vector epilogue (AIC/AIV split) and want to grow the
  cube tile that the shared-UB form could not.

## Bundled tools

| Tool | Role |
|------|------|
| `hint_l1_tile.py` | Analytic: enumerate `(TM,TN,TK,pin)` tile candidates for one matmul, ranked by DMA. **Directional only** — it doesn't know your bound pipe or the device's exact double-buffered L1 model. |
| `tile_budget.py` | Constraint: for a *chosen* tile, report which on-chip buffer (Acc/L0C, Mat/L1) it blows or has room in, and whether the K tile clears the cache-line floor. Prints the K↔N trade to escape a Mat wall. |

Both are standalone (no PyPTO import). Run from the repo root; replace the dims/bytes
with your matmul's (M is the row tile, N and K come from the weight):

```bash
# Analytic candidates — example INT8 A&B, INT32 accumulator
python .claude/skills/cube-tile-tuning/hint_l1_tile.py --M <M> --N <N> --K <K> \
    --bytes-a 1 --bytes-b 1 --bytes-c 4              # rank by sequential DMA (wave-aware)
python .claude/skills/cube-tile-tuning/hint_l1_tile.py --M <M> --N <N> --K <K> \
    --bytes-a 1 --bytes-b 1 --bytes-c 4 --no-wave-considered   # rank by total DMA only

# Constraint check for a chosen tile — --weights = weight operands held at once
# (a two-weight fused scope = 2, a single-weight scope = 1; --accum defaults to it)
python .claude/skills/cube-tile-tuning/tile_budget.py --M <M> --N <N> --K <K> --weights <n>
```

`tile_budget.py` defaults to 910B (a2a3). For a5/950 pass `--platform a5` (L0C is
256 KB there). Override budgets with `--mat-bytes` / `--acc-bytes`.

## The method

1. **Map every cube matmul.** For each AIC scope write down `M × N × K`, the elem
   bytes, how many *weights* it holds at once (a fused two-weight scope = 2, a single
   weight = 1), and how many accumulators are live. M is usually the row/occupancy
   tile; N and K come from the weight.

2. **Analytic pass — `hint_l1_tile.py`.** Run per matmul in the default wave-considered
   mode and again with `--no-wave-considered`. Read it for *direction* (does it want
   bigger N? pin A or B?), not as a target. It ignores your bound pipe and the exact L1
   model, so it over-pushes TN and can suggest a sub-cache-line TK.

3. **Read the memory report.** After any build, open
   `build_output/<case>/report/memory_after_AllocateMemoryAddr.txt`. Per function it
   shows Mat (L1), Left/Right (L0A/L0B), Acc (L0C), Vec (UB) used vs. limit. An
   under-used Acc means you have M/N room; a near-full Right/Mat means the weight tile
   is the wall.

4. **The constraint model (the three walls).** Use `tile_budget.py`, or by hand:

   | wall | usage | budget (910B) | grows with |
   |------|-------|--------------|------------|
   | **Acc** (L0C) | `M·N·bytes_out·n_accum` | 128 KB | M, N |
   | **Mat** (L1) | `(N·n_weights + M)·K·bytes_in·dbuf` | 512 KB | **N, K, n_weights** |
   | **cache-line** | `K·bytes_in ≥ 512 B` | — | (a floor, not a budget) |

   Mat is almost always the wall when you grow N, because it scales with `N·K·weights`
   and is double-buffered. The Mat number is an estimate; the device compile is ground
   truth (it adds ~10-25% alignment) — `tile_budget.py` flags MARGINAL within 15%.

5. **Occupancy / row tile first.** The M tile that controls *weight re-streaming* is
   the dominant lever for grouped/MoE GEMM: size it so a typical group is one row-tile
   (row tile ≥ rows/group). Below that, every extra row-tile re-streams the whole
   weight — usually the single biggest win.

6. **Give each task its own tile knobs — decouple wherever you can.** Never share one
   tiling constant across tasks that have different bottlenecks or shapes: a shared
   knob lets the task that needs the *smallest* tile cap the one that could use the
   *largest*. Expose a separate row/N/K tile **per matmul / per scope** so each is an
   independent axis, free to size to its own wall. The most common instance: a cube
   sharing its N-fragment with its vector epilogue is pinned to whatever fits UB —
   split them into separate scopes meeting through a GM intermediate (coding-style §6;
   follow an existing decoupled-scope kernel for the idiom) so the cube N-frag sizes
   to L0C/L1 and the vector N-frag to UB, independently. The same applies between two
   cube tasks with different shapes (e.g. distinct matmuls): give them distinct tile
   constants, not one reused name. *The decouple itself is often ~perf-neutral; its
   value is freeing each task's tile — and that is also what makes the per-task sweep
   tractable, since independent knobs are independent axes.*

7. **Grow N; when Mat-walled, trade K down.** A bigger cube N-frag cuts the
   matmul-setup count (wins for scalar/address-gen-bound cubes) and A reloads. When N
   hits the Mat wall, **shrink the K tile to buy the room** — but keep
   `K·bytes_in ≥ 512 B` (the cache-line floor; B is K-contiguous under `b_trans`, so a
   short K wastes the DMA line). A wider N that forces a sub-cache-line K can be net
   negative. `tile_budget.py` prints the exact `N≤… / K≤…` trade.

8. **Empirical sweep — the truth.** Profile candidates with
   `python <kernel>.py -p a2a3 -d <dev> --enable-l2-swimlane`; the headline is
   **"Total Test Time"**. Run it as **parallel rounds, never one-by-one**, funnelling
   the top-N of each round into the next — see **Sweep recipe** below.

9. **Land the simplest winner.** Within the noise band, take the fewest-knob config.
   Record *why each tile is what it is* — the wall it sits under — in the constant's
   comment, so the next person doesn't re-derive it.

## Sweep recipe

Two hard rules, then the per-run hygiene.

### 1. Fan out — never sweep serially

Each variant is an independent compile + device run (minutes each); running them
one-by-one wastes hours. **Dispatch the whole round at once** — one variant per NPU
device (`-d 0..7`) via parallel agents or a `Workflow` (`parallel`/`pipeline`), each
in its own git worktree (`isolation: 'worktree'`) so concurrent patches of the same
kernel file don't collide. N variants then cost ~one variant's wall-clock, not N.
Each agent: copy a base kernel in, patch the constants, build, run
`--enable-l2-swimlane` ×3, return `{label, ttt, pass, fatal}`.

### 2. Iterate in rounds — top-N seeds the next round (≥3 rounds)

One sweep never finds the optimum; the axes interact (a wider N only pays once K is
traded; a tile only helps at the right occupancy; an unlocked axis exposes a new
wall). Run a funnel, **at least 3 rounds, often 4-5**:

1. **Round 1 — isolate each axis.** One knob per variant off a common base (row tile,
   N-frag, K-frag, spmd grouping, …) plus the references. Learn which axes move the
   number, and in which direction.
2. **Round 2 — refine around the top-N.** Take the best ~2-3 configs, stack their
   wins, probe the neighbours (next N up, the paired K trade). Drop the dead axes.
3. **Round 3+ — push the unlocked axis.** A Round-2 win usually exposes a fresh wall
   (growing N hits Mat); the next round trades a second dim to clear it. **Keep
   spawning rounds while one still beats the running best by more than the noise band;
   stop the first round that doesn't.**

Carry each round's winner forward as the next round's **base** (snapshot it), so a
round refines a known-good config instead of re-deriving from scratch. Decide the next
round's variants *from the last round's table* — this is the one inherently sequential
part (rounds are serial; the variants **within** a round are always parallel).

### 3. Per-run hygiene

- **Seed the input harness** so every variant sees identical data — inject
  `torch.manual_seed(...)` right after `parse_args()`, before the tensor-spec builder.
  Differences then come from tiling, not data.
- **Best-of-3, read the median**, not the min (the min is the luckiest run); noise is
  ~±5%. A "win" inside the noise band isn't one.
- **Read compile failures as constraint signals**, not dead ends:
  `Function '...': Mat buffer usage (X) exceeds platform limit` → L1 wall, trade K
  down or shrink N; a UB `AllocateMemoryAddr` failure → the *vector* frag, not the
  cube. Confirm the diagnosis with `tile_budget.py`.
- **Keep the harness occupancy honest.** If the deploy target is N rows/group, the
  harness must generate ~N rows, or the winning tile won't match production (a tile
  tuned at low occupancy can look great on the bench and lose in production).

## Shape of a tuning run (round structure)

Abstract template — the rounds are serial, the variants in each are parallel:

| round | variants (one knob each, fanned across devices) | what you learn / do |
|------|------|------|
| 1 — isolate | row/occupancy tile; N-frag; K-frag; spmd grouping; + references | the dominant lever and each axis's direction; prune the dead ones |
| 2 — refine | stack the top-2–3 wins; probe their neighbours | confirm the stack; surface the next constraint |
| 3 — push | drive the most-promising axis to its limit | usually hits a buffer wall (Mat/Acc/UB) — *the failure tells you which* |
| 4 — trade | swap a second dim (e.g. K↓ to fit a wider N) to clear that wall | land the winner; stop when a round adds nothing past noise |

The funnel is what makes it work: a round's **failures and near-misses** decide the
next round's variants (e.g. an L1/Mat overflow when pushing N says "trade K", not
"give up"). A single broad sweep misses this.

## Pitfalls

- **The hint tool is directional.** It over-pushes TN and can suggest TK below the
  cache-line floor. Cross-check every candidate with `tile_budget.py`.
- **`Mat` overflow ≠ UB overflow.** Growing a *cube* N-frag overflows L1/Mat (it holds
  the weights), not UB. Decoupling the vector frag does not help here — trade K.
- **A shared tile constant couples unrelated tasks.** If two tasks reuse one tiling
  name, the tighter-constrained one silently caps the other. Split the knob per task
  (step 6) *before* sweeping, or the sweep can't reach the better config at all.
- **Cache-line floor is real.** `K·bytes_in < 512 B` quietly wastes MTE2 DMA; a bigger
  N that needs a sub-line K can be net-negative.
- **Bound-pipe matters.** Weight/MTE2-bound cubes barely move on tile (the weight-byte
  floor is fixed); scalar/address-gen-bound cubes (lots of tiny matmuls) win most from
  fewer, larger tiles. Check the PMU/swimlane before sweeping blindly.
- **Don't trust one run, or the min of a few.** Median of ≥3, seeded harness.
- **Match the harness to the target occupancy**, or you tune for the wrong workload.
