# Optimization Rules

Normative performance rules for PyPTO-Lib kernels. Follow these when writing or
modifying a kernel unless a measurement in the specific case says otherwise (and
if so, record that measurement in a comment). Pairs with `docs/performance-tuning.md`
(how to measure) and `docs/pypto-coding-style.md` (how to structure kernels).

## Rule 1 — Cube matmul K-reductions: single-pipeline `if k==0` form

**Always express a cube (matmul) K-reduction as ONE `pl.pipeline` that starts at the
first chunk, branching `if k == 0: pl.matmul(...)` else `pl.matmul_acc(...)`. Never
peel the first `pl.matmul` outside the pipeline and loop the rest.**

### Canonical form (do this)

```python
acc = pl.create_tensor([M_TILE, N_TILE], dtype=OUT_DTYPE)   # pre-declared pipeline carry
for k0 in pl.pipeline(0, K, K_TILE, stage=2):               # start at 0, step by the K tile
    a_k = a[..., k0 : k0 + K_TILE]
    b_k = b[k0 : k0 + K_TILE, ...]
    if k0 == 0:
        acc = pl.matmul(a_k, b_k, out_dtype=OUT_DTYPE)      # iter 0: fresh matmul
    else:
        acc = pl.matmul_acc(acc, a_k, b_k)                  # rest: accumulate
out[...] = acc                                              # (or pl.assemble)
```

### Anti-pattern (do NOT do this)

```python
acc = pl.matmul(a[..., 0:K_TILE], b[0:K_TILE, ...], out_dtype=OUT_DTYPE)  # peeled OUTSIDE
for kb in pl.pipeline(1, K // K_TILE, stage=2):             # pipeline starts at 1
    k0 = kb * K_TILE
    acc = pl.matmul_acc(acc, a[..., k0:k0+K_TILE], b[k0:k0+K_TILE, ...])
out[...] = acc
```

### Why

The peeled first `pl.matmul` runs *before* the `stage=2` pipeline's software-prefetch
spins up, so it executes serially with no load/compute overlap. Folding chunk 0 into
the single pipeline lets the prologue prefetch chunk 1's operands while chunk 0
computes — an MTE2-bound cube then stays busy from the first iteration. The benefit
scales with the peeled fraction (1 / num_K_chunks) and with how MTE2-bound the cube is.

Measured on DeepSeek-V4 CSA decode (a2a3, 6 runs/form, aggregate cube exec):
- `qproj_matmul` (INT8, 4-chunk K): **−14%** (548 → 472 µs), disjoint ranges.
- `proj_b_mm` (INT8, 4-chunk K): **−15.7%** (722 → 609 µs), disjoint ranges.
- `proj_a_mm` (BF16, 16-chunk K): within noise (peel is only 1/16) — still written in
  this form for consistency; it is never slower.

### Caveats

1. **Iteration 0 must be a real `pl.matmul`, never `pl.matmul_acc` from a zero carry.**
   Seeding a zero accumulator and `matmul_acc`-ing every chunk trips the TLOAD DN→NZ
   assertion (pypto#1540). The `if k == 0: pl.matmul` branch is exactly what avoids it,
   so this form is both faster *and* the safe way to keep a single pipeline.
2. **Pre-declare the carry with the matmul's true output shape/rank.** A grouped GEMM
   with a 3-D weight (`w[g:g+1, N, K]`) yields a 3-D accumulator (e.g.
   `[1, M_TILE, N_TILE]`), not 2-D — the compiler will reject a mismatched
   `pl.create_tensor`, so read the error and match the rank.
3. The loop variable may be the K *offset* (`pl.pipeline(0, K, K_TILE)`, check `k0 == 0`)
   or the chunk *index* (`pl.pipeline(0, K // K_TILE)`, check `kb == 0`) — pick whichever
   makes the slice math read cleanly; both are the same form.

### Reference implementations

- `models/deepseek/v4/qkv_proj_rope.py` — `qproj_matmul`
- `models/deepseek/v4/decode_sparse_attn.py` — `proj_a_mm`, `proj_b_mm`

## Rule 2 — L2-traversing tile transfers: keep the innermost dim above the backend granularity

**For every `tile.load` / `tile.store` whose tile lives in an L2-traversing memory space
(Vec/UB, GM-facing), size the innermost (contiguous) axis so that
`shape[-1] * sizeof(dtype)` is at least the backend's recommended innermost-dim bytes.
A sub-granularity innermost dim under-fills each DMA burst / L2 cache line and wastes
GM bandwidth.**

### Thresholds (`GetRecommendedInnermostDimBytes()`)

| Backend | Platform flag | Recommended innermost | GM access granularity | L2 cache line |
| ------- | ------------- | --------------------- | --------------------- | ------------- |
| Ascend910B/C | `a2a3` | **512 B** | 512 B | 512 B |
| Ascend950 | `a5` | **128 B** | 128 B | 512 B |

512 B is the stricter (910B) floor; 128 B is the a5 floor. When a kernel targets both,
size to 512 B and it satisfies a5 too. Examples of the innermost-byte figure: `fp32[16]`
= 64 B (under both floors), `fp16[128]` = 256 B (ok on a5, short on a2a3), `bf16[256]`
= 512 B (ok on both).

### Why

The recommendation models an L2-cache-line / GM-access-granularity concern: the innermost
axis is the contiguous burst of a DMA transfer, so an innermost dim below the cache
line / access granularity means each burst under-fills a line and pays a partial-line
penalty on GM traffic. Widening the innermost axis (or restructuring the slice so the
contiguous run is longer) makes each transfer a full-line burst.

### The check surfaces this automatically (PH001)

The compiler runs `TileInnermostDimGranularity` (hint code **PH001**) at end-of-pipeline,
on by default. Under `compile()` / `pl.jit` (which install a report instrument) the full
hits land in `${output_dir}/perf_hints.log` and stderr shows a one-line
`[perf_hint] N hints across M sites; see …` summary; without a report instrument each
hint prints in full to stderr. Each hint echoes the `(dtype[innermost], target_memory)`
tuple and the evaluated byte size, e.g.:

```text
[perf_hint PH001] TileInnermostDimGranularity: tile.load has innermost dim = 64B
(tile fp32[16], target_memory=Vec); recommended >= 128B for backend a5
(L2 cache line = 512B). Consider increasing tile shape on the innermost axis.
```

Read `perf_hints.log` after a build; treat a PH001 hit as a prompt to widen the axis,
not noise to mute. Suppress the whole check
(`disabled.insert(DiagnosticCheck.TileInnermostDimGranularity)`) only when a measurement
justifies the narrow tile — and record that measurement in a comment, per the rule at the
top of this file.

### Caveats / exceptions

1. **Cube-private L0/L1 tiles are exempt and the check already skips them.** Tiles whose
   `target_memory` is `Mat` / `Left` / `Right` / `Acc` never traverse L2, so a small
   innermost dim there is fine — do not widen a cube operand tile to satisfy this rule.
   The hint only fires on L2-traversing (`Vec`/UB, GM-facing) transfers.
2. **The hint's source span is the post-pipeline IR location, not your DSL line** (the
   inner-dim constant is not named — pypto#1305). Map it back by the echoed
   `(dtype[innermost], target_memory)` tuple and byte size rather than trusting the
   `<string>:line:col` to point at the originating `pl.at` / slice.
3. **A narrow innermost dim is sometimes unavoidable** (e.g. an inherently small last
   axis you cannot fuse or transpose). When so, the penalty is real but may be dominated
   by other costs — measure before contorting the layout, and if you keep it, note why.

### Reference

- pypto diagnostics pass doc `docs/en/dev/passes/92-diagnostics.md` (PH001) — thresholds
  come from `BackendHandler::GetRecommendedInnermostDimBytes()`.
