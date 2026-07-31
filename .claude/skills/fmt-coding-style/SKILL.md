---
name: fmt-coding-style
description: Apply this project's kernel style conventions to a pypto kernel file — one statement per line (split ops instead of wrapping), what-not-why comments including the module docstring, slice/assemble syntax sugar, model-vs-tiling header grouping, _TILE naming with block counts inlined, no asserts, and host-level create_tensor/reshape sunk to their first consuming scope. Use when asked to fmt-coding-style / restyle / clean up the formatting of a kernel, or before committing a new kernel.
---

# Kernel Style Pass

Mechanical restyle of a pypto kernel file. **Style only — the generated kernel
must be identical.** Never retune a tile value, reorder ops, change a `pl.at` /
loop construct, or "improve" the algorithm while doing this pass. If a rule
cannot be applied without changing behavior, leave the code and say so in the
report.

## Invocation

The argument is the file to format. Apply every rule below to that file
in place — this is an edit pass, not a review; do not stop at a list of
suggestions.

```text
$fmt-coding-style models/deepseek/v4-flash/mtp_projection.py
$fmt-coding-style mtp_projection.py
$fmt-coding-style                      # no argument: the file open in the IDE
```

Resolve a bare name by unique path-suffix match, then unique basename match
(`rg --files`); ask when ambiguous. With no argument, use the IDE-opened file;
if there is none, ask which file instead of guessing. Format exactly the one
named file — do not sweep sibling kernels that share the same smells.

Reference points:

- `docs/pypto-coding-style.md` — the canonical DSL reference (loop constructs,
  `pl.at`, slice/assemble); this skill is the *formatting* layer on top of it.
- `models/deepseek/v4-flash/qkv_proj_rope.py` — reference header layout.
- `ruff.toml` — `line-length = 110`, `target-version = py310`.

## Rule 1 — One statement per line

A statement occupies exactly one line. When it does not fit in 110 columns, do
**not** wrap the arguments across lines — **split the op** into named
intermediates until each line fits.

```python
# before — one statement wrapped over four lines
hidden_sq_sum = pl.add(
    hidden_sq_sum,
    pl.reshape(pl.row_sum(pl.mul(hidden_chunk, hidden_chunk)), [1, T_TILE]),
)

# after — one op per line
hidden_sq = pl.mul(hidden_chunk, hidden_chunk)
hidden_row_sum = pl.reshape(pl.row_sum(hidden_sq), [1, T_TILE])
hidden_sq_sum = pl.add(hidden_sq_sum, hidden_row_sum)
```

**110 is a soft target.** A wrap that rescues only a few columns is worse than
the long line — when a statement cannot be split into ops (a single
`TensorSpec` / `pl.slice` / `pl.create_tensor` call), keep it whole up to about
120 columns instead of trailing one keyword onto a continuation line.

```python
# before — 96 + 15 columns split across two lines
next_hidden_spec = TensorSpec("next_pre_hc_hidden", [N_RANKS, T, HC_MULT, D], torch.float32,
                              is_output=True)

# after — 112 columns, one line
next_hidden_spec = TensorSpec("next_pre_hc_hidden", [N_RANKS, T, HC_MULT, D], torch.float32, is_output=True)
```

Past ~120, go back to Rule 1: split the op, or group the arguments as below.
`decode_fwd.py`'s per-layer `pl.slice` statements run well past 110 on purpose —
one statement, one line.

The same applies to loop headers and long slice expressions: hoist the offset
arithmetic onto its own line rather than breaking the subscript.

```python
# before
for hidden_norm_idx in pl.spmd(
    t_dim // T_TILE,
    name_hint="mtp_projection_hidden_norm",
    allow_early_resolve=True,
):

# after
for hidden_norm_idx in pl.spmd(t_dim // T_TILE, name_hint="mtp_projection_hidden_norm", allow_early_resolve=True):
```

**Call sites put several arguments per line.** A call has no ops to split, so
one argument per line is never right. A call that fits in 110 columns is a
single line:

```python
hc_head(pre_hc_hidden_out, hc_head_fn, hc_head_scale, hc_head_base, x_head)
```

A call that does not fit follows `decode_fwd.py`: the callee and `(` on their
own line, arguments indented one level and **grouped by role** — one line per
group — a trailing comma, and `)` on its own line.

```python
attention_csa(
    hidden,
    hc_attn_fn_csa, hc_attn_scale_csa, hc_attn_base_csa,
    attn_norm_w_csa, wq_a_csa, wq_b_csa, wq_b_scale_csa,
    ...
    kv_cache_csa, cmp_kv_csa, cmp_block_table,
    position_ids, kv_seq_lens,
    x_attn_csa,
)
```

Group by what the arguments *are* — the HC triple, the projection weights, the
caches and their block tables, the slot mappings, the scalars, the output —
and give the output its own line. Do **not** greedily fill each line to 110
columns; a group that is short stays short.

Grouping decides **only where the line breaks go**. Argument order is positional
and must never change; if a category is split across the signature, it stays
split. When the roles are not obvious, take the groups from the callee's
parameter list — kernel signatures are already written one role per run of
parameters.

```python
# before — 12 lines, one argument each
return mtp_projection(
    hidden_states,
    prev_hidden_states,
    enorm_w,
    ...
)

# after
return mtp_projection(
    hidden_states, prev_hidden_states,
    enorm_w, hnorm_w,
    e_proj_w, e_proj_w_scale, e_proj_smooth,
    h_proj_w, h_proj_w_scale, h_proj_smooth,
    hidden_states_out,
)
```

**This applies to the whole file**, not just the DSL body — `golden_*`,
`build_tensor_specs`, and the `__main__` argparse block get the same treatment:

```python
# before
parser.add_argument(
    "--enable-l2-swimlane",
    type=int,
    nargs="?",
    const=1,
    default=0,
    choices=(0, 1, 2, 4),
)

# after
parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2, 4))
```

Structures that stay one-item-per-line:

- the **kernel signature** — one annotated parameter per line, already the
  convention;
- the **harness call**, `run_jit(...)` / `run(...)` — leave it exactly as it is.
  One option per line is what makes it easy to toggle `compile_cfg` /
  `runtime_cfg` / `rtol` / `compare_fn` entries while testing;
- list literals whose elements are themselves full-width expressions, such as
  the `TensorSpec` return list — one element per line, because each
  `TensorSpec(...)` already fills a line, and each of those must itself be a
  single line.

**List literals of short items follow the call-site rule**: several per line,
grouped by role. The grouped `ordered_names` and `replicated_attention` in
`decode_mtp.py`, and `CSA_LAYER_STACKED_NAMES` in `decode_fwd.py`, are current
reference forms.

Element order is load-bearing here — `ordered_names` drives the spec order —
so, as with call arguments, regroup the line breaks only and never reorder.

Naming the intermediates is part of the rule — reuse the existing local naming
idiom (`<thing>_<what>`: `hidden_sq`, `prev_deq`, `q_acc`), and do not reuse a
name already live in the same `@pl.jit.inline` scope. Search the full scope
before introducing a name; duplicate live names can collide during SSA
conversion.

## Rule 2 — Comments state WHAT, not WHY

Every comment tersely names the operation. Strip rationale prose: perf
justification, before/after numbers, "so that ...", history ("the old form
re-read ..."), and cross-references to the reference implementation. Keep
formulas and shape/layout notes — they describe *what*. Keep genuine
bug-preventing warnings (hardware constraints, ordering requirements).

This includes the **module docstring**: state what the file computes, and stop
there. Drop "mirrors the official implementation" framing, and drop the layout
recap too — tensor shapes are already in the kernel signature, and a layout note
belongs next to the code it describes, not in the header.

```python
# before
"""DeepSeek-V4 MTP input projection scaffold.

Mirrors the MTP-only prolog in the official implementation:
``e_proj(enorm(hidden_states)) + h_proj(hnorm(prev_hidden_states))``.

``prev_hidden_states`` and ``hidden_states_out`` use token-major pre-``hc_head``
hidden states with HC lanes: ``[T, HC_MULT, D]``.
"""

# after
"""DeepSeek-V4 MTP input projection: e_proj(enorm(hidden_states)) + h_proj(hnorm(prev_hidden_states))."""
```

One line is the target. Go to a second line only when the file computes
something a single sentence cannot name — see `qkv_proj_rope.py`, where the
docstring also has to say the projection serves both the decode and prefill
paths.

Trailing tiling comments follow the same rule — `# qr_proj D (K) reduction
tile` is what; `# 8 slices -> deep stage=2 pipeline, double-buffered Mat fits`
is why and goes away.

**Avoid lint-suppression pragmas** — `# ruff: noqa: F401,F403,F405,F821`,
per-line `# noqa: ...`, `# type: ignore`. They are usually copied from another
kernel and suppress nothing real. Test before deciding, with `--ignore-noqa`,
which defeats the file-level `# ruff: noqa` and every per-line `# noqa` at once:

```bash
ruff check --select F --ignore-noqa <file>
```

Clean → delete the pragma. Real violations → fix the cause: drop the unused
import (F401), replace `import *` with explicit names (F403/F405, which is also
what usually produces F821). Keep a suppression only when the violation cannot
be removed without changing behavior, and then narrow it to a per-line
`# noqa: <code>` instead of a file-level blanket.

## Rule 3 — Prefer slice / assemble sugar

Use subscript syntax for loads and stores:

```python
# load
tile = hidden_states[t0 : t0 + T_TILE, k0 : k0 + D_TILE]
# store
hidden_xg[t0 : t0 + T_TILE, k0 : k0 + D_TILE] = hidden_xg_tile
```

Rewrite `x = pl.assemble(x, tile, [r0, c0])` into the subscript store, dropping
the reassignment.

Keep the explicit `pl.slice` / `pl.assemble` call **only** when it carries an
argument the sugar cannot express:

- `pl.slice(t, sizes, offsets, valid_shape=[rows, cols])` — padded load;
- `pl.assemble(dst, src, offsets, atomic=pl.AtomicType.Add)` — atomic store.

`pl.set_validshape(tile, rows, cols)` applies to an already-computed tile, so a
store of a `set_validshape` result can still use the sugar — convert it and
verify it compiles; if the compiler rejects the sugar for that store, restore
`pl.assemble` and note it.

## Rule 4 — Group the header constants

Module-level constants are grouped and labelled, in this order:

```python
# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # T = B * S

# model config
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
EPS = M.rms_norm_eps

# tiling
T_TILE = 8
QUANT_TILE = 256
```

`model config` holds everything read off the model config plus values derived
purely from it (`HC_DIM = HC_MULT * D`, `D_INV = 1.0 / D`). `tiling` holds the
on-chip partitioning knobs. Add a third labelled group only when the file has
constants that are genuinely neither (e.g. quant constants imported from
`config`).

## Rule 5 — `_TILE` naming, inlined block counts, no asserts

- Every tiling constant ends in `_TILE`. Rename `_CHUNK`, `_STEP`, `_SLICE`,
  `_GRAN` and friends: `D_CHUNK` → `D_TILE`, `QUANT_CHUNK` → `QUANT_TILE`,
  `LINEAR_OUT_CHUNK` → `LINEAR_OUT_TILE`.
- **Inline the block counts.** Constants like `D_BLOCKS = D // D_TILE` exist
  only to be a loop bound — delete them and write the expression at the loop.
  Prefer the start/stop/step loop form, which also removes the `k0 = kb * TILE`
  line:

  ```python
  # before
  D_BLOCKS = D // D_CHUNK
  ...
  for kb in pl.pipeline(D_BLOCKS, stage=1):
      k0 = kb * D_CHUNK

  # after
  for k0 in pl.pipeline(0, D, D_TILE, stage=1):
  ```

  When the loop genuinely needs the block index (fused multi-dimension `spmd`
  fan-outs like `(t_linear // LINEAR_T_TILE) * LINEAR_OUT_BLOCKS`), inline the
  division into the `pl.spmd` bound and the two decode lines; keep no named
  `_BLOCKS` constant.
- **Drop the asserts.** Module-level `assert (DECODE_BATCH * DECODE_SEQ) %
  T_TILE == 0` shape guards go away. Keep an assert only when it guards a
  constraint that is silent and expensive to debug (an L0C capacity bound, a
  hardware 16-row/16-col rule) — and then keep exactly that one, not the whole
  block.

## Rule 6 — Define where first used

Host-level definitions that sit outside the `pl.spmd` / `pl.parallel` / `pl.at`
scopes — `pl.create_tensor`, `pl.reshape`, dim reads, padded-size arithmetic —
go **immediately before the scope that first consumes them**, not in one block
at the top of the kernel body.

```python
# before — everything declared up front
hidden_flat = pl.reshape(hidden_states, [t_dim, D])
hidden_xg = pl.create_tensor([t_linear, D], dtype=pl.FP32)
hidden_i8 = pl.create_tensor([t_linear, D], dtype=pl.INT8)
hidden_scale_q = pl.create_tensor([t_linear, 1], dtype=pl.FP32)

for hidden_norm_idx in pl.spmd(...):      # uses hidden_flat, hidden_xg, hidden_scale_q
    ...
for t0 in pl.parallel(0, t_dim, T_TILE):  # first use of hidden_i8
    ...

# after — each definition sinks to its first consumer
hidden_flat = pl.reshape(hidden_states, [t_dim, D])
hidden_xg = pl.create_tensor([t_linear, D], dtype=pl.FP32)
hidden_scale_q = pl.create_tensor([t_linear, 1], dtype=pl.FP32)
for hidden_norm_idx in pl.spmd(...):
    ...

hidden_i8 = pl.create_tensor([t_linear, D], dtype=pl.INT8)
for t0 in pl.parallel(0, t_dim, T_TILE):
    ...
```

`hidden_acc_pad` / `prev_acc_pad` in `mtp_projection.py` already sit right above
their `pl.spmd` — that is the target shape for every declaration in the file.

Constraints:

- A tensor consumed by several scopes goes before the **first** one.
- **Sink only, never hoist**, and keep the same nesting level — moving a
  definition *into* a scope changes what the compiler sees. Some slices must in
  fact live inside the consuming `pl.at` (the cos/sin case), but that is a
  correctness fix, not this pass's job; leave such code alone.
- Keep the definition order within a group unchanged, so a `pl.reshape` still
  follows the dim read it depends on.
- Blank-line separate each definition group from the scope above it.

## Cross-file safety

Renaming a module-level constant can break importers. Before finishing, grep
each renamed symbol repo-wide and update every reference:

```bash
rg -n '\b(D_CHUNK|QUANT_CHUNK|D_BLOCKS)\b'
```

Also check whether the file's `__main__` / `build_tensor_specs` / golden
functions use the renamed constants — they usually do.

## Verify

Style passes touch a lot of lines, so verify mechanically:

1. `ruff check <file>` — must be clean (undefined names catch a half-finished
   rename).
2. `ruff check --select E501 --line-length 120 <file>` — advisory, not a gate.
   The repo lints only `F`, so treat each hit as a candidate for op-splitting
   or argument grouping; a statement that reads better whole may stay long.
3. Import the module so the decorators and the DSL body are executed:

   ```bash
   PYTHONPATH="$(dirname <file>):$PYTHONPATH" python -c "import <module>"
   ```

4. Behavior must be unchanged. If the file was passing before, re-run it the
   normal way — `perf-a2a3` for a device run, or `python <file> -p a2a3 ...
   --compile-only` when the target has that flag (no device, no `task-submit`).
   State clearly in the report whether this was run or skipped.

Do not "fix" a pre-existing failure inside a style pass; report it separately.

## Report

- resolved file;
- per rule, what changed (counts are enough: "9 wrapped statements split",
  "11 `pl.assemble` → subscript stores, 2 kept for `atomic=`");
- constants renamed, with old → new;
- block-count constants and asserts removed;
- anything intentionally left alone, with the reason;
- verification: ruff result, import result, and whether a device / compile-only
  rerun was done.
