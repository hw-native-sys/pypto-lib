# Precision Tuning

A practical guide to keeping a pypto-lib kernel numerically faithful to its
torch reference — and to diagnosing the divergence when it is not. It pairs
with [debugging.md](debugging.md) (the mechanics of `golden_data` replay and
tensor dump) and [performance-tuning.md](performance-tuning.md) (when a
precision fix and a perf fix pull in opposite directions).

The recurring lesson: most "the kernel is wrong"
mismatches are not logic bugs. They are an avoidable precision loss — a wrong
rounding mode, a dtype that silently widened or narrowed at a boundary, an
extra cast hop, or a test that was measuring the wrong thing on near-zero
output. Work through the checklist below before assuming the algorithm is
broken.

---

## 1. Pick the right `pl.cast` rounding mode

`pl.cast(x, target_type, mode=...)` takes a rounding mode that changes the
last-bit result. The accepted names (`pypto.ir.utils.CAST_MODE_NAMES`):

| mode | int | meaning |
|------|-----|---------|
| `none`  | 0 | reinterpret / no explicit rounding (Acc→Vec move, identity-width) |
| `rint`  | 1 | round to nearest, ties to **even** (RNE) |
| `round` | 2 | round to nearest, ties **away from zero** (the default) |
| `floor` | 3 | toward −∞ |
| `ceil`  | 4 | toward +∞ |
| `trunc` | 5 | toward zero |
| `odd`   | 6 | round to nearest, ties to odd |

The default is `mode="round"` (ties-away). **torch's `.to(dtype)` uses RNE
(ties-to-even).** So if you want to match a torch reference for a
float-narrowing cast, you must pass `mode="rint"` explicitly — the default
diverges from torch on exact ties.

For `fp32 → bf16` the two modes differ *only* on exact ties, so the
whole-tensor impact is tiny (measured ~1e-4 rel-L2 on dsv4 attn output). It is
still worth getting right, because the golden harness emulates the device and
you want the golden and the kernel to round the same way.

### Recommended modes vs. torch-CPU

| conversion | recommended `mode` | why |
|------------|--------------------|-----|
| `fp32 → bf16` | `rint` | matches torch `.to(torch.bfloat16)` (RNE) |
| `fp32 → fp16` | `rint` | matches torch `.to(torch.float16)` (RNE) |
| `fp32 → int8` (quant) | `rint` | round-to-nearest-even on the scaled value, matching a torch `round().to(int8)` quantizer |
| `int32 → fp16/fp32` (de-quant) | `round` | exact for in-range integers; default is fine |
| `acc → fp32` (cube/Acc move) | `none` | width-preserving move, no rounding involved |
| float → `int` for **indices** / lane math | `trunc` or `floor` | deterministic floor/truncation, never ties-away (which would jump an index) |

> When the golden deliberately reorders an algebraically-equivalent op for the
> kernel's benefit (e.g. folding a scale), align the **golden** to the
> kernel's order rather than de-optimizing the kernel — *except* when the
> golden is the ground-truth numeric (an RNE cast), in which case fix the
> kernel. See `docs/debugging.md` §2 for the replay loop.

---

## 2. Make the kernel and golden implementations identical

The golden is the device-emulating reference, so it must compute the same
thing the **same way** as the kernel — identical op order *and* identical
dtype at every step. Floating-point arithmetic is non-associative and lossy at
each narrowing, so any divergence shows up as a spurious mismatch that looks
like a kernel bug but is really a golden bug.

- **Op order.** An algebraically-equivalent reorder — folding a scale,
  changing accumulation order, reassociating adds, casting at a different
  point — changes the last bits. Keep the two in lockstep; when the kernel
  reorders an op for perf, mirror that order in the golden (see §1's note)
  rather than letting them drift.
- **Dtype at each step.** The golden must narrow and accumulate at the same
  dtype the kernel uses at each stage. A golden that stays fp32 where the
  kernel drops to bf16 (or quantizes to int8) under-reports the kernel's true
  error and hides the real bottleneck.

---

## 3. Align every input and output dtype

The single most common silent error: a kernel parameter, a golden tensor, and
the real weight disagree on dtype. When injecting a real weight, cast it to
the kernel's declared spec dtype precisely to avoid this:

```python
dt = _spec_torch_dtype(spec)            # the dtype the kernel declared
if dt is not None and v.dtype != dt:
    v = v.to(dt)                        # force the real weight to match
```

Checklist:

- The `TensorSpec.dtype` of every input must equal the dtype the kernel's
  `pl.load` / matmul actually expects. A bf16 param fed an fp32 buffer (or
  vice-versa) either mis-reads bytes or silently widens.
- Cube matmul on a2a3 honors the **stored dtype of each operand** — fp32×fp32,
  fp16×fp16, bf16×bf16, int8 are all real paths. A weight stored FP32 runs an
  fp32 matmul; one stored BF16 runs bf16. Do not assume "everything is bf16":
  match the matmul precision to the weight's stored dtype, and make the golden
  do the same.
- Output dtype must match between kernel and golden, or the comparator casts
  one side and hides a real narrowing. Compare at the dtype the kernel emits.

---

## 4. Keep intermediates wide; never cast in two hops

Activations and intermediate tensors should stay in the **highest precision
that is free** — almost always `fp32` — and only narrow at the boundary that
genuinely requires it (a bf16 matmul input, an int8 quantized buffer, the
final output).

Two rules:

1. **Prefer fp32 for intermediates.** Accumulate, normalize, apply RoPE, and
   do residual adds in fp32. Narrow to bf16/int8 only at the op that consumes
   the narrow type. The dsv4 layers carry `acc`/`res_row`/`y_row` in fp32 and
   cast to bf16 only on the store.

2. **Never cast through an intermediate dtype.** A direct `fp32 → int8` is
   strictly better than `fp32 → bf16 → int8`: the second hop throws away
   mantissa bits *before* quantization, so the int8 result is rounded off a
   value that is already wrong. Quantize from the widest source you have.
   Likewise `fp32 → int8` for quant, not `fp32 → fp16 → int8`.

   ```python
   # BAD: loses fp32 mantissa before quantizing
   x_bf16 = pl.cast(x_fp32, pl.BF16, mode="rint")
   x_i8   = pl.cast(pl.mul(x_bf16, inv_scale), pl.INT8, mode="rint")

   # GOOD: quantize straight from fp32
   x_i8   = pl.cast(pl.mul(x_fp32, inv_scale), pl.INT8, mode="rint")
   ```

   The same applies to the int8→float dequant path: go straight from the
   integer to fp32, not via fp16.

---

## 5. Choose the quantization scheme deliberately

When a quantized stage is the precision bottleneck, the scheme is a knob, not
a constant. Things to vary and measure:

- **Granularity** — per-tensor vs. per-channel / per-token scales. A single
  per-tensor scale loses badly when one channel has a much larger dynamic
  range; per-channel (or per-token activation) scales recover it. The dsv4
  weights ship per-channel `weight_scale`; match that granularity in the
  kernel and golden.
- **Symmetric vs. asymmetric** — symmetric (offset 0) is cheaper and is what
  the dsv4 W8A8 checkpoints use (`weight_offset == 0`). Only reach for a
  zero-point if the data is genuinely one-sided.
- **Scale source** — compute the activation scale from the *actual* dynamic
  range of the activation, not a static constant. Real-weight activations on
  dsv4 q-proj have ~5× the dynamic range of random fixtures, which is why the
  quant error there is ~0.8% rel-L2 with real weights vs ~0.03% with random.
- **Rounding** — RNE (`mode="rint"`) on the scaled value to match a torch
  `round()` quantizer (see §1).

Measure each variant with the error-distribution report (§6) rather than a
single pass/fail — the schemes differ in *where* the error lands, not just
whether it passes.

---

## 6. Sweep threshold levels to see the error shape

A single tolerance hides the shape of the error. Use the `error_distribution`
comparator from the golden harness to see the precision distribution instead:

```python
from golden import error_distribution

run_jit(
    ...,
    compare_fn={
        "x_next": error_distribution(),    # measure, never fails
    },
)
```

It always passes (it is a *measurement*, not a gate) and prints, for the named
output: whole-tensor **rel-L2** and **cosine** (the trustworthy verdict for
quantized / low-magnitude tensors), a **`frac>thd` table** showing what
fraction of points exceed each threshold (so you can pick the tolerance the
output actually needs), and **percentiles** of the relative diff, absolute
diff, and golden magnitude (the magnitude row tells you whether a large
relative diff is just an output pressed low near zero). Custom levels:
`error_distribution(diff_thds=(1e-3, 1e-2), quantiles=(0.5, 0.99, 1.0))`.

Once the shape is known, pick a real gate (`ratio_allclose` or
`ratio_reldiff`) at the threshold and bad-point budget the distribution
justifies.

---

## 7. Localize the offending code with tensor dump + error-distribution

`error_distribution` (§6) tells you *how much* and *what shape* the output is
off, but not *which stage*. Combine it with tensor dump to pin the exact op —
to go from "the whole kernel is 0.4% off" to "the q-proj dequant is the
source" (full dump procedure in [debugging.md](debugging.md) §2 and §5):

1. Pin the inputs with `golden_data=<dir>` so every re-run sees identical
   tensors.
2. Tag the suspect intermediates with `pl.dump_tag(t)` and run with
   `enable_dump_tensor=1` (partial dump — keep it to a few tags to avoid the
   full-dump AICPU timeouts).
3. For each dumped intermediate, run `error_distribution` against its torch
   reference and walk them in dependency order. The **first** stage whose
   distribution blows up is the culprit; everything downstream just inherits
   its error.

This narrows a `norm → matmul → dequant → activation` chain to the single op
where the error first appears, which is where one of §1–§5 was violated.

---

## 8. Test with real weights and matched data distribution

Random fixtures hide precision bugs that only real weights expose, because the
error is data-dependent:

- **Dynamic range** drives quant error. Real activations on dsv4 q-proj have
  ~5× the dynamic range of `torch.randn` fixtures, pushing q-proj quant error
  from ~0.03% (random) to ~0.8% (real) rel-L2 — a ~24× difference that random
  testing would never surface. A standalone module can look perfect on random
  data and still be the dominant error term in the full model.
- **Seed your fixtures.** Bare `torch.randn` is unseeded and flaky run-to-run;
  the strict cache tolerances (`max_error_ratio=0.0`) need a *seeded* normal
  so the distribution is reproducible. The prefill fixtures use a seeded
  normal for exactly this.
- **Inject real weights** by extracting the converted checkpoint per layer,
  casting each tensor to its spec dtype (§3), and overriding the input via
  `TensorSpec.init_value`. Run with real weights before trusting any precision
  number.
- **Measure by rel-L2 / cosine, not per-element reldiff**, on real-weight runs
  — near-zero entries make per-element relative diff meaningless (§6).
