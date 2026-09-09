# Golden and Run

Every runnable kernel file carries its own validation: the kernel, the specs
that describe its arguments, a Torch reference, and a `__main__` that runs the
three together and exits non-zero on a mismatch. This page is how those parts
are written; [Golden Harness](../run-and-validate/golden-harness.md) is the
reference for what the harness does with them.

## File layout

Four parts, in this order:

```python
@pl.jit
def rms_norm_test(x, norm_w, x_normed: pl.Out[...]):     # 1. the kernel
    ...


def golden_rms_norm(x, norm_w):                          # 2. the reference
    ...


def build_tensor_specs(B, S):                            # 3. the arguments
    ...


if __name__ == "__main__":                               # 4. the CLI
    ...
```

Import `torch` and `golden` **inside** `build_tensor_specs` and `__main__`, not
at module level: the kernel half of the file is imported by other kernels and by
compile-only paths that must not pay for Torch.

---

## Specs

One spec per kernel parameter, in the **same order with the same names** as the
signature. A spec never declares a direction — the harness stamps `In` / `Out` /
`InOut` from the compiled artifact, so the kernel signature is the only place
direction is written.

```python
def build_tensor_specs(B, S):
    import torch
    from golden import TensorSpec

    T = B * S

    def init_x():
        return torch.randn(T, D) - 0.5

    def init_norm_w():
        return torch.randn(D) * 0.1 + 1.0

    return [
        TensorSpec("x", [T, D], torch.bfloat16, init_value=init_x),
        TensorSpec("norm_w", [D], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("x_normed", [T, D], torch.bfloat16),      # output: no init_value
    ]
```

`init_value` defaults to a zero-filled tensor — **random input is opt-in**.
Pass `torch.randn` for plain noise, or a no-argument closure when the
distribution matters. It usually does: a kernel that normalizes, divides by a
row sum, or quantizes against a row maximum behaves differently on `N(0, 1)`
than on a realistic activation, and a golden that passes on the wrong
distribution proves little.

Scalars use `ScalarSpec(name, dtype, value)` and are specialized into the
artifact by default; the harness page covers `compile_runtime=` and the L3
resident and stepped forms.

### Parameterize the shape

A kernel that serves both decode and prefill takes its shape from arguments
rather than module constants, so one file validates every case it claims to
support:

```python
MODES = {
    "decode":  (DECODE_BATCH // TP, DECODE_SEQ),
    "prefill": (PREFILL_BATCH, PREFILL_SEQ),
}
...
for mode_name in modes_to_run:
    B, S = MODES[mode_name]
    result = run(fn=..., specs=build_tensor_specs(B, S), ...)
```

---

## The golden function

The function passed to `golden_fn` receives one dict keyed by spec name and
fills every output **in place**. Keep the math in a plain helper the callback
calls, so the same reference is reusable from another kernel's golden:

```python
def golden_rms_norm(x, norm_w):                  # the math, reusable
    import torch

    x = x.float()
    norm_w = norm_w.float()
    inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    return (x * inv * norm_w).to(torch.bfloat16)


def golden_rms_norm_test(tensors):               # the harness callback
    tensors["x_normed"][:] = golden_rms_norm(tensors["x"], tensors["norm_w"])
```

Three rules:

- **Write the math, not the kernel.** A golden that mirrors the kernel's tiling,
  its accumulation order, or its quant scheme reproduces the kernel's bugs and
  validates nothing. Express the operation the way the model defines it.
- **Compute in FP32, cast once at the end.** A BF16 reference accumulates its
  own error and turns a tolerance into a guess. See
  [Precision Tuning](../debug-and-tune/precision-tuning.md) for the rounding
  modes that make a cast match the device.
- **Do not read an output's initial content.** The harness hands the golden
  cloned inputs and separate zero-filled pure outputs, so a runtime write can
  never corrupt the reference — but the zeros are the harness's, not a promise
  from the kernel.

### Regions the kernel does not write

A pure `pl.Out` buffer is allocator residue wherever the kernel does not write
it, so a golden that leaves that region at zero asserts a value the kernel never
promised (see
[L2 Programming](l2-programming.md#a-plout-region-the-kernel-does-not-write-is-undefined)).
Fill such a region with `float("nan")` in the golden and pass
`ignore_nan=True` to the comparator, bound the comparison with
`valid_rows=` / `valid_axis=`, or make the parameter `pl.InOut` so the host
zeros actually reach the device.

---

## The run call

```python
result = run(
    fn=rms_norm_test,
    specs=build_tensor_specs(B, S),
    golden_fn=golden_rms_norm_test,
    runtime_dir=args.runtime_dir,
    golden_data=args.golden_data,
    config=dict(
        platform=args.platform,
        device_id=args.device,
        enable_chip_swimlane=args.enable_chip_swimlane,
        dump_passes=args.dump_passes,
    ),
    rtol=5e-3,
    atol=5e-3,
    compare_fn={"x_normed": ratio_allclose(atol=1e-4, rtol=1.0 / 128)},
    compile_only=args.compile_only,
)
if not result.passed:
    if result.error:
        print(result.error)
    raise SystemExit(1)
```

`config` is one `RunConfig` dict carrying both the compile and the dispatch
half — write every key side by side. `rtol` / `atol` are the default
`torch.allclose` gate; `compare_fn` overrides it per output name with one of
`golden.validation`'s comparators when allclose is the wrong rule (a bounded
outlier ratio, a top-k pair, a slot-mapped paged pool).

Choose a tolerance from the kernel's numerical contract, never by loosening it
until a failing run passes.

### CLI flags

The `__main__` block is argparse plus that call. Beyond `-p` / `-d`, four flags
are conventional, and each one exists to skip work you are not testing:

| Flag | Use it when |
|---|---|
| `--compile-only` | Checking that the DSL lowers; no device needed |
| `--enable-chip-swimlane [0-4]` | Capturing the task timeline (see [Performance Tuning](../debug-and-tune/performance-tuning.md)) |
| `--runtime-dir <dir>` | Re-running a build whose generated `.cpp` / `.pto` you edited |
| `--save-data` / `--golden-data <dir>` | Freezing the reference once, then replaying it (see [Save and Replay](../run-and-validate/save-and-replay.md)) |

Add the replay pair when a kernel is about to be tuned: recomputing a large
Torch golden on every iteration is usually the slowest part of the loop.

---

## See also

- [Golden Harness](../run-and-validate/golden-harness.md) — the full spec, `RunConfig`
  and comparator reference.
- [Save and Replay Golden Data](../run-and-validate/save-and-replay.md) — freezing a
  reference for repeated timing runs.
- [Precision Tuning](../debug-and-tune/precision-tuning.md) — choosing a tolerance, and
  what to do when the mismatch itself is the subject.
