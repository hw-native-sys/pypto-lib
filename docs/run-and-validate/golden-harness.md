# Golden Harness

The `golden/` package provides the repository's compile, execute, and
correctness-validation path. Its public entry points are exported from
[`golden/__init__.py`](../../golden/__init__.py):

- `TensorSpec` and `ScalarSpec` describe ordered kernel arguments;
- `run` drives a `@pl.program` program;
- `run_jit` drives a module-level `@pl.jit` function;
- validation helpers provide output-specific comparison policies.

## Describe the arguments

The `specs` list must use the same order and names as the kernel parameters.

### TensorSpec

```python
TensorSpec(
    name,
    shape,
    dtype,
    init_value=None,
    is_output=False,
)
```

`init_value` controls tensor creation:

| Value | Result |
|---|---|
| `None` | A zero-filled tensor |
| `int` or `float` | A tensor filled with that value |
| `torch.Tensor` | That tensor converted to the requested dtype |
| `torch.randn`, `torch.rand`, `torch.zeros`, or `torch.ones` | The factory called with the requested shape and dtype |
| another callable | The no-argument result converted to a tensor and the requested dtype |

Random input is therefore explicit: use `init_value=torch.randn` or another
random factory. `init_value=None` does not generate random values.

Set `is_output=True` for every tensor that validation must compare. An output
with a non-`None` initializer is an input/output state tensor: its initial
value is supplied to the runtime and its final value is validated.

### ScalarSpec

`ScalarSpec(name, dtype, value)` represents a scalar kernel argument. The
harness stores it as a zero-dimensional PyTorch tensor and converts it to the
runtime ABI form during dispatch. The name and position must still match the
kernel signature.

See [`golden/spec.py`](../../golden/spec.py) for the complete dtype and
resident-tensor contract.

## Write the golden function

A golden function receives a dictionary keyed by spec name. It must fill all
output entries in place.

The first example uses:

```python
def golden_hello_world(values):
    values["y"][:] = values["x"] + values["a"]
```

The harness gives the golden function cloned inputs and separate zero-filled
pure outputs, so runtime writes cannot mutate the already computed reference.

## Choose run or run_jit

Use `run_jit` for a module-level `@pl.jit` function:

```python
result = run_jit(
    fn=hello_world,
    specs=build_specs(),
    golden_fn=golden_hello_world,
    runtime_cfg={
        "platform": args.platform,
        "device_id": args.device,
    },
    rtol=1e-5,
    atol=1e-5,
)
```

Use `run` for a built `@pl.program`:

```python
result = run(
    program=build_program(),
    specs=build_specs(),
    golden_fn=golden_fn,
    runtime_cfg={
        "platform": args.platform,
        "device_id": args.device,
    },
)
```

Both entry points perform the same input, golden, runtime, and validation
stages after their respective compile path. The detailed sequence and
configuration groups are documented in
[Compile and Runtime Workflow](compile-runtime-workflow.md).

## Validation

Every `TensorSpec` marked as an output is compared with the corresponding
golden output. The default is:

```python
torch.allclose(actual, expected, rtol=rtol, atol=atol)
```

Choose tolerances from the numerical contract of the kernel, not simply to
make a failing result pass. For output types that need a different correctness
rule, pass a comparator for that output name through `compare_fn`.

If neither `golden_fn` nor `golden_data` is provided, the runtime can still
execute, but validation is explicitly reported as skipped. Such a run is not
a correctness check.

## Handle RunResult

`run` and `run_jit` return:

```python
RunResult(
    passed=...,
    error=...,
    execution_time=...,
    work_dir=...,
    bench=...,
    outputs=...,
)
```

Runnable scripts should return a non-zero process status when `passed` is
false:

```python
if not result.passed:
    if result.error:
        print(result.error)
    raise SystemExit(1)
```

`work_dir` identifies the generated build directory and is the reliable way
to locate its reports and optional saved data.

Set `return_outputs=True` on `run` or `run_jit` to expose the live tensors for
every `TensorSpec(is_output=True)` through `result.outputs`. The default is
`None`, avoiding unnecessary resident-output readback. With output capture
enabled, resident state is copied back once before its worker is released,
which lets execution-only staged drivers chain an activation or cache without
enabling a golden comparison. This remains a launchability mechanism, not a
correctness check.

## Golden CPU threads

Importing `golden` configures PyTorch to use 16 intra-op CPU threads for the
reference computation. Override that repository default with a positive
integer when necessary:

```bash
PYPTO_GOLDEN_NUM_THREADS=8 \
  PYTHONPATH="$PWD" \
  python path/to/kernel.py -p a2a3 -d 0
```

## Next steps

- Use [Save and Replay Golden Data](save-and-replay.md) for repeated timing or
  profiling iterations whose numerical contract is unchanged.
- Use [Debugging](../debug-and-tune/debugging.md) when compile, runtime, or
  validation fails.
- Use [Precision Tuning](../debug-and-tune/precision-tuning.md) when the mismatch itself is
  the subject of the work.
