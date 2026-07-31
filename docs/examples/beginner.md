# Beginner Examples

The beginner examples introduce the common shape of a PyPTO-Lib executable:

1. declare a `@pl.jit` kernel;
2. divide the problem into `pl.parallel` tiles;
3. perform work inside a `pl.at` core-group scope;
4. describe inputs and outputs with Golden Harness specs;
5. compare the device or simulator result with a Torch reference.

Both examples accept `-p {a2a3,a2a3sim,a5,a5sim}`. Repository CI exercises
`a2a3`, `a2a3sim`, and `a5sim`.

## Hello World

[View the source](../../examples/beginner/hello_world.py).

`hello_world.py` adds one scalar to every element of an FP32 matrix. It is the
smallest example that shows the complete path from a tensor signature to
validated output.

Pay attention to:

- `pl.parallel` distributing row tiles across core groups;
- `pl.range` walking column tiles within each row tile;
- a tensor slice becoming the tile consumed by `pl.add`;
- `ScalarSpec` and `TensorSpec` describing the test fixture.

Run it on the A2/A3 simulator:

```bash
python examples/beginner/hello_world.py -p a2a3sim
```

## Matmul

[View the source](../../examples/beginner/matmul.py).

`matmul.py` computes an FP32 matrix multiplication with M and N tiling. The K
dimension fits in one tile, making this a useful first cube example before
adding a reduction loop.

Pay attention to:

- nested `pl.parallel` loops selecting output tiles;
- slicing A and B to match the output tile;
- `pl.matmul` producing one tile of C;
- the tolerance passed to the Golden Harness.

Run it on the A5 simulator:

```bash
python examples/beginner/matmul.py -p a5sim
```

## Next step

Continue with [GEMM](intermediate.md#gemm), which adds K tiling and
`pl.matmul_acc`, then compare the normalization examples to see how row
reductions and broadcasts are expressed.
