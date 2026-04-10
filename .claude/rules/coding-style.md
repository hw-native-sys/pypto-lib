# PyPTO Frontend Coding Style

## Module Import

```python
import pypto.language as pl
```

Always use `pl` as the module prefix. Do not use `import pypto.language as ir` or other aliases.

## Program Structure

Use `@pl.program` class with `@pl.function` methods:

```python
@pl.program
class MyProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def my_kernel(self, ...):
        ...

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(self, ...):
        ...
```

## Function Types

PyPTO supports three function types:

| Type | Purpose | When to use |
|------|---------|-------------|
| `pl.FunctionType.InCore` | Runs on AICore. Manually load/store between GM and UB. | Explicit control over data movement and memory placement |
| `pl.FunctionType.Orchestration` | Host/AICPU scheduling. Calls InCore kernels, manages tensor allocation. | Composing InCore kernels into a computation graph |
| `pl.FunctionType.Opaque` | Compiler decides InCore/Orchestration boundary. Use with `pl.at()`. | When you don't need manual placement control |

### Explicit InCore + Orchestration (pypto standard style)

```python
@pl.program
class HelloWorldProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_add(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
        tile_b = pl.load(b, offsets=[0, 0], shapes=[128, 128])
        tile_c = pl.add(tile_a, tile_b)
        out_c = pl.store(tile_c, offsets=[0, 0], output_tensor=c)
        return out_c

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        out_c: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], dtype=pl.FP32)
        out_c = self.tile_add(a, b, out_c)
        return out_c
```

### Opaque + auto_incore (pypto-lib style)

```python
@pl.program
class SoftmaxProgram:
    @pl.function(type=pl.FunctionType.Opaque)
    def softmax(self, input_tensor: pl.Tensor[[B, S, H], pl.FP32], ...):
        with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
            for b in pl.parallel(0, B, 1, chunk=4):
                ...
```

## Type System

| Category | Examples |
|----------|----------|
| Scalar types | `pl.FP16`, `pl.FP32`, `pl.BF16`, `pl.INT8`, `pl.INT32`, `pl.INT64`, `pl.UINT8`, `pl.BOOL` |
| Tensor | `pl.Tensor[[M, N], pl.FP32]` — global memory tensor |
| Tile | `pl.Tile[[16, 16], pl.FP16]` — on-chip buffer data |
| Scalar wrapper | `pl.Scalar[pl.FP32]` |

### Parameter Direction

| Direction | Syntax | Meaning |
|-----------|--------|---------|
| In (default) | `x: pl.Tensor[...]` | Read-only input |
| Out | `x: pl.Out[pl.Tensor[...]]` | Write-only output |
| InOut | `x: pl.InOut[pl.Tensor[...]]` | Read-write |

## Data Movement (InCore functions)

InCore functions explicitly manage data transfer between memory levels:

```python
# GM → UB (vector unit buffer)
tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])

# UB → GM
out = pl.store(tile_c, offsets=[0, 0], output_tensor=c)

# Between memory spaces (for matmul pipeline)
tile_l1 = pl.load(a, offsets=[0, 0], shapes=[64, 64], target_memory=pl.MemorySpace.Mat)
tile_l0a = pl.move(tile_l1, target_memory=pl.MemorySpace.Left)
tile_l0b = pl.move(tile_l1, target_memory=pl.MemorySpace.Right)
tile_l0c = pl.matmul(tile_l0a, tile_l0b)
```

### Memory Spaces

| Space | Alias | Purpose |
|-------|-------|---------|
| `pl.MemorySpace.Vec` | UB | Vector unit buffer (elementwise, reduction) |
| `pl.MemorySpace.Mat` | L1 | Matrix buffer (matmul staging) |
| `pl.MemorySpace.Left` | L0A | Matmul left operand |
| `pl.MemorySpace.Right` | L0B | Matmul right operand |

## Tensor/Tile Operations

### Elementwise
`pl.add`, `pl.sub`, `pl.mul`, `pl.div`, `pl.exp`, `pl.sqrt`, `pl.relu`, `pl.neg`

### Reduction (row-wise)
`pl.row_max`, `pl.row_sum`

### Row/Column broadcast operations
`pl.row_expand_sub`, `pl.row_expand_div`, `pl.col_expand_mul`

### Tensor management
`pl.create_tensor([M, N], dtype=pl.FP32)` — allocate GM tensor (Orchestration)
`pl.create_tile([M, N], dtype=pl.FP32, target_memory=...)` — allocate on-chip tile (InCore)

### Tensor access
`pl.slice(tensor, [offset_0, offset_1], [size_0, size_1])`

### Shape manipulation
`pl.reshape(tile, [new_shape])`

### Linear algebra
`pl.matmul(left, right)` — cube unit matrix multiply

## Orchestration Output Pattern

In Orchestration functions, output tensors follow this pattern:

```python
# 1. Allocate output tensor
out: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
# 2. Pass as pl.Out parameter to InCore kernel, capture return
out = self.my_kernel(input_a, input_b, out)
# 3. Return or pass to next kernel
return out
```

## Constants and Configuration

Define shape/tiling constants as module-level UPPER_SNAKE_CASE variables:

```python
BATCH = 16
MAX_SEQ = 4096
HIDDEN = 5120
HEAD_DIM = 128
K_CHUNK = 256       # tiling chunk sizes with brief comments
```

## Loop Constructs

- `pl.range(n)` — sequential loop
- `pl.parallel(start, end, step, chunk=N)` — parallel loop with chunk size
- `pl.break_()` — early exit from loops

## Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Program classes | PascalCase, suffix `Program` | `SoftmaxProgram`, `HelloWorldProgram` |
| InCore functions | snake_case, prefix `kernel_` or descriptive | `kernel_add`, `tile_softmax`, `matmul` |
| Orchestration functions | `orchestrator` or `orch_*` | `orchestrator`, `orch_vector` |
| Module constants | UPPER_SNAKE_CASE | `NUM_HEADS`, `HEAD_DIM` |
| Builder functions | `build_*_program` | `build_softmax_program` |
| Loop variables | Short lowercase | `b`, `s`, `h`, `hb` |
| Tensor variables | Descriptive snake_case | `input_tensor`, `max_vals` |
| Tile variables | Descriptive, optionally with memory suffix | `tile_a`, `tile_a_l1`, `tile_a_l0a` |

## GM Tensor Alignment

All `pl.slice` of global memory (GM) tensors must be >= 512 bytes to satisfy hardware alignment requirements.

## File Structure

1. Copyright header
2. Module docstring (design goals, constraints)
3. Imports (`os`, `pypto.language as pl`)
4. Module-level constants (shapes, chunk sizes)
5. `@pl.program` class (or builder function containing it)
6. `if __name__ == "__main__":` block for standalone execution
