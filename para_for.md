# Loop Grammar: Unified `for` and `pl.range`

Orchestration functions run **serially**: the control flow (including all `for` loops) executes in order. Whether a loop is **sequential** or **parallel** (dividable) is a **semantic** distinction for the compiler: it affects how the loop may be split and how generated code is scheduled at runtime. This document defines a **single, unified** loop syntax: one `for` construct and one `pl.range` API, with **optional parameters** to select sequential vs parallel and chunking.

---

## 1. Unified syntax: one `for`, one `pl.range`

All loops use the same form:

```python
for <index_var> in pl.range(<range_args>, <optional_kwargs>):
    <body>
```

- **Range args:** scalar range (half-open interval), same as today.
- **Optional kwargs:** control **sequential vs parallel** and **chunking**. Omission means sequential.

No separate keyword (e.g. `para_for` or `pl.para_range`) is needed; the same `for` and `pl.range` are used for both sequential and parallel loops.

---

## 2. Scalar range: `pl.range` forms

`pl.range` is a **scalar range**: the loop variable takes integer values over a half-open interval.

| Form | Range | Step |
|------|--------|------|
| `pl.range(end)` | **[0, end)** | 1 |
| `pl.range(start, end)` | **[start, end)** | 1 |
| `pl.range(start, end, step)` | **[start, end)** | step |

---

## 3. Optional parameters: sequential vs parallel and chunking

All of the following are **keyword-only** (or positional range args followed by kwargs).

### 3.1 `parallel`

- **`parallel=False`** (default): **Sequential** loop. Iterations run in strict order; loop-carried dependencies are allowed. The compiler does **not** split the loop.
- **`parallel=True`**: **Parallel** (dividable) loop. Iterations are assumed **independent** (no iteration reads what another writes); the compiler may expand or split the loop (e.g. chunk expansion into chunk loop + in_chunk loop, or subranges) and schedule them in parallel at runtime. This is a **user guarantee** that the compiler can use to simplify interchange and correctness analysis (see §5.5).

```python
for i in pl.range(0, N):                    # sequential
for i in pl.range(0, N, parallel=True):     # parallel, compiler chooses split
```

### 3.2 `chunk`

- **`chunk=C`** (integer, optional): When present, the loop is treated as **parallel** and the iteration space is split into contiguous **chunks of size C**. The compiler performs **loop chunk expansion** (see §5): one chunked loop is converted into **two nested loops** — a **chunk loop** (outer) and an **in_chunk loop** (inner). The body is **not** duplicated.  
- If `chunk` is omitted, `parallel=True` alone does not force a fixed chunk size; the compiler may choose any partition or leave the loop as a single parallel region.

```python
for i in pl.range(0, 4096, chunk=1024):     # parallel, 4 chunks → chunk loop + in_chunk loop
```

### 3.3 `chunk_policy`

- **`chunk_policy`** (optional, default = **`"leading_full"`**): Only meaningful when **`chunk`** is set. Controls how chunk boundaries are chosen.
  - **`"leading_full"`**: All **leading chunks have exactly size C**; only the **last chunk** may be smaller. Boundaries: `start`, `start + C*step`, `start + 2*C*step`, …
  - **`"aligned"`**: Chunk boundaries are **multiples of C** in index space (e.g. 0, C, 2C, …). The first and last chunks may be smaller where trimmed by [start, end).

```python
for i in pl.range(0, N, chunk=1024)                           # leading_full (default)
for i in pl.range(0, N, chunk=1024, chunk_policy="aligned")
```

### 3.4 Summary of parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `parallel` | `False` | `True` → parallel (dividable), no loop-carried dependency |
| `chunk` | — | If set → parallel + loop chunk expansion (chunk loop + in_chunk loop), no body duplication |
| `chunk_policy` | `"leading_full"` | With `chunk`: boundary rule (leading_full vs aligned) |

**Unified examples:**

```python
for t in pl.range(0, 4096):                        # sequential
for t in pl.range(0, 4096, parallel=True):         # parallel, no fixed chunk
for t in pl.range(0, 4096, chunk=1024):            # parallel, 4 chunks (chunk + in_chunk)
for t in pl.range(0, 4096, chunk=1024, chunk_policy="aligned")
```

---

## 4. Semantics

### 4.1 Sequential (`parallel` not set or `parallel=False`)

- Iterations execute in **strict program order**.
- Loop-carried dependencies are allowed.
- The compiler does **not** split the loop; lowering preserves a single loop.

### 4.2 Parallel (`parallel=True` or `chunk=C`)

1. **Independence:** No iteration may read a value written by another iteration of the same loop. The compiler may assume there are no loop-carried dependencies.
2. **Dividable:** The loop is equivalent to one or more child loops over **disjoint subranges** whose union is the original range; each child has the same body, with the index variable restricted to its subrange.
3. **Execution:** At runtime, subranges (or chunks) may be scheduled in parallel; order is unspecified.
4. **Reductions:** If the body reduces into a shared buffer, the user must use explicit reduction primitives or the compiler must insert a reduction phase (see §8).

### 4.3 Splitting rules (when `chunk` is set)

- **Extent** = number of iterations = `(end - start + step - 1) // step`.
- **leading_full:** Chunk c (0-based) has indices `[start + c*C*step, start + min((c+1)*C, extent)*step)`.
- **aligned:** Boundaries at multiples of C in index space; intersect with [start, end) to get chunks (first/last may be smaller).
- The compiler does **not** create discrete copies of the loop body per chunk. Instead it performs **loop chunk expansion** (see §5): one chunked loop is converted into **two nested loops** — an **outer chunk loop** (over chunks) and an **inner in_chunk loop** (over iterations within each chunk).

### 4.4 `with pl.incore` inside a splittable loop: common anonymous incore

The **`with pl.incore`** grammar defines an **implicit** (anonymous) **incore function** boundary. When it is used inside chunked (splittable) loops, the compiler expands each chunked loop into (chunk_loop, in_chunk_loop), applies loop interchange when legal, and places the incore scope so it encompasses only the in_chunk loops and body (§5). The **anonymous incore** is **common** to all chunk iterations: outlined **once**, invoked per chunk with the appropriate bounds; the chunk loop stays in orchestration.

---

## 5. Loop chunk expansion: how the compiler processes `chunk`

Instead of creating **discrete copies** of the loop body for each chunk, the compiler converts each chunked loop into **two nested loops**. This transformation is called **loop chunk expansion**.

### 5.1 One chunked loop → chunk loop (outer) + in_chunk loop (inner)

For a source loop:

```python
for t in pl.range(0, 4096, chunk=1024):
    <body>
```

the compiler expands it into:

- **Outer loop (chunk loop):** iterates over **chunk indices** `c = 0, 1, 2, 3` (one iteration per chunk).
- **Inner loop (in_chunk loop):** for each chunk `c`, iterates over the **index within the chunk** — i.e. `t` runs over the subrange of that chunk (e.g. for chunk 0: `t in [0, 1024)`; for chunk 1: `t in [1024, 2048)`; etc.).

**Expanded structure (skeleton):**

```python
# num_chunks = 4 (for extent 4096, chunk 1024)
for c in pl.range(0, num_chunks):           # chunk loop (outer)
    t_start = c * 1024
    t_end   = min(t_start + 1024, 4096)
    for t in pl.range(t_start, t_end):      # in_chunk loop (inner)
        <body>   # same body, t ranges over this chunk only
```

So there is **one** body, with **two** nested loops: the outer loop selects the chunk, the inner loop runs over the indices in that chunk. No duplication of the body across chunks.

### 5.2 Nested chunked loops and `with pl.incore`

When a **`with pl.incore`** scope **covers** nested chunked loops, each such chunked loop is expanded into (chunk_loop, in_chunk_loop). So we get multiple layers of (chunk, in_chunk) pairs.

**Example:** two chunked loops inside one incore scope.

```python
with pl.incore():
    for i in pl.range(0, 4096, chunk=1024):
        for j in pl.range(0, 2048, chunk=512):
            <body using i, j>
```

After expansion (each chunked loop → chunk + in_chunk):

- Loop `i` → outer `c_i` (chunk over i), inner `i` (in_chunk over i).
- Loop `j` → outer `c_j` (chunk over j), inner `j` (in_chunk over j).

Initial nesting (expansion order) might be: `c_i` → `i` → `c_j` → `j` → body, or similar, depending on how the compiler expands.

### 5.3 Loop interchange: chunk loops outside, in_chunk loops inside

The compiler should **try to interchange** loop nesting so that:

- All **chunk loops** (outer loops from expansion) are moved **to the outside**.
- All **in_chunk loops** (inner loops from expansion) are moved **to the inside**.

**Rule:** A loop layer may be interchanged with an adjacent layer **only if** the interchange does **not** cause a **semantic discrepancy** (e.g. no violation of data dependencies, no change in the order of reads/writes that affect correctness). If interchanging would change program semantics, that interchange is **not** allowed. When interchange can break semantics and how to handle it is detailed in §5.5.

**Goal:** After all legal interchanges, the structure becomes:

```text
for c_1 in ...:           # chunk loop 1
  for c_2 in ...:         # chunk loop 2
    ...
      with pl.incore():   # incore placed here (see §5.4)
        for i in ...:     # in_chunk loop 1
          for j in ...:   # in_chunk loop 2
            <body>
```

So all chunk loops are outermost; all in_chunk loops are innermost; the incore scope encloses only the in_chunk loops and the body.

### 5.4 Incore scope placement: minimize scope, preserve semantics

When a **`with pl.incore`** scope originally covers nested chunked loops (or a mix of chunked and non-chunked loops), the compiler:

1. Expands each chunked loop into (chunk_loop, in_chunk_loop).
2. Interchanges loop layers as allowed by semantics so that all chunk loops are outer and all in_chunk loops are inner.
3. **Places the incore boundary** at the **innermost** position that is still **outside** all in_chunk loops: the incore scope **encompasses** the in_chunk loops (and the body) and is **inside** the last (innermost) chunk loop that is allowed to enclose it. Any chunk loop that cannot be moved outside the incore without causing a semantic discrepancy remains in its current position (the compiler does not force an illegal interchange).

**Effect:** The **size of the incore scope** is **minimized**: it contains only the in_chunk loops and the body, not the chunk loops. The chunk loops stay in orchestration; the incore is invoked once per (chunk) iteration with the appropriate bounds. This preserves the **original semantic equivalence** of the program while limiting what is compiled into the anonymous incore function.

**Summary:**

| Step | Action |
|------|--------|
| 1 | **Loop chunk expansion:** Each chunked loop → (chunk_loop, in_chunk_loop). |
| 2 | **Loop interchange:** Move chunk loops outward and in_chunk loops inward wherever semantics allow. |
| 3 | **Incore placement:** Put `with pl.incore` so it encloses the in_chunk loops (and body) and is inside the chunk loops, minimizing incore scope. |

**Example (after expansion and placement):** One chunked loop with `with pl.incore` in the body.

```python
# Chunk loop stays in orchestration
for c in pl.range(0, 4):
    t_start = c * 1024
    t_end   = min(t_start + 1024, 4096)
    r_start = t_start * TILE_M   # or derived from t_start
    x_tile = pl.slice(x, [TILE_M, N], [r_start, 0])
    with pl.incore():    # placed to encompass only the in_chunk loop + body
        for t in pl.range(t_start, t_end):   # in_chunk loop inside incore
            # body: e.g. load(x_tile), softmax, store
            ...
```

The incore scope is **limited** to the in_chunk loop and body; the chunk loop and view setup are outside. Semantics are preserved; incore size is minimized.

### 5.5 When loop interchange breaks semantic equivalence, and how to handle it

#### 5.5.0 How the `parallel` parameter simplifies correctness analysis

The **`parallel`** (and **`chunk`**) parameter is a **user guarantee**: the user asserts that no iteration of that loop reads a value written by another iteration of the same loop (§4.2). The compiler can use this to **simplify** interchange correctness:

- **Chunk-loop interchange:** Under this guarantee, different chunks are independent. The compiler may **always** interchange chunk loops (e.g. `c_i` and `c_j`) without dependency analysis — see §5.5.2.
- **In_chunk-loop interchange when all source loops are parallel/chunked:** If **every** loop in a nested group was marked **parallel** or **chunk** in the source, then the combined iteration space `(i, j, …)` has no cross-iteration dependency: no `(i,j)` reads what `(i',j')` wrote for `(i',j') ≠ (i,j)`. So the compiler may **interchange any pair of in_chunk loops** derived from those parallel/chunked loops **without** doing fine-grained dependency or reduction-axis analysis; any iteration order within the chunk is valid.
- **When the compiler must still do full analysis:** If the nest mixes **sequential** and parallel/chunked loops (e.g. outer loop parallel, inner loop sequential), the sequential loop carries dependencies. The compiler **cannot** assume independence for the in_chunk loop that came from the sequential loop, and must fall back to **dependency analysis** (and reduction-axis analysis) as in §5.5.1 before interchanging that in_chunk loop with another.

**Summary:** The more loops the user marks as `parallel` or `chunk`, the fewer constraints there are on interchange; the compiler can safely reorder chunk loops and (when all involved loops are parallel) in_chunk loops without proving the absence of dependencies. Only when a loop is **sequential** must the compiler prove that a specific interchange does not violate dependencies.

Interchange can still change the **order** in which iterations run. The following situations may cause **semantic discrepancy** when the above guarantee does not apply or when other constraints (e.g. incore placement) are involved; the compiler must detect them and **refrain from** the offending interchange (or apply a compensating transformation).

#### 5.5.1 Dependencies between two in_chunk loops (e.g. `i` and `j`)

**When this applies:** When at least one of the source loops was **sequential** (not marked `parallel`/`chunk`), the compiler cannot rely on the “all iterations independent” guarantee and must perform the analysis below. When **all** source loops are parallel/chunked, §5.5.0 applies and interchange of in_chunk loops is allowed without this analysis.

After expansion we have two (or more) **in_chunk** loops, e.g. over index `i` (within chunk of the first dimension) and index `j` (within chunk of the second). The **order** of these two loops (which is outer, which is inner) affects iteration order and thus semantics if the body has a **dependency** between the two indices.

- **Flow / anti / output dependency between `i` and `j`:**  
  Example: the body first does a loop over `i` writing to a buffer (e.g. `row_sum[i]`), then a loop over `j` that reads that buffer. The intended order is “for each `i` complete row_sum[i], then use row_sum in `j`”. If we interchange so `j` is outer and `i` is inner, the read-before-write order is broken.  
  **Rule:** Do **not** interchange two in_chunk loops if there exists a **data dependency** between the loop indices (one loop’s iterations produce values that the other loop’s iterations consume, in an order that would change after interchange). Dependency analysis (e.g. distance/direction vectors, or simple def-use chains across the two loops) must be applied; if the dependency is not compatible with both orderings, keep the original nesting.

- **Reduction along one dimension:**  
  Example: `for i in chunk_i: for j in chunk_j: row_max[i] = max(row_max[i], x[i,j])`. The outer loop (i) carries the reduction; the inner loop (j) is the reduction domain. Interchanging to `for j: for i: row_max[i] = ...` preserves semantics (same row_max[i] updates). But if the reduction were written so that the *outer* index is the reduction domain (e.g. col_max[j] with i outer), then interchanging would change which index is “reduction dimension” and can break semantics. So: **interchange of in_chunk loops is only valid when the dependency/reduction structure allows both orderings**; otherwise the compiler must keep the original order.

- **Element-wise only (no cross-index dependency):**  
  If the body is purely element-wise (e.g. `out[i,j] = f(in[i,j])`) with no shared buffer indexed by both `i` and `j` in a dependent way, both (i-outer, j-inner) and (j-outer, i-inner) are semantically equivalent. Interchange is safe.

#### 5.5.2 Order of chunk loops (`c_i` vs `c_j`)

Under the **parallel** semantics (§4.2), different chunk indices are **independent**: no chunk may read what another chunk wrote. So interchanging the **chunk loops** (e.g. which of `c_i` and `c_j` is outermost) does **not** by itself change the result of the computation, and is **safe** for correctness.

- **Reduction across chunks:**  
  If the program uses explicit reduction across chunks (e.g. §8), the reduction is associative; the order in which chunk partials are combined is unspecified. So chunk-loop order need not be preserved for correctness.

- **Observable side effects:**  
  If the body has side effects (e.g. I/O, printing) that are visible and order-sensitive, the user is already outside the “no loop-carried dependency” guarantee; the compiler may still interchange chunk loops and treat order as unspecified.

#### 5.5.3 Chunk loop cannot be moved outside incore

Sometimes a **chunk loop** is forced to stay **inside** the incore scope (e.g. because moving it out would require passing large or dynamic data that the current incore ABI does not support, or because some internal constraint ties the chunk index to incore). In that case, the compiler does **not** perform that interchange: the chunk loop remains where it is, and the incore scope **encompasses** that chunk loop as well (§5.4). The result is a **larger** incore scope, but semantics are preserved.

#### 5.5.4 Handling summary

| Situation | Risk | Handling |
|-----------|------|----------|
| **All loops parallel/chunked** (§5.5.0) | — | **No extra analysis:** chunk-loop and in_chunk-loop interchange are **allowed**; user guarantee implies no cross-iteration dependency. |
| Dependency between two in_chunk indices (e.g. i, j) | Interchanging the two in_chunk loops changes read/write order → wrong result | **Do not interchange** those two in_chunk loops (or only when dependency allows both orders). Applies when at least one source loop is **sequential**; otherwise §5.5.0 applies. |
| Reduction structure fixes “outer = reduction index” | Interchanging in_chunk loops can swap reduction vs. reduction domain | **Do not interchange** if it would change which dimension is reduced over; otherwise allow. Relevant when loop is sequential or reduction is not expressed as independent per (i,j). |
| Chunk loops (c_i, c_j) order | Under parallel semantics, chunks are independent | Interchange of chunk loops is **allowed**; order is unspecified. |
| Chunk loop must stay inside incore (ABI / constraints) | Moving it out would break semantics or contract | **Do not interchange** that chunk loop out of incore; leave incore scope larger. |

The compiler may **first** check whether all involved loops are marked `parallel` or `chunk`; if so, interchange (chunk and in_chunk) is allowed without dependency analysis. Otherwise, the compiler should implement **dependency analysis** (and, if needed, reduction-axis analysis) before applying interchange; only apply interchange when the transformation is **semantically legal**. When an interchange is disallowed, the generated code keeps the current nesting and, if applicable, a larger incore scope.

---

## 6. Chunk expansion and generated code (4K range, 4 chunks)

When `pl.range(0, 4096, chunk=1024)` is used, the compiler applies **loop chunk expansion** (and optionally interchange / incore placement as in §5). The result is one chunk loop and one in_chunk loop, not four separate copies of the body.

| Chunk `c` | `t` (in_chunk) range |
|-----------|------------------------|
| 0 | `[0, 1024)` |
| 1 | `[1024, 2048)` |
| 2 | `[2048, 3072)` |
| 3 | `[3072, 4096)` |

**Source (unified syntax):**

```python
for t in pl.range(0, 4096, chunk=1024):
    r_start = t * TILE_M
    ...
```

**Generated code after expansion (skeleton):** one outer chunk loop, one inner in_chunk loop; body is not duplicated.

```python
for c in pl.range(0, 4):                    # chunk loop (outer)
    t_start = c * 1024
    t_end   = min(t_start + 1024, 4096)
    for t in pl.range(t_start, t_end):      # in_chunk loop (inner)
        r_start = t * TILE_M
        ...
```

The runtime may still schedule different chunk iterations in parallel (e.g. different `c` values); the important point is that the **compiler** representation is two nested loops, not four separate loop copies.

---

## 7. Example: large-tensor softmax (4K range, 4 chunks)

Softmax over a matrix `x` of shape `[M, N]` per row. We use **4096** tile iterations and **4 chunks** of 1024; the compiler expands to a chunk loop + in_chunk loop (§5–§6). The **dividable** part is the loop over tile index; the **non-dividable** part is the sequential reduction (row_max → row_sum → div) inside each tile.

### 7.1 Program (unified `for` syntax)

```python
num_tiles = 4096   # 4K

# Dividable: parallel over row tiles (unified for + pl.range with chunk)
for t in pl.range(0, num_tiles, chunk=1024):
    r_start = t * TILE_M
    r_end = min(r_start + TILE_M, M)
    x_tile = pl.slice(x, [r_end - r_start, N], [r_start, 0])

    # Non-dividable: sequential reduction within the tile
    for phase in pl.range(3):
        if phase == 0:
            row_max_tile = self.incore_row_max(x_tile, tmp_tile)
        elif phase == 1:
            exp_tile = self.incore_exp_sub(x_tile, row_max_tile)
            row_sum_tile = self.incore_row_sum(exp_tile, tmp_tile)
        else:
            y_tile = self.incore_div(exp_tile, row_sum_tile)
            pl.store(y_tile, ..., out[r_start:r_end, :])
```

- Outer loop: **one** `for` with `pl.range(0, 4096, chunk=1024)` → compiler expands to chunk loop + in_chunk loop (§5).
- Inner loop: **one** `for` with `pl.range(3)` (no `parallel`, no `chunk`) → sequential, not split.

### 7.2 Generated code after expansion (chunk + in_chunk)

After **loop chunk expansion** (§5–§6): one **chunk loop** over `c in [0, 4)` and one **in_chunk loop** over `t in [t_start, t_end)`; the body is not duplicated. The inner `for phase in pl.range(3)` remains inside the in_chunk loop.

---

## 8. Reduction across the 4 chunks: options and generated code

When the chunked loop (after expansion: chunk loop + in_chunk loop) splits a dimension over which a reduction is defined (e.g. column-split softmax), the compiler must combine partial results. Below, **all options** use the **unified** syntax (`pl.range(..., chunk=1024)` in source). Generated code may show either the **expanded form** (one chunk loop + one in_chunk loop) or, equivalently, a flattened view with one loop per chunk for clarity; semantics are the same.

**Setup:** Column-split softmax; each of the 4 **chunks** (iterations of the chunk loop) owns columns `[c*1024, (c+1)*1024)` and produces partial row_max and row_sum. We need to merge into global row_max and row_sum before the final div.

---

### Option A: Two-phase — parallel partials, then sequential combine

**Generated code:** Phase 1 and Phase 3 use the **expanded** form (chunk loop + in_chunk loop) for consistency with §5–§6.

```python
# Phase 1: chunk loop + in_chunk loop (from pl.range(0, 4096, chunk=1024))
for c in pl.range(0, 4):
    t_start, t_end = c * 1024, min((c + 1) * 1024, 4096)
    for t in pl.range(t_start, t_end):
        # compute partial_row_max_c[r], partial_row_sum_c[r] for cols [c*1024, (c+1)*1024)
        ...

# Phase 2: sequential combine
for r in pl.range(M):
    global_max[r] = max(partial_row_max_0[r], partial_row_max_1[r],
                        partial_row_max_2[r], partial_row_max_3[r])
    global_sum[r] = partial_row_sum_0[r] + partial_row_sum_1[r] + \
                    partial_row_sum_2[r] + partial_row_sum_3[r]

# Phase 3: final div (chunk loop + in_chunk loop again)
for c in pl.range(0, 4):
    t_start, t_end = c * 1024, min((c + 1) * 1024, 4096)
    for t in pl.range(t_start, t_end):
        # y = exp(x - global_max) / global_sum
        ...
```

---

### Option B: Tree reduction

**Generated code:** Level 0 and Final use chunk loop + in_chunk loop.

```python
# Level 0: chunk loop + in_chunk loop (4 chunks)
for c in pl.range(0, 4):
    t_start, t_end = c * 1024, min((c + 1) * 1024, 4096)
    for t in pl.range(t_start, t_end):
        # partial_max_c, partial_sum_c
        ...

# Level 1: 2 merges (can be parallel over rows)
for r in pl.range(M):
    combined_max_01[r] = max(partial_max_0[r], partial_max_1[r])
    combined_sum_01[r] = partial_sum_0[r] + partial_sum_1[r]
for r in pl.range(M):
    combined_max_23[r] = max(partial_max_2[r], partial_max_3[r])
    combined_sum_23[r] = partial_sum_2[r] + partial_sum_3[r]

# Level 2: 1 merge
for r in pl.range(M):
    global_max[r] = max(combined_max_01[r], combined_max_23[r])
    global_sum[r] = combined_sum_01[r] + combined_sum_23[r]

# Final: chunk loop + in_chunk loop
for c in pl.range(0, 4):
    t_start, t_end = c * 1024, min((c + 1) * 1024, 4096)
    for t in pl.range(t_start, t_end):
        # y = exp(x - global_max) / global_sum
        ...
```

---

### Option C: Single sequential merge over partial index

**Generated code:**

```python
# Chunk loop + in_chunk loop (4 chunks)
for c in pl.range(0, 4):
    t_start, t_end = c * 1024, min((c + 1) * 1024, 4096)
    for t in pl.range(t_start, t_end):
        # partial_max_c, partial_sum_c
        ...

# Sequential reduce over 4 partials
for k in pl.range(4):
    for r in pl.range(M):
        if k == 0:
            global_max[r] = partial_max_0[r]
            global_sum[r] = partial_sum_0[r]
        else:
            global_max[r] = max(global_max[r], partial_max_k[r])
            global_sum[r] += partial_sum_k[r]

# Final: chunk loop + in_chunk loop
for c in pl.range(0, 4):
    t_start, t_end = c * 1024, min((c + 1) * 1024, 4096)
    for t in pl.range(t_start, t_end):
        ...
```

---

### Option D: No cross-chunk reduction (row-split)

Each of the 4 chunks owns disjoint rows; row max/sum are local. **Generated code:** chunk loop + in_chunk loop only (as in §6–§7.2); no combine phase. All loops use the same unified `pl.range` syntax.

---

### Summary of reduction options

| Option | When to use | Generated structure |
|--------|-------------|----------------------|
| **A** | Column-split reduction; simple single combine | chunk+in_chunk → 1 sequential combine → chunk+in_chunk for div |
| **B** | More parallelism in combine | chunk+in_chunk → 2 parallel merges → 1 merge → chunk+in_chunk for div |
| **C** | Explicit “reduce over 4” loop | chunk+in_chunk → 1 sequential loop over k=0..4 (and rows) → chunk+in_chunk for div |
| **D** | Row-split (reduction not over split dim) | chunk+in_chunk only; no combine |

---

## 9. IR / lowering

- **Frontend:** Parse `for x in pl.range(..., parallel=..., chunk=..., chunk_policy=...)` as a single loop AST node; attach attributes `parallel`, `chunk`, `chunk_policy` from the range call.
- **IR:** One loop node; if `parallel=True` or `chunk` is set, mark as dividable/splittable. If `chunk` is set, lowering performs **loop chunk expansion**: one **chunk loop** (outer) and one **in_chunk loop** (inner); no body duplication.
- **Lowering:** Sequential loop → single loop. Chunked loop → chunk loop + in_chunk loop (expansion). Other parallel loops → single parallel region or subranges; backend may emit OpenMP-style parallel for, or separate tasks, or SPMD.

---

## 10. Summary

- **One syntax:** All loops use `for x in pl.range(...)` with optional kwargs. No separate `para_for` or `pl.para_range`.
- **Sequential:** `pl.range(end)`, `pl.range(start, end)`, `pl.range(start, end, step)` — default, ordered, not split.
- **Parallel:** `pl.range(..., parallel=True)` or `pl.range(..., chunk=C)` — iterations independent, dividable; compiler may expand (chunk + in_chunk) or split subranges.
- **Chunking:** `chunk=C` + optional `chunk_policy="leading_full"` | `"aligned"` — fixed-size chunks; compiler performs **loop chunk expansion** (chunk loop + in_chunk loop), no body duplication.
- **Orchestration** remains serial; parallelism is a semantic hint for code generation and runtime scheduling.
- **`with pl.incore`** inside a splittable loop body defines an anonymous incore boundary; when the loop is chunk-expanded, that incore is **common** (outlined once, shared by all chunk iterations), not duplicated per chunk.
- **Chunk processing:** The compiler performs **loop chunk expansion**: a chunked loop becomes **two nested loops** — **chunk loop** (outer, over chunks) and **in_chunk loop** (inner, over indices within each chunk). No discrete copies of the body. For nested chunked loops under `with pl.incore`, the compiler **interchanges** loop layers when semantics allow so that all chunk loops are outer and all in_chunk loops are inner, then **places the incore scope** to encompass only the in_chunk loops (and body), minimizing incore size while preserving semantics.

This keeps the grammar minimal and consistent: one `for`, one `pl.range`, with parameters controlling behavior; chunking is implemented by expansion and interchange, and incore scope is minimized.
