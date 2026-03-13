# Qwen3 32B Variable Sequence Length Handling Analysis

## 1. Overview

Both the **decode** (`qwen3_32b_decode.py`) and **prefill** (`qwen3_32b_prefill.py`)
programs support variable sequence lengths per session within a batch.  The input tensor
`seq_lens: Tensor[[BATCH], INT32]` carries the per-session length.  This document
traces exactly how `seq_lens` is read, propagated, and consumed at the
**orchestration** (Opaque function / host-side control flow) and **InCore** (on-chip
kernel) levels.

---

## 2. Decode Program

### 2.1 Input Contract

| Tensor | Shape | Role |
|--------|-------|------|
| `hidden_states` | `[BATCH, HIDDEN]` | One new token per session (fixed shape, no padding) |
| `seq_lens` | `[BATCH]`, INT32 | `seq_lens[b]` = total context length for session `b` (including the new token) |
| `k_cache` / `v_cache` | `[BATCH * NUM_KV_HEADS * MAX_SEQ, HEAD_DIM]` | Flattened KV cache, padded to MAX_SEQ per head per session |
| `rope_cos` / `rope_sin` | `[MAX_SEQ, HEAD_DIM]` | RoPE table, indexed by absolute position |

### 2.2 Program Structure (3 Scopes)

```
Scope 1:  auto_incore  ─── Input RMSNorm + Q/K/V projection
Scope 2:  for b         ── per-batch orchestration loop ─┐
          ├── KV RoPE + cache write (orchestration)       │  Uses seq_lens
          └── auto_incore ── attention (InCore)           │
Scope 3:  auto_incore  ─── Output proj + MLP + residual
```

### 2.3 Scope 1 — Input RMSNorm + Q/K/V Projection

**Does NOT use `seq_lens`.**

All tensors have fixed shapes (`[BATCH, HIDDEN]`, `[BATCH, KV_HIDDEN]`).  In decode,
every batch item contributes exactly one token, so there is no sequence-axis variability
at this stage.  The scope operates on the full batch without any dynamic bounds.

### 2.4 Scope 2 — RoPE + Cache Update + Attention

This is where `seq_lens` drives **all** dynamic behavior.

#### 2.4.1 Orchestration Level

```python
for b in pl.parallel(0, BATCH, 1, chunk=4):       # (A) batch loop
    ctx_len = pl.tensor.read(seq_lens, [b])        # (B) READ seq_lens
    pos = ctx_len - 1                               # (C) decode position
    ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE  # (D) tile count
```

| Step | Variable | Derivation | Type |
|------|----------|-----------|------|
| (B) | `ctx_len` | `seq_lens[b]` | runtime scalar (per-batch) |
| (C) | `pos` | `ctx_len - 1` | runtime scalar |
| (D) | `ctx_blocks` | `⌈ctx_len / SEQ_TILE⌉` | runtime scalar → InCore loop bound |

**Derived control flow:**

- **RoPE cos/sin lookup**: `rope_cos[pos, :]`, `rope_sin[pos, :]` — position-dependent
  indexing into the RoPE table.
- **KV cache write**: `cache_row = b * NUM_KV_HEADS * MAX_SEQ + kvh * MAX_SEQ + pos` —
  the new K/V vectors are written to position `pos` in the flattened cache.

#### 2.4.2 InCore Level (Attention Kernel — `incore_2`)

```python
with pl.auto_incore():                              # enters InCore
    for h in pl.parallel(0, NUM_HEADS, 1, chunk=8): # per-head parallelism
        # ... Q RoPE (uses pos from orchestration) ...

        for sb in pl.range(ctx_blocks):             # (E) dynamic loop bound
            s0 = sb * SEQ_TILE
            valid_len = pl.min(SEQ_TILE, ctx_len - s0)  # (F) tail tile size
            cache_row0 = b * NUM_KV_HEADS * MAX_SEQ + kvh * MAX_SEQ + s0  # (G) address

            k_tile = pl.slice(k_cache, [SEQ_TILE, HEAD_DIM], [cache_row0, 0],
                             valid_shape=[valid_len, HEAD_DIM])
            v_tile = pl.slice(v_cache, [SEQ_TILE, HEAD_DIM], [cache_row0, 0],
                             valid_shape=[valid_len, HEAD_DIM])
            # ... online softmax attention ...
```

| Item | How `seq_lens` reaches it | Effect |
|------|--------------------------|--------|
| **(E)** `ctx_blocks` | `⌈seq_lens[b] / SEQ_TILE⌉` | Determines how many cache-tile iterations the attention loop runs. A session with `ctx_len=50` runs 1 block; `ctx_len=4096` runs 35 blocks. |
| **(F)** `valid_len` | `min(SEQ_TILE, seq_lens[b] - s0)` | On the **last** tile, only `valid_len` rows of `k_tile` / `v_tile` contain real cache entries. The rest is padding. |
| **(G)** `cache_row0` | Derived from `b`, `kvh`, `s0`, and `MAX_SEQ` | Computes the starting row in the flattened KV cache for each tile. `MAX_SEQ` is the padded dimension; `s0` steps through actual cache entries. |

**Tail-tile handling inside InCore:**

```
scores[1, SEQ_TILE] = q @ k_tile^T * scale
scores_valid[1, valid_len] = view(scores, [1, valid_len])  ← mask padding
cur_mi = row_max(scores_valid)
exp_scores = exp(scores_valid - cur_mi)
cur_li = row_sum(exp_scores)
exp_pad[1, SEQ_TILE] = assemble(zeros, exp_scores, [0,0])  ← zero-pad back
oi_tmp = exp_pad @ v_tile                                    ← correct weighted sum
```

The `scores_valid` view + `exp_pad` zero-padding is the current workaround ensuring
`row_max` / `row_sum` only consider valid cache entries.  Once the compiler propagates
`valid_shape` from `k_tile` / `v_tile`, this workaround can be removed (see
`TODO(valid_shape)` in code).

### 2.5 Scope 3 — Output Projection + MLP + Residual

**Does NOT use `seq_lens`.**

Same as Scope 1 — operates on fixed `[BATCH_TILE, HIDDEN]` shapes.  All batch items
produce exactly one output token.

### 2.6 Decode `seq_lens` Flow Diagram

```
seq_lens[b]  (GM tensor, INT32)
      │
      ▼
  pl.tensor.read(seq_lens, [b])        ── Orchestration: read scalar
      │
      ├──► ctx_len = seq_lens[b]
      │       │
      │       ├──► pos = ctx_len - 1
      │       │       │
      │       │       ├──► RoPE lookup: rope_cos[pos], rope_sin[pos]  (Orchestration)
      │       │       └──► KV cache write: cache[..., pos, :] = k_rot / v  (Orchestration)
      │       │
      │       ├──► ctx_blocks = ⌈ctx_len / SEQ_TILE⌉
      │       │       │
      │       │       └──► InCore: for sb in range(ctx_blocks)  (dynamic loop bound)
      │       │
      │       └──► valid_len = min(SEQ_TILE, ctx_len - s0)
      │               │
      │               ├──► InCore: k_tile / v_tile valid_shape
      │               ├──► InCore: scores_valid = view(scores, [1, valid_len])
      │               └──► InCore: exp_pad zero-padding for matmul
      │
      └── (not used by Scope 1, Scope 3)
```

---

## 3. Prefill Program

### 3.1 Input Contract

| Tensor | Shape | Role |
|--------|-------|------|
| `hidden_states` | `[BATCH, MAX_SEQ, HIDDEN]` | Input tokens, padded to MAX_SEQ on axis 1 |
| `seq_lens` | `[BATCH]`, INT32 | `seq_lens[b]` = actual number of input tokens for session `b` |
| `k_cache` / `v_cache` | `[BATCH * NUM_KV_HEADS * MAX_SEQ, HEAD_DIM]` | Flattened KV cache |
| `rope_cos` / `rope_sin` | `[MAX_SEQ, HEAD_DIM]` | RoPE table |
| `out` | `[BATCH, MAX_SEQ, HIDDEN]` | Output, padded to MAX_SEQ |

### 3.2 Program Structure

```
for b in pl.parallel(BATCH):            ── Orchestration: batch loop
  seq_len_b = read(seq_lens, [b])       ── READ seq_lens
  tok_blocks = ⌈seq_len_b / TOK_TILE⌉
  for p0_idx in pl.range(tok_blocks):   ── Orchestration: dynamic token-tile loop
    p0 = p0_idx * TOK_TILE
    valid_tok = min(TOK_TILE, seq_len_b - p0)
    ├── Scope 1: auto_incore ── RMSNorm + Q/K/V projection
    ├── Scope 2: auto_incore ── RoPE + KV cache + attention
    └── Scope 3: auto_incore ── Output proj + MLP + residual
```

Unlike decode, the prefill program's outer orchestration loop itself is dynamic:
`tok_blocks` varies per session.

### 3.3 Orchestration Level — Outer Loops

```python
for b in pl.parallel(0, BATCH, 1, chunk=4):
    seq_len_b = pl.tensor.read(seq_lens, [b])          # (A) READ
    tok_blocks = (seq_len_b + TOK_TILE - 1) // TOK_TILE # (B) token tile count
    for p0_idx in pl.range(tok_blocks):                  # (C) dynamic loop
        p0 = p0_idx * TOK_TILE                           # (D) token offset
        valid_tok = pl.min(TOK_TILE, seq_len_b - p0)     # (E) tail tile size
```

| Step | Variable | Derivation | Effect |
|------|----------|-----------|--------|
| (A) | `seq_len_b` | `seq_lens[b]` | Runtime per-session token count |
| (B) | `tok_blocks` | `⌈seq_len_b / TOK_TILE⌉` | Outer loop iteration count. Session with 10 tokens → 3 tiles; session with 4096 tokens → 1024 tiles. |
| (C) | `p0_idx` loop | `range(tok_blocks)` | Each iteration processes one token tile |
| (D) | `p0` | `p0_idx * TOK_TILE` | Starting token position for this tile |
| (E) | `valid_tok` | `min(TOK_TILE, seq_len_b - p0)` | On the **last** tile, fewer than TOK_TILE tokens are valid |

### 3.4 Scope 1 — RMSNorm + Q/K/V Projection (InCore)

**Uses `seq_lens` indirectly via `p0` and `valid_tok`.**

```python
with pl.auto_incore():
    for kb in pl.range(HIDDEN_BLOCKS):
        x_chunk = pl.slice(hidden_states, [TOK_TILE, K_CHUNK], [b, p0, k0],
                          valid_shape=[valid_tok, K_CHUNK])     # ← dynamic offset p0
        # ... RMSNorm ...

    q_proj_tile = pl.create_tensor([TOK_TILE, HIDDEN], dtype=BF16,
                                   valid_shape=[valid_tok, HIDDEN])  # ← tail tile awareness
    # ... Q/K/V projections ...
```

| `seq_lens` derivative | Where used | Effect |
|-----------------------|-----------|--------|
| `p0` (offset) | `pl.slice(hidden_states, ..., [b, p0, k0])` | Selects which token tile to read from the padded `[BATCH, MAX_SEQ, HIDDEN]` tensor. |
| `valid_tok` (valid_shape) | `pl.slice(..., valid_shape=[valid_tok, K_CHUNK])` | Annotates that only `valid_tok` rows of the tile contain real data. |
| `valid_tok` (valid_shape) | `pl.create_tensor(..., valid_shape=[valid_tok, ...])` | Projection output tiles track valid rows via `valid_shape`. |

**512-B alignment strategy**: The storage shape is always `[TOK_TILE, ...]` (fixed, aligned).
On the tail tile where `valid_tok < TOK_TILE`, the extra rows read from the
`MAX_SEQ`-padded region of `hidden_states`.  These padding rows are computed through
RMSNorm and projections (producing garbage), but `valid_shape` marks them as invalid.

### 3.5 Scope 2 — RoPE + KV Cache + Attention (InCore)

**Uses `seq_lens` indirectly via `valid_tok`, plus derives `ctx_len` per token.**

```python
with pl.auto_incore():
    attn_tile = pl.create_tensor([TOK_TILE, HIDDEN], dtype=FP32,
                                 valid_shape=[valid_tok, HIDDEN])
    for ti in pl.range(valid_tok):                  # (F) dynamic loop bound
        pos = p0 + ti                                # (G) absolute position
        ctx_len = pos + 1                            # (H) causal context length
        ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE

        # ... RoPE on q/k at position pos ...
        # ... Write k_rot, v to cache at row (b, kvh, pos) ...

        for sb in pl.range(ctx_blocks):              # (I) dynamic inner loop
            valid_len = pl.min(SEQ_TILE, ctx_len - s0)
            k_tile = pl.slice(k_cache, [SEQ_TILE, HEAD_DIM], [cache_row0, 0],
                             valid_shape=[valid_len, HEAD_DIM])
            # ... attention with scores_valid + exp_pad workaround ...
```

| Item | Derivation chain | Effect |
|------|-----------------|--------|
| **(F)** `valid_tok` loop | `min(TOK_TILE, seq_lens[b] - p0)` | Only valid tokens in the tile are processed for RoPE and KV cache write. Prevents garbage from being written into the cache. |
| **(G)** `pos` | `p0 + ti` (both from `seq_lens`) | Absolute sequence position — used for RoPE lookup and cache row calculation. |
| **(H)** `ctx_len` | `pos + 1` | Causal context: token at position `pos` attends to positions `[0, pos]`. |
| **(I)** `ctx_blocks` | `⌈ctx_len / SEQ_TILE⌉` | Grows with each token in the tile: first token (`ti=0`) may attend to `p0+1` entries; last token (`ti=valid_tok-1`) attends to `p0+valid_tok` entries. |

**Critical correctness constraint**: The `for ti in pl.range(valid_tok)` loop ensures that
only valid tokens update the KV cache.  If the full `TOK_TILE` were iterated, padding
tokens would write garbage into cache entries that later tokens (or subsequent decode
steps) would read as valid context.

### 3.6 Scope 3 — Output Projection + MLP + Residual (InCore)

**Uses `seq_lens` indirectly via `valid_tok` and `p0`.**

```python
with pl.auto_incore():
    resid1_tile = pl.create_tensor([TOK_TILE, HIDDEN], dtype=FP32,
                                   valid_shape=[valid_tok, HIDDEN])
    # ... output projection using attn_tile ...
    resid = pl.slice(hidden_states, [TOK_TILE, Q_OUT_CHUNK], [b, p0, o0],
                    valid_shape=[valid_tok, Q_OUT_CHUNK])  # ← residual add
    # ... post-RMSNorm + MLP ...
    out = pl.assemble(out, result_bf16, [b, p0, d0])       # ← output write at (b, p0)
```

| `seq_lens` derivative | Where used | Effect |
|-----------------------|-----------|--------|
| `valid_tok` (valid_shape) | `resid1_tile`, `post_norm_tile`, `down_proj_tile` creation | Marks valid rows for tail tile |
| `p0` (offset) | `pl.slice(hidden_states, ..., [b, p0, o0])` | Reads residual from correct token positions |
| `p0` (offset) | `pl.assemble(out, ..., [b, p0, d0])` | Writes final output to correct positions in padded output tensor |

### 3.7 Prefill `seq_lens` Flow Diagram

```
seq_lens[b]  (GM tensor, INT32)
      │
      ▼
  pl.tensor.read(seq_lens, [b])        ── Orchestration: read scalar
      │
      ├──► seq_len_b = seq_lens[b]
      │       │
      │       ├──► tok_blocks = ⌈seq_len_b / TOK_TILE⌉
      │       │       │
      │       │       └──► Orchestration: for p0_idx in range(tok_blocks)  ← OUTER DYNAMIC LOOP
      │       │
      │       └──► valid_tok = min(TOK_TILE, seq_len_b - p0)
      │               │
      │               ├──► Scope 1 (InCore):
      │               │       ├── hidden_states views: offset [b, p0, k0], valid_shape [valid_tok, ...]
      │               │       └── proj tiles: create_tensor valid_shape [valid_tok, ...]
      │               │
      │               ├──► Scope 2 (InCore):
      │               │       ├── for ti in range(valid_tok)  ← INNER DYNAMIC LOOP
      │               │       │       │
      │               │       │       ├──► pos = p0 + ti
      │               │       │       │       ├── RoPE lookup: rope_cos[pos], rope_sin[pos]
      │               │       │       │       └── KV cache write: cache[b, kvh, pos] = k_rot / v
      │               │       │       │
      │               │       │       ├──► ctx_len = pos + 1   (causal)
      │               │       │       │       ├── ctx_blocks = ⌈ctx_len / SEQ_TILE⌉
      │               │       │       │       │       └── for sb in range(ctx_blocks)
      │               │       │       │       │               └── valid_len = min(SEQ_TILE, ctx_len - s0)
      │               │       │       │       │                       ├── k/v_tile valid_shape
      │               │       │       │       │                       └── scores_valid + exp_pad
      │               │       │       │       └── cache_row0 = f(b, kvh, s0, MAX_SEQ)
      │               │       │       │
      │               │       │       └──► attn_tile[ti, :] = attention output
      │               │       │
      │               │       └── attn_tile valid_shape [valid_tok, HIDDEN]
      │               │
      │               └──► Scope 3 (InCore):
      │                       ├── hidden_states residual view: offset [b, p0, o0], valid_shape [valid_tok, ...]
      │                       ├── intermediate tiles: valid_shape [valid_tok, ...]
      │                       └── output write: out[b, p0, d0] = result
      │
      └── p0 = p0_idx * TOK_TILE  (from tok_blocks loop)
```

---

## 4. Comparison: Decode vs. Prefill

| Aspect | Decode | Prefill |
|--------|--------|---------|
| **Tokens per session** | 1 (fixed) | 1..MAX_SEQ (variable) |
| **`hidden_states` shape** | `[BATCH, HIDDEN]` — no seq axis | `[BATCH, MAX_SEQ, HIDDEN]` — padded |
| **Where `seq_lens` is read** | Scope 2 orchestration | Top-level orchestration |
| **Outer dynamic loop** | None (batch loop has fixed BATCH) | `for p0_idx in range(tok_blocks)` — iteration count varies per session |
| **Scope 1 uses seq_lens?** | No | Yes — `p0` offset and `valid_tok` valid_shape |
| **Scope 2 dynamic loops** | `range(ctx_blocks)` (cache tile loop) | `range(valid_tok)` (token loop) × `range(ctx_blocks)` (cache tile loop) |
| **Scope 3 uses seq_lens?** | No | Yes — `p0` offset and `valid_tok` valid_shape |
| **KV cache positions** | Writes to single `pos` per session | Writes to `pos = p0..p0+valid_tok-1` per tile per session |
| **512-B alignment** | Only KV cache views need tail handling | `hidden_states`, KV cache, and output views all need tail handling |

### 4.1 Dynamic Bounds Summary

| Dynamic variable | Decode source | Prefill source |
|-----------------|--------------|----------------|
| `ctx_blocks` | `⌈seq_lens[b] / SEQ_TILE⌉` | `⌈(p0 + ti + 1) / SEQ_TILE⌉` — grows per token |
| `valid_len` | `min(SEQ_TILE, seq_lens[b] - s0)` | `min(SEQ_TILE, (p0+ti+1) - s0)` |
| `tok_blocks` | N/A (always 1 token) | `⌈seq_lens[b] / TOK_TILE⌉` |
| `valid_tok` | N/A | `min(TOK_TILE, seq_lens[b] - p0)` |

---

## 5. How `valid_shape` Supports Variable `seq_lens`

### 5.1 The Fundamental Problem

Each session `b` in the batch has a different sequence length `seq_lens[b]`, but the
hardware ISA requires every tensor operand's storage to be a multiple of **512 bytes**.
This creates a tension:

```
Session 0: seq_len = 37   → needs 37 tokens processed
Session 1: seq_len = 4096 → needs 4096 tokens processed
Session 2: seq_len = 1    → needs 1 token processed
...
Hardware:  every pl.slice / pl.create_tensor must produce storage ≥ 512 B and aligned
```

The programs resolve this by always using **fixed, aligned tile shapes** (`TOK_TILE`,
`SEQ_TILE`) for storage, even when the last tile contains fewer valid elements.  The
`valid_shape` parameter then tells the compiler exactly which portion of that aligned
storage contains real data derived from `seq_lens`.

### 5.2 The Two-Step Bridge: `seq_lens` → Runtime Bounds → `valid_shape`

The connection from `seq_lens` to `valid_shape` follows a consistent two-step pattern
in both programs:

```
Step 1: Orchestration arithmetic
  seq_lens[b]  ──►  runtime scalars (valid_tok, valid_len, pos, ctx_len, ctx_blocks)

Step 2: valid_shape annotation
  runtime scalars  ──►  valid_shape parameter on pl.slice / pl.create_tensor
```

**Step 1** happens at the orchestration level (host-side control flow), where
`pl.tensor.read(seq_lens, [b])` extracts a per-session scalar and derives all
dynamic bounds:

| Derived scalar | Formula | Meaning |
|---------------|---------|---------|
| `valid_tok` | `min(TOK_TILE, seq_len_b - p0)` | Valid token count in current token tile (prefill only) |
| `valid_len` | `min(SEQ_TILE, ctx_len - s0)` | Valid KV cache entries in current cache tile |
| `pos` | `ctx_len - 1` (decode) or `p0 + ti` (prefill) | Absolute sequence position for RoPE / cache write |
| `ctx_len` | `seq_lens[b]` (decode) or `pos + 1` (prefill) | Causal context length for attention |
| `ctx_blocks` | `⌈ctx_len / SEQ_TILE⌉` | Number of cache tile iterations |
| `tok_blocks` | `⌈seq_len_b / TOK_TILE⌉` | Number of token tile iterations (prefill only) |

**Step 2** happens at the InCore level, where the runtime scalars become `valid_shape`
arguments:

```python
# Token-axis: storage is [TOK_TILE, ...], valid data is [valid_tok, ...]
x_chunk = pl.slice(hidden_states, [TOK_TILE, K_CHUNK], [b, p0, k0],
                  valid_shape=[valid_tok, K_CHUNK])

# Cache-axis: storage is [SEQ_TILE, HEAD_DIM], valid data is [valid_len, HEAD_DIM]
k_tile = pl.slice(k_cache, [SEQ_TILE, HEAD_DIM], [cache_row0, 0],
                 valid_shape=[valid_len, HEAD_DIM])
```

### 5.3 Two Categories of `valid_shape` Usage

The programs use `valid_shape` along two independent axes, each driven by a different
aspect of variable `seq_lens`:

#### Category A: Token-Axis Padding (Prefill Only)

When `seq_len_b` is not a multiple of `TOK_TILE`, the **last token tile** contains
fewer than `TOK_TILE` valid tokens.

```
Example: seq_len_b = 10, TOK_TILE = 4
  Tile 0: p0=0,  valid_tok = min(4, 10-0)  = 4  (full)
  Tile 1: p0=4,  valid_tok = min(4, 10-4)  = 4  (full)
  Tile 2: p0=8,  valid_tok = min(4, 10-8)  = 2  (TAIL — 2 valid, 2 padding)
```

The `valid_shape=[valid_tok, ...]` annotation appears on:

| Operation | Storage shape | valid_shape | Why needed |
|-----------|--------------|-------------|------------|
| `pl.slice(hidden_states, ...)` | `[TOK_TILE, K_CHUNK]` | `[valid_tok, K_CHUNK]` | Marks which input rows are real tokens vs. padding from `MAX_SEQ`-padded storage |
| `pl.create_tensor(q/k/v_proj_tile)` | `[TOK_TILE, HIDDEN/KV_HIDDEN]` | `[valid_tok, ...]` | Projection outputs carry the same valid extent as their inputs |
| `pl.create_tensor(attn_tile)` | `[TOK_TILE, HIDDEN]` | `[valid_tok, HIDDEN]` | Attention output: only `valid_tok` rows have meaningful values |
| `pl.create_tensor(resid1/post_norm/down_proj)` | `[TOK_TILE, HIDDEN]` | `[valid_tok, HIDDEN]` | MLP intermediates track valid rows |
| `pl.slice(hidden_states, ...)` for residual | `[TOK_TILE, Q_OUT_CHUNK]` | `[valid_tok, Q_OUT_CHUNK]` | Residual connection reads valid rows |

**512-B alignment strategy**: The storage shape always uses `TOK_TILE` rows (e.g.,
`TOK_TILE=4` × `K_CHUNK=256` × 2B = 2048B, aligned).  On the tail tile, the extra
rows read from the `MAX_SEQ`-padded region of `hidden_states`, producing garbage that
flows through RMSNorm and projections.  The `valid_shape` marks those rows as invalid,
so downstream operations (especially the `for ti in range(valid_tok)` loop in scope 2)
only process real tokens.

In the decode program, token-axis `valid_shape` is **not needed** because every batch
item contributes exactly one token — there is no sequence axis in
`hidden_states: [BATCH, HIDDEN]`.

#### Category B: Cache-Axis Padding (Both Programs)

When `ctx_len` is not a multiple of `SEQ_TILE`, the **last cache tile** contains fewer
than `SEQ_TILE` valid KV entries.

```
Example (decode): ctx_len = seq_lens[b] = 250, SEQ_TILE = 120
  Tile 0: s0=0,   valid_len = min(120, 250-0)   = 120  (full)
  Tile 1: s0=120, valid_len = min(120, 250-120)  = 120  (full)
  Tile 2: s0=240, valid_len = min(120, 250-240)  = 10   (TAIL — 10 valid, 110 padding)
```

The `valid_shape=[valid_len, HEAD_DIM]` annotation appears on:

| Operation | Storage shape | valid_shape | Why needed |
|-----------|--------------|-------------|------------|
| `pl.slice(k_cache, ...)` | `[SEQ_TILE, HEAD_DIM]` | `[valid_len, HEAD_DIM]` | Only `valid_len` rows are real cached keys; rest is uninitialized or stale |
| `pl.slice(v_cache, ...)` | `[SEQ_TILE, HEAD_DIM]` | `[valid_len, HEAD_DIM]` | Same for cached values |

This is critical for **attention correctness**: the `matmul(q, k^T)` produces
`[1, SEQ_TILE]` scores, but only the first `valid_len` columns correspond to real
cache entries.  Without `valid_shape`, `row_max` and `row_sum` would include garbage
scores from padding entries, corrupting the softmax computation.

### 5.4 The Derivation Chain: `seq_lens[b]` → `valid_shape`

The following diagram shows the complete derivation from `seq_lens` to every
`valid_shape` usage, highlighting the two categories:

```
seq_lens[b]  (GM tensor, INT32)
      │
      ▼
  pl.tensor.read(seq_lens, [b])
      │
      │   ┌──────────────────────────────────────────────────────┐
      │   │  Category A: Token-axis valid_shape (prefill only)   │
      │   │                                                      │
      │   │  seq_len_b ──► tok_blocks = ⌈seq_len_b / TOK_TILE⌉  │
      │   │       │              │                               │
      │   │       │              └──► for p0_idx in range(tok_blocks)
      │   │       │                        │                     │
      │   │       └──► valid_tok = min(TOK_TILE, seq_len_b - p0) │
      │   │                   │                                  │
      │   │                   ├──► pl.slice(hidden_states, [TOK_TILE, ...],  │
      │   │                   │         valid_shape=[valid_tok, ...])       │
      │   │                   ├──► pl.create_tensor([TOK_TILE, ...],       │
      │   │                   │         valid_shape=[valid_tok, ...])       │
      │   │                   └──► for ti in range(valid_tok) ← loop guard │
      │   │                              (prevents cache pollution)         │
      │   └──────────────────────────────────────────────────────┘
      │
      │   ┌──────────────────────────────────────────────────────┐
      │   │  Category B: Cache-axis valid_shape (both programs)  │
      │   │                                                      │
      │   │  ctx_len ──► ctx_blocks = ⌈ctx_len / SEQ_TILE⌉      │
      │   │       │            │                                 │
      │   │       │            └──► for sb in range(ctx_blocks)  │
      │   │       │                       │                      │
      │   │       └──► valid_len = min(SEQ_TILE, ctx_len - s0)   │
      │   │                   │                                  │
      │   │                   ├──► pl.slice(k_cache, [SEQ_TILE, HEAD_DIM], │
      │   │                   │         valid_shape=[valid_len, HEAD_DIM]) │
      │   │                   └──► pl.slice(v_cache, [SEQ_TILE, HEAD_DIM], │
      │   │                             valid_shape=[valid_len, HEAD_DIM]) │
      │   └──────────────────────────────────────────────────────┘
```

Where `ctx_len` is derived differently per program:
- **Decode**: `ctx_len = seq_lens[b]` (total context = all cached tokens including the new one)
- **Prefill**: `ctx_len = pos + 1 = p0 + ti + 1` (causal: each token attends only to preceding tokens)

### 5.5 Concrete Walkthrough: Prefill with `seq_lens[b] = 10`

Configuration: `TOK_TILE = 4`, `SEQ_TILE = 120`, `K_CHUNK = 256`, `HEAD_DIM = 128`.

**Outer loop** produces 3 token tiles (`tok_blocks = ⌈10/4⌉ = 3`):

| `p0_idx` | `p0` | `valid_tok` | Tile type |
|----------|------|-------------|-----------|
| 0 | 0 | 4 | Full |
| 1 | 4 | 4 | Full |
| 2 | 8 | 2 | **Tail** |

**Scope 1** (RMSNorm + projections) for the tail tile (`p0=8, valid_tok=2`):

```python
x_chunk = pl.slice(hidden_states, [4, 256], [b, 8, k0],
                  valid_shape=[2, 256])
#         storage: 4 × 256 × 2B = 2048B  ✓ (512B-aligned)
#         valid:   rows 0-1 = real tokens at positions 8-9
#                  rows 2-3 = padding (read from MAX_SEQ-padded region, produces garbage)

q_proj_tile = pl.create_tensor([4, 5120], dtype=BF16,
                               valid_shape=[2, 5120])
#         storage: 4 × 5120 × 2B = 40960B  ✓
#         valid:   rows 0-1 = projections of real tokens
#                  rows 2-3 = projections of garbage (ignored downstream)
```

**Scope 2** (attention) for the tail tile — the `for ti in range(valid_tok)` loop
runs **only 2 iterations** (ti=0, ti=1), not 4:

| `ti` | `pos` | `ctx_len` | `ctx_blocks` | Cache tile 0 `valid_len` |
|------|-------|-----------|--------------|-------------------------|
| 0 | 8 | 9 | 1 | `min(120, 9) = 9` |
| 1 | 9 | 10 | 1 | `min(120, 10) = 10` |

```python
# ti=1, sb=0:
k_tile = pl.slice(k_cache, [120, 128], [cache_row0, 0],
                 valid_shape=[10, 128])
#         storage: 120 × 128 × 2B = 30720B  ✓
#         valid:   rows 0-9 = cached keys for positions 0-9
#                  rows 10-119 = stale/uninitialized (excluded by valid_shape)
```

**Scope 3** (output projection + MLP) for the tail tile:

```python
resid = pl.slice(hidden_states, [4, 64], [b, 8, o0],
                valid_shape=[2, 64])
#         Only valid_tok=2 rows participate in residual addition

out = pl.assemble(out, result_bf16, [b, 8, d0])
#         Writes to positions 8-11 in the padded [BATCH, MAX_SEQ, HIDDEN] output.
#         Rows 2-3 (positions 10-11) are garbage but fall in the padding zone
#         that the caller ignores (because the caller also knows seq_lens[b]=10).
```

### 5.6 Concrete Walkthrough: Decode with `seq_lens[b] = 250`

Configuration: `SEQ_TILE = 120`, `HEAD_DIM = 128`.

Decode has no token-axis variability (1 token per session), so only cache-axis
`valid_shape` is used.

```python
ctx_len = 250        # = seq_lens[b]
pos = 249            # = ctx_len - 1
ctx_blocks = 3       # = ⌈250 / 120⌉
```

| `sb` | `s0` | `valid_len` | `k_tile` / `v_tile` storage | `valid_shape` |
|------|------|-------------|---------------------------|---------------|
| 0 | 0 | 120 | `[120, 128]` (30720B ✓) | `[120, 128]` (full) |
| 1 | 120 | 120 | `[120, 128]` (30720B ✓) | `[120, 128]` (full) |
| 2 | 240 | 10 | `[120, 128]` (30720B ✓) | **`[10, 128]`** (tail) |

On tile 2, the `matmul(q, k^T)` produces `scores: [1, 120]`, but only columns 0–9
hold real dot products.  The `valid_shape=[10, 128]` on `k_tile` marks this boundary.

### 5.7 `valid_shape` Propagation Through Compute Operations

Once `valid_shape` is set on the input tiles, it propagates through compute
operations according to the rules in `tensor_valid_shape.md`:

| Compute Op | valid_shape Rule | Example in these programs |
|-----------|-----------------|--------------------------|
| `pl.matmul(A, B)` | `[A.vs[0], B.vs[1]]` | `matmul(q[1,128], k_tile[120,128]^T)` → `scores.vs = [1, valid_len]` (when compiler supports propagation) |
| `pl.row_max(X)` | `[X.vs[0], 1]` | Only valid rows/columns contribute to the max |
| `pl.row_sum(X)` | `[X.vs[0], 1]` | Only valid entries summed |
| `pl.add(A, B)` | `[min(A.vs[i], B.vs[i])]` | Residual add: `valid_shape` intersection preserves tail info |
| `pl.assemble(T, S, offset)` | `T.vs` (target extent unchanged) | Accumulation into `attn_tile` / `out` preserves the target's valid extent |
| `pl.cast(X)` | `X.vs` | Type conversion preserves valid extent |

### 5.8 Annotation Inventory

#### Decode Annotations

| Location | Operation | Storage Shape | valid_shape | Derived from |
|----------|----------|---------------|-------------|-------------|
| Scope 2, cache read | `pl.slice(k_cache, ...)` | `[SEQ_TILE, HEAD_DIM]` | `[valid_len, HEAD_DIM]` | `min(SEQ_TILE, seq_lens[b] - s0)` |
| Scope 2, cache read | `pl.slice(v_cache, ...)` | `[SEQ_TILE, HEAD_DIM]` | `[valid_len, HEAD_DIM]` | `min(SEQ_TILE, seq_lens[b] - s0)` |

#### Prefill Annotations

| Location | Operation | Storage Shape | valid_shape | Derived from |
|----------|----------|---------------|-------------|-------------|
| Scope 1, input read | `pl.slice(hidden_states, ...)` | `[TOK_TILE, K_CHUNK]` | `[valid_tok, K_CHUNK]` | `min(TOK_TILE, seq_lens[b] - p0)` |
| Scope 1, proj tiles | `pl.create_tensor(...)` | `[TOK_TILE, HIDDEN/KV_HIDDEN]` | `[valid_tok, ...]` | `min(TOK_TILE, seq_lens[b] - p0)` |
| Scope 2, attn tile | `pl.create_tensor(...)` | `[TOK_TILE, HIDDEN]` | `[valid_tok, HIDDEN]` | `min(TOK_TILE, seq_lens[b] - p0)` |
| Scope 2, cache read | `pl.slice(k_cache, ...)` | `[SEQ_TILE, HEAD_DIM]` | `[valid_len, HEAD_DIM]` | `min(SEQ_TILE, (p0+ti+1) - s0)` |
| Scope 2, cache read | `pl.slice(v_cache, ...)` | `[SEQ_TILE, HEAD_DIM]` | `[valid_len, HEAD_DIM]` | `min(SEQ_TILE, (p0+ti+1) - s0)` |
| Scope 3, residual | `pl.slice(hidden_states, ...)` | `[TOK_TILE, Q_OUT_CHUNK]` | `[valid_tok, Q_OUT_CHUNK]` | `min(TOK_TILE, seq_lens[b] - p0)` |
| Scope 3, intermediates | `pl.create_tensor(...)` | `[TOK_TILE, HIDDEN]` | `[valid_tok, HIDDEN]` | `min(TOK_TILE, seq_lens[b] - p0)` |

### 5.9 Correctness Guarantees Provided by `valid_shape`

`valid_shape` ensures three key correctness properties when `seq_lens` varies:

1. **No KV cache pollution** — In prefill scope 2, the `for ti in range(valid_tok)`
   loop bound (derived from `seq_lens`) prevents padding tokens from writing garbage
   into the KV cache.  Without this, subsequent decode steps would attend to garbage
   entries.  The `valid_shape` on `k_proj_tile` / `v_proj_tile` serves as the semantic
   annotation that enables this guard.

2. **Correct attention masking** — The `valid_shape=[valid_len, HEAD_DIM]` on cache
   tiles ensures that only real cached entries participate in attention score
   computation.  On the last cache tile, `row_max` / `row_sum` (once compiler support
   is complete) will automatically ignore padding columns, preventing softmax
   corruption.

3. **Safe output writes** — On the prefill tail tile, `pl.assemble(out, ..., [b, p0, d0])`
   writes `TOK_TILE` rows including garbage in padding positions.  These fall into the
   `MAX_SEQ`-padded region of `out` that the caller ignores (the caller also has
   `seq_lens[b]` and knows to read only `seq_lens[b]` positions).  The `valid_shape`
   annotation on intermediates documents this contract.

### 5.10 Current Workaround (Until Compiler Fully Propagates `valid_shape`)

In the attention loop of **both** programs, the compiler does not yet propagate
`valid_shape` from `k_tile` / `v_tile` through `matmul` to the output `scores`.
Ideally:

```python
# Future (compiler propagates valid_shape):
scores = pl.matmul(q, k_tile, b_trans=True)   # k_tile.vs = [valid_len, HEAD_DIM]
# → scores.vs auto-deduced as [1, valid_len]
# → row_max(scores) only considers valid_len columns ← automatic
# → row_sum(exp(scores)) only sums valid_len columns  ← automatic
```

Today, this requires **two manual workaround steps**:

1. **Extract valid scores**: `scores_valid = pl.slice(scores, [1, valid_len], [0, 0])`
   — creates a view with `valid_len` columns so `row_max` / `row_sum` operate only on
   real scores.

2. **Zero-pad for matmul**: `exp_pad = zeros → assemble(exp_scores, [0,0])` — pads
   `exp(scores)` back to `[1, SEQ_TILE]` for the `matmul` with `v_tile`, ensuring
   padding positions contribute `0 × v = 0` to the weighted sum.

Once the `ConvertTensorToBlockOps` pass forwards tensor-level `valid_shape` to
`block.load valid_shapes`, `row_max` and `row_sum` will automatically operate
on valid columns only, and both workarounds can be removed.  The TODO markers in
the source code (`TODO(valid_shape)`) identify these sites.
