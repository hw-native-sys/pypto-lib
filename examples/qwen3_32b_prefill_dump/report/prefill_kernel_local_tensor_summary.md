# Qwen3 32B Prefill — Kernel Local Tensor SRAM Summary

## 1) Overview

Single-layer prefill forward for Qwen3 32B with **variable-length input** per
session.  Each batch item can have a different input sequence length (up to
4096), passed via the `seq_lens` input tensor (`[BATCH], INT32`).  Tensors are
padded to `MAX_SEQ` on the sequence axis; only valid tokens are processed.

## 2) Model Config

- `BATCH=16`, `MAX_SEQ=4096`, `HIDDEN=5120`
- `NUM_HEADS=64`, `NUM_KV_HEADS=8`, `HEAD_DIM=128`
- `INTERMEDIATE=25600`

## 3) Tuning Knobs

- `K_CHUNK=256`
- `Q_OUT_CHUNK=64`
- `KV_OUT_CHUNK=32`
- `SEQ_TILE=120`
- `MLP_OUT_CHUNK=256`
- `TOK_TILE=4`

## 4) Variable Sequence Length Support

- `seq_lens: Tensor[[BATCH], INT32]` — per-session input token count.
- Token iteration: `tok_blocks = ceil(seq_len_b / TOK_TILE)`.
- **Scope 1 & 3**: Always use full `[TOK_TILE, ...]` storage shapes from GM
  tensors (512-B aligned); padding rows in the tail tile map to
  allocated-but-unused MAX_SEQ slots.
- **Scope 2** (attention + KV cache write): iterates only over `valid_tok`
  tokens (`for ti in pl.range(valid_tok)`) to avoid writing garbage into
  the KV cache.  Padding rows in `attn_tile` stay zero; scope 3 writes them
  to the padding area of `out` which the caller ignores.

### `valid_shape` Integration (per `tensor_valid_shape.md` design)

**GM tensor views** carry explicit `valid_shape` annotations:

| Location | Storage Shape | valid_shape | Purpose |
|---|---|---|---|
| `hidden_states` views (Scope 1 & 3) | `[TOK_TILE, K_CHUNK]` | `[valid_tok, K_CHUNK]` | Tail token tile |
| KV-cache views (Scope 2) | `[SEQ_TILE, HEAD_DIM]` | `[valid_len, HEAD_DIM]` | Tail cache tile |
| `hidden_states` residual view (Scope 3) | `[TOK_TILE, Q_OUT_CHUNK]` | `[valid_tok, Q_OUT_CHUNK]` | Tail token tile |

**Local tensor creation** with `valid_shape`:

| Tensor | Storage Shape | valid_shape |
|---|---|---|
| `q_proj_tile` | `[TOK_TILE, HIDDEN]` | `[valid_tok, HIDDEN]` |
| `k_proj_tile` / `v_proj_tile` | `[TOK_TILE, KV_HIDDEN]` | `[valid_tok, KV_HIDDEN]` |
| `attn_tile` | `[TOK_TILE, HIDDEN]` | `[valid_tok, HIDDEN]` |
| `resid1_tile` | `[TOK_TILE, HIDDEN]` | `[valid_tok, HIDDEN]` |
| `post_norm_tile` | `[TOK_TILE, HIDDEN]` | `[valid_tok, HIDDEN]` |
| `down_proj_tile` | `[TOK_TILE, HIDDEN]` | `[valid_tok, HIDDEN]` |

**Current workaround** (until the compiler propagates `valid_shape`):
`scores_valid = pl.slice(scores, [1, valid_len], ...)` + zero-padded `exp_pad`
are used to mask garbage scores from padding cache rows in the attention loop.
Once the compiler's `ConvertTensorToBlockOps` pass forwards tensor-level
`valid_shape` to `block.load valid_shapes`, these workarounds can be removed.

## 5) Function-Level Statistics

| InCore Function | Local Tensor Size (B) | Buffers |
|---|---:|---:|
| `qwen3_prefill_layer_incore_2_aic` | 248,256 | 17 |
| `qwen3_prefill_layer_incore_0_aic` | 140,288 | 13 |
| `qwen3_prefill_layer_incore_1_aic` | 140,288 | 21 |
| `qwen3_prefill_layer_incore_3_aic` | 140,288 | 13 |
| `qwen3_prefill_layer_incore_4_aic` | 132,096 | 8 |
| `qwen3_prefill_layer_incore_3_aiv` | 104,960 | 11 |
| `qwen3_prefill_layer_incore_4_aiv` | 101,376 | 9 |
| `qwen3_prefill_layer_incore_0_aiv` | 72,704 | 13 |
| `qwen3_prefill_layer_incore_2_aiv` | 57,696 | 41 |
| `qwen3_prefill_layer_incore_1_aiv` | 48,128 | 20 |
| **Total** | **1,186,080** | - |

## 6) Group-Level Statistics (AIC / AIV split)

| Logical Kernel | AIC (B) | AIV (B) | Solo (B) |
|---|---:|---:|---:|
| `qwen3_prefill_layer_incore_2` | 248,256 | 57,696 | 0 |
| `qwen3_prefill_layer_incore_0` | 140,288 | 72,704 | 0 |
| `qwen3_prefill_layer_incore_1` | 140,288 | 48,128 | 0 |
| `qwen3_prefill_layer_incore_3` | 140,288 | 104,960 | 0 |
| `qwen3_prefill_layer_incore_4` | 132,096 | 101,376 | 0 |

## 7) Constraint Check

- **AIC 256KB limit**: PASS (max AIC = `248,256 B` < 262,144)
- **AIV 192KB limit**: PASS (max AIV = `104,960 B` < 196,608)

## 8) Tuning Notes

- `qwen3_prefill_layer_incore_4_aic` uplifted to `132,096 B` (from `17,408 B`
  baseline) by increasing `MLP_OUT_CHUNK` to 256.
- Current smallest item: `qwen3_prefill_layer_incore_1_aiv` (`48,128 B`).
