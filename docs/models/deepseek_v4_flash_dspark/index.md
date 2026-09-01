# DeepSeek V4-Flash, DSpark point

`models/deepseek_v4_flash_dspark/` is the second deployment point of the same
V4-Flash checkpoint: a wide-batch serving configuration whose speculation comes
from a **DSpark drafter** instead of one MTP layer, and whose attention is
tensor-parallel and context-parallel instead of purely data-parallel.

The model body — 43 layers, the three attention paths, the 256-expert MoE, the
hyper-connection stack — is the one described on the
[V4-Flash MTP page](../deepseek_v4_flash_mtp/index.md), and both trees read the
same `FLASH` preset. This page covers what the DSpark point changes.

## Deployment configuration

[config.py](../../../models/deepseek_v4_flash_dspark/config.py) carries the same
`DeepSeekV4Config` presets as the MTP tree and its own deployment constants
below the presets.

| Deployment property | Value |
| --- | --- |
| Speculative decoding | DSpark — a three-layer drafter proposes `DSPARK_SPEC_TOKENS = 7` drafts per request, so the target model verifies `S = 8` token rows per step (`DECODE_SEQ`) |
| Decode batch per card | 64 requests → 512 token rows per step (`DECODE_BATCH`, `DECODE_TOKENS`) |
| Decode context length | up to 1,048,576 positions, paged in **32-token** pages (`max_position_embeddings`, `BLOCK_SIZE`) |
| Prefill shape | one packed request stream per CP group, `PREFILL_SEQ = 512` tokens per dispatch; longer prompts arrive as chunks against a resident prefix |
| Platform | Ascend A2/A3, single node |
| Tensor parallelism | `--tp 1/2/4`; the deployment point is TP 4 — the grouped output projection and the LM head are vocab/group-sharded over it |
| Context parallelism | DSA-CP reuses that same physical TP group: each rank owns a slice of the step's token rows, and the attention KV stream is replicated rank-major across the group |
| Expert parallelism | `--ep 2/4/8/16`; the deployment point is EP 16, and each rank holds `256 / ep` routed experts |
| Data parallelism | `DP = 4` groups per node, so the deployment point is 16 cards (`TP * DP`) |
| Quantization | W8A8 INT8, identical to the MTP tree — INT8 weights with FP32 dequant scales, activations quantized per token at the INT8 matmuls |

`BLOCK_SIZE = 32` sets four page sizes at once: the paged KV cache, the
compressed KV cache, the indexer cache, and — through
`C4A_COMPRESSOR_BLOCK_SIZE = 2` and `C128_COMPRESSOR_BLOCK_SIZE = 8` — both
compressor-state pools.

### What DSpark changes

| | MTP point | DSpark point |
| --- | --- | --- |
| Drafting | one MTP layer, 1 draft token | a 3-layer drafter, 7 draft tokens, plus a Markov head |
| Rows per decode step, per card | 4 requests × 2 = 8 | 64 requests × 8 = 512 |
| Attention parallelism | data-parallel; each rank owns its own micro-batch | TP-sharded output projection over a DSA-CP token split |
| Page size | 128 | 32 |
| Context ceiling | `max_position_embeddings` truncated to 16,384 | the checkpoint's own 1,048,576, and the cache capacities are sized from it |

The wider verify window is the reason for the rest of the table: 512 rows per
step is too much attention work for one card, so the token axis is split across
the CP group and the output projection is sharded along with it.

## Model structure, top down

### `decode_fwd`

[decode_fwd.py](../../../models/deepseek_v4_flash_dspark/decode_fwd.py) hand-unrolls
the 43-layer schedule inside one rank-generic `@pl.jit` kernel, launched per
rank from an `@pl.jit.host` driver — the same shape as the MTP tree's forward,
with each attention and MoE stage in its own `pl.scope()` under
`auto_scope=False`:

```
decode_fwd
├── preamble          embedding lookup, metadata lowering, CP token all-gather
├── layers 0, 1       decode_swa  → moe
├── loop ×20          decode_csa  → moe        (layers 2, 4, …, 40)
│                     decode_hca  → moe        (layers 3, 5, …, 41)
├── layer 42          decode_csa  → moe
└── tail              hc_head → rms_norm → lm_head (TP vocab shard)
```

Every `decode_{swa,csa,hca}` entry has a `_tp1` twin: the single-rank form runs
the layer without the CP gather and the TP publish, and is what the golden
compares against. The batch is dynamic: `--start-pos` takes one position per
request and its length is the batch, defaulting to the 16 per-rank requests the
MoE token budget is sized for.

### `prefill_fwd`

[prefill_fwd.py](../../../models/deepseek_v4_flash_dspark/prefill_fwd.py) mirrors
that structure for a packed prompt: the same per-rank kernel shape, the same
per-stage scopes, `prefill_{swa,hca,csa}` in place of the decode
orchestrations, and the same `hc_head → rms_norm → lm_head` tail. Prompt
sequence lengths that do not divide the CP group are padded rather than
rejected, so one program serves an arbitrary chunk against a resident prefix.

### Attention under DSA-CP

The three attention paths are the MTP tree's, re-cut along the token axis:

```
decode_swa   hc_pre → rmsnorm → qkv_proj_rope → decode_sparse_attn_swa
                    → decode_o_proj (TP publish)                      → hc_post
decode_hca   … → decode_compressor_ratio128 → decode_sparse_attn_hca  → …
decode_csa   … → decode_compressor_ratio4 (main, inner)
                 → decode_indexer → decode_indexer_compressor
                 → decode_sparse_attn_csa                             → …
```

- [decode_cp_token_allgather.py](../../../models/deepseek_v4_flash_dspark/decode_cp_token_allgather.py)
  gathers the CP group's token rows into rank-major order on **every** rank.
  Each rank then writes the group's whole KV stream into its own replicated
  cache, so a compressor or indexer sees the full context while its queries stay
  on their token owner.
- [decode_o_proj.py](../../../models/deepseek_v4_flash_dspark/decode_o_proj.py)
  owns the grouped output projection and its TP communication: each rank
  dequantizes and projects its own `o_groups` shard, then publishes the result
  to the group so every rank leaves the stage with the complete rows.
- The prefill side is the same decomposition over
  [prefill_cp_token_allgather.py](../../../models/deepseek_v4_flash_dspark/prefill_cp_token_allgather.py)
  and [prefill_o_proj.py](../../../models/deepseek_v4_flash_dspark/prefill_o_proj.py).

### MoE and output stages

[moe.py](../../../models/deepseek_v4_flash_dspark/moe.py) is unchanged in shape
from the MTP tree — `gate` produces the top-6 routing and the per-token INT8
view, `dispatch` / `combine` are the EP collectives, `expert_shared` and
`expert_routed` are the two FFN paths — but it carries `DP * DECODE_TOKENS`
worth of receive capacity, because a DSpark step dispatches 512 rows per card
rather than 8.

`hc_head` folds the hyper-connection stack back to one hidden row, the final
`rms_norm` normalizes it, and [lm_head.py](../../../models/deepseek_v4_flash_dspark/lm_head.py)
all-gathers the group's hidden rows, projects them against this card's
`vocab / tp` shard, and all-to-alls the logits back to their row owners.

### The DSpark drafter

The drafter is a small model of its own, run after the target step over the
target's hidden states:

```
dspark_proj        main_proj(concat of 3 target layers' hidden) → RMSNorm
dspark_context_kv  project the target's token stream into each draft layer's
                   paged SWA cache (per proposal, decode rows or prompt chunk)
dspark_drafter     ×3  hc_pre → rmsnorm → qkv_proj_rope
                          → dspark_attention → o_proj publish → hc_post
                          → moe
markov_head        low-rank (256) Markov embedding + full-vocabulary logits
dspark_markov      lm_head → sequential Markov sampling → confidence head
```

- [dspark_proj.py](../../../models/deepseek_v4_flash_dspark/dspark_proj.py)
  collapses three target layers' hidden states (`dspark_target_layer_ids`) into
  one drafter hidden row. `main_proj` stays BF16: the W8A8 checkpoint quantizes
  it only under an FP8 quant method.
- [dspark_attention.py](../../../models/deepseek_v4_flash_dspark/dspark_attention.py)
  runs one anchor-first draft query block of 7 rows per request against the
  paged sliding window. Every draft row sees the trailing window plus the whole
  block through one index list, so there is no causal mask inside the block.
- [dspark_markov.py](../../../models/deepseek_v4_flash_dspark/dspark_markov.py)
  emits the 7 drafts sequentially — each step's sampled id feeds the next
  through a rank-256 Markov transition — and a sigmoid confidence head scores
  the block for the acceptance policy.
- [dspark_prefill.py](../../../models/deepseek_v4_flash_dspark/dspark_prefill.py)
  is the drafter's prefill entry: prompt-context KV insertion followed by the
  same seven-query proposal.

The drafter and the target forward are compiled and validated as separate
programs; there is no single entry composing a full target-plus-draft serving
step yet.

## Status

Under development, and not wired into `pypto-serving`. Every executable file
carries its own Golden Harness fixture and CI markers, and the daily model
workflow sweeps the directory like any other model tree; the
`decode_fwd` / `prefill_fwd` / `decode_layer` / `prefill_layer` compositions and
the distributed communication oracles are device-only (`ci: no-sim`).

```bash
python models/deepseek_v4_flash_dspark/decode_layer.py -p a2a3 --tp 2 --ep 2 -d 0,1
python models/deepseek_v4_flash_dspark/decode_fwd.py -p a2a3 --tp 2 --ep 2 -d 0,1
python models/deepseek_v4_flash_dspark/dspark_drafter.py -p a2a3 --tp 4 --ep 4 -d 0,1,2,3
```

`--tp` and `--ep` are read at import time, because the shapes they derive
freeze before the kernels are traced; passing a value the module did not import
with is rejected rather than silently ignored.

## Files

| Group | Files |
| --- | --- |
| Full forward | [decode_fwd.py](../../../models/deepseek_v4_flash_dspark/decode_fwd.py), [prefill_fwd.py](../../../models/deepseek_v4_flash_dspark/prefill_fwd.py) |
| Layer composition | [decode_layer.py](../../../models/deepseek_v4_flash_dspark/decode_layer.py), [prefill_layer.py](../../../models/deepseek_v4_flash_dspark/prefill_layer.py) |
| DSpark drafter | [dspark_drafter.py](../../../models/deepseek_v4_flash_dspark/dspark_drafter.py), [dspark_prefill.py](../../../models/deepseek_v4_flash_dspark/dspark_prefill.py), [dspark_proj.py](../../../models/deepseek_v4_flash_dspark/dspark_proj.py), [dspark_attention.py](../../../models/deepseek_v4_flash_dspark/dspark_attention.py), [dspark_context_kv.py](../../../models/deepseek_v4_flash_dspark/dspark_context_kv.py) |
| DSpark sampling | [dspark_markov.py](../../../models/deepseek_v4_flash_dspark/dspark_markov.py), [markov_head.py](../../../models/deepseek_v4_flash_dspark/markov_head.py) |
| Decode attention orchestration | [decode_swa.py](../../../models/deepseek_v4_flash_dspark/decode_swa.py), [decode_csa.py](../../../models/deepseek_v4_flash_dspark/decode_csa.py), [decode_hca.py](../../../models/deepseek_v4_flash_dspark/decode_hca.py) |
| Decode sparse attention | [decode_sparse_attn_swa.py](../../../models/deepseek_v4_flash_dspark/decode_sparse_attn_swa.py), [decode_sparse_attn_csa.py](../../../models/deepseek_v4_flash_dspark/decode_sparse_attn_csa.py), [decode_sparse_attn_hca.py](../../../models/deepseek_v4_flash_dspark/decode_sparse_attn_hca.py) |
| Decode compressors and indexer | [decode_compressor_ratio4.py](../../../models/deepseek_v4_flash_dspark/decode_compressor_ratio4.py), [decode_compressor_ratio128.py](../../../models/deepseek_v4_flash_dspark/decode_compressor_ratio128.py), [decode_indexer.py](../../../models/deepseek_v4_flash_dspark/decode_indexer.py), [decode_indexer_compressor.py](../../../models/deepseek_v4_flash_dspark/decode_indexer_compressor.py) |
| Prefill attention and cache | [prefill_swa.py](../../../models/deepseek_v4_flash_dspark/prefill_swa.py), [prefill_csa.py](../../../models/deepseek_v4_flash_dspark/prefill_csa.py), [prefill_hca.py](../../../models/deepseek_v4_flash_dspark/prefill_hca.py), [prefill_sparse_attn.py](../../../models/deepseek_v4_flash_dspark/prefill_sparse_attn.py), [prefill_compressor_ratio4.py](../../../models/deepseek_v4_flash_dspark/prefill_compressor_ratio4.py), [prefill_compressor_ratio128.py](../../../models/deepseek_v4_flash_dspark/prefill_compressor_ratio128.py), [prefill_indexer.py](../../../models/deepseek_v4_flash_dspark/prefill_indexer.py), [prefill_indexer_compressor.py](../../../models/deepseek_v4_flash_dspark/prefill_indexer_compressor.py) |
| Output projection and CP transport | [decode_o_proj.py](../../../models/deepseek_v4_flash_dspark/decode_o_proj.py), [prefill_o_proj.py](../../../models/deepseek_v4_flash_dspark/prefill_o_proj.py), [decode_cp_token_allgather.py](../../../models/deepseek_v4_flash_dspark/decode_cp_token_allgather.py), [prefill_cp_token_allgather.py](../../../models/deepseek_v4_flash_dspark/prefill_cp_token_allgather.py) |
| Shared transforms | [rmsnorm.py](../../../models/deepseek_v4_flash_dspark/rmsnorm.py), [qkv_proj_rope.py](../../../models/deepseek_v4_flash_dspark/qkv_proj_rope.py), [hc_pre.py](../../../models/deepseek_v4_flash_dspark/hc_pre.py), [hc_post.py](../../../models/deepseek_v4_flash_dspark/hc_post.py), [hc_head.py](../../../models/deepseek_v4_flash_dspark/hc_head.py), [rope_interleave.py](../../../models/deepseek_v4_flash_dspark/rope_interleave.py), [lookup_embedding.py](../../../models/deepseek_v4_flash_dspark/lookup_embedding.py) |
| MoE and output | [moe.py](../../../models/deepseek_v4_flash_dspark/moe.py), [gate.py](../../../models/deepseek_v4_flash_dspark/gate.py), [expert_shared.py](../../../models/deepseek_v4_flash_dspark/expert_shared.py), [expert_routed.py](../../../models/deepseek_v4_flash_dspark/expert_routed.py), [lm_head.py](../../../models/deepseek_v4_flash_dspark/lm_head.py) |
| Metadata and host helpers | [decode_metadata.py](../../../models/deepseek_v4_flash_dspark/decode_metadata.py), [prefill_metadata.py](../../../models/deepseek_v4_flash_dspark/prefill_metadata.py), [config.py](../../../models/deepseek_v4_flash_dspark/config.py), [utils.py](../../../models/deepseek_v4_flash_dspark/utils.py) |

`config.py`, `utils.py`, `rope_interleave.py`, and `prefill_o_proj.py` have no
`__main__` block: they are imported rather than run.
