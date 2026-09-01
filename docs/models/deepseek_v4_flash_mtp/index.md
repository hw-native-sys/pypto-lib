# DeepSeek V4-Flash, MTP point

`models/deepseek_v4_flash_mtp/` is the reference V4-Flash tree: the operators,
the single-layer compositions, and the prefill/decode full forwards.
[V4-Flash DSpark](../deepseek_v4_flash_dspark.md) is the same checkpoint at a
second deployment point — wider batch, seven draft tokens, context-parallel
attention.

## Deployment configuration

The tree implements the HuggingFace **DeepSeek-V4-Flash** checkpoint — the
`FLASH` preset in
[config.py](../../../models/deepseek_v4_flash_mtp/config.py) mirrors that model's
`config.json` field for field, and `config.py` is a per-directory singleton
that every kernel imports as a bare sibling module.

| Deployment property | Value |
| --- | --- |
| Speculative decoding | MTP = 1 — one draft token verified against the previous one, so a decode step carries `S = 2` token rows per request |
| Decode batch per card | 4 requests → 8 token rows per step (`DECODE_BATCH`, `DECODE_SEQ`) |
| Decode context length | up to 16,384 positions, paged in 128-token pages (`max_position_embeddings`, `BLOCK_SIZE`) |
| Prefill shape | one request per rank partition, each with up to 8,192 active tokens per dispatch; the program walks the dynamic extent in 128-token tiles (`PREFILL_BATCH`, `PREFILL_SEQ`) |
| Platform | Ascend A2/A3, single node |
| Expert parallelism | `--ep 2/4/8`; the deployment point is EP 8, and each rank holds `256 / ep` routed experts |
| LM-head parallelism | `--tp 2/4/8/16` vocab shards over DP row owners, `--tp <= --ep` |
| Other components | no tensor parallelism — attention is data-parallel (each rank owns its own decode micro-batch) and the MoE is expert-parallel |
| Quantization | W8A8 INT8: INT8 weights with FP32 dequant scales, activations quantized per token at the INT8 matmuls |

### What is quantized

Activations are quantized **dynamically per token**: each row's amax (floored
by `INT8_AMAX_EPS = 1e-4`) is rescaled to `INT8_SCALE_MAX = 127`, so no
calibration data or static activation scale is carried. `gate` produces the
per-token INT8 view once and both the shared expert and the dispatch payload
reuse it.

| Tensors | Storage |
| --- | --- |
| Q up-projection `wq_b`, output projection `wo_b`, indexer Q projection `csa_idx_wq_b`, MoE `routed_w{1,2,3}` and `shared_w{1,2,3}` | **INT8** weights, each with an FP32 per-output-channel `*_scale` |
| Q down-projection `wq_a`, KV projection `wkv`, `wo_a`, compressor `*_cmp_wkv` / `*_cmp_wgate` / `csa_inner_*`, `csa_weights_proj`, `csa_hadamard_idx`, token embedding, `lm_head_weight`, every RMSNorm gamma, RoPE `freqs_cos/sin` | BF16 |
| Hyper-connection projections, scales and bases, `attn_sink`, router `gate_w` / `gate_bias`, the APE tables, and all dequant scales | FP32 |
| Original and compressed KV caches (`kv_cache`, `cmp_kv`) | BF16 |
| Indexer KV cache `idx_kv_cache` | INT8, quantized on write, with an FP32 per-row `idx_kv_scale` |
| Compressor states (`hca_compress_state`, `csa_compress_state`, `csa_inner_compress_state`) | FP32 |
| Activations | per-token INT8 into the INT8 matmuls; the inter-layer hyper-connection hidden state stays FP32; `x_out` is BF16 and logits are FP32 |

The precision fields the preset carries are metadata copied from the model
card. The tracked kernels consume the INT8 layout above; each harness's tensor
specs and golden function remain the authority.

### Layer schedule

`compress_ratios` assigns an attention path per layer. The tuple carries 44
entries: the 43 model layers plus the MTP layer.

| Ratio | Path | Layers | Count |
| ---: | --- | --- | ---: |
| 0 | SWA — sliding window (128) only; no compressor, no indexer, no YaRN scaling | 0, 1, and the MTP layer | 2 + 1 |
| 4 | CSA — ratio-4 overlapping compressor plus the learned indexer (top-512) | 2, 4, …, 42 | 21 |
| 128 | HCA — ratio-128 non-overlapping compressor, deterministic top-k | 3, 5, …, 41 | 20 |

Every layer pairs its attention stage with one MoE stage: 1 shared expert plus
top-6 of 256 routed experts, `moe_intermediate_size = 2048`. The first three
layers route by hash (`num_hash_layers = 3`) rather than by gate score. The
hyper-connection stack is 4 streams wide (`hc_mult = 4`).

## Model structure, top down

### `decode_fwd`

[decode_fwd.py](../../../models/deepseek_v4_flash_mtp/decode_fwd.py) hand-unrolls
the layer schedule inside one rank-generic `@pl.jit` kernel, launched once per
EP rank from an `@pl.jit.host` driver:

```
decode_fwd
├── layers 0, 1       decode_swa  → moe
├── loop ×20          decode_csa  → moe        (layers 2, 4, …, 40)
│                     decode_hca  → moe        (layers 3, 5, …, 41)
├── layer 42          decode_csa  → moe
└── tail              hc_head → rms_norm → lm_head_with_sampling
```

Each attention and each MoE stage runs in its own `pl.scope()` under
`auto_scope=False`. The paged pools (`kv_cache`, `cmp_kv`, `idx_kv_cache`, the
three compressor states) are passed in flat and sliced per layer.

### `prefill_fwd`

[prefill_fwd.py](../../../models/deepseek_v4_flash_mtp/prefill_fwd.py) mirrors
that structure for a packed prompt: the same per-rank kernel shape, the same
per-stage scopes, `prefill_{swa,hca,csa}` in place of the decode
orchestrations, and the same `hc_head → rms_norm → lm_head` tail over selected
hidden rows.

### `decode_fwd_mtp`

[decode_fwd_mtp.py](../../../models/deepseek_v4_flash_mtp/decode_fwd_mtp.py) is
the third top-level composition: it chains the main decode forward, the draft
verification, and the MTP decode layer into one serving step. Its device-only
CLI fixture composes the standalone forward and MTP tensor fixtures with a
persistent recurrent-state pool. Daily CI runs the default EP2/TP2 fixture on
two devices; component-level golden checks remain with the standalone paths.

### One layer

A layer is an attention stage followed by a MoE stage, both wrapped in
hyper-connection mixing:

```
attention   hc_pre → rmsnorm → qkv_proj_rope → (compress / index) → sparse_attn → hc_post
moe         hc_pre → gate → expert_shared → dispatch → expert_routed → combine → hc_post
```

`hc_pre` mixes the four hyper-connection streams into one hidden row (RMS,
sigmoid gates, a Sinkhorn-normalized combine matrix); `hc_post` folds the
sublayer output back into the stack. `decode_layer` and `prefill_layer` are
exactly this pair exposed as standalone two-rank harnesses.

### Attention paths

The three paths share the skeleton and differ in what sits between the
projection and the sparse attention:

```
decode_swa   hc_pre → rmsnorm → qkv_proj_rope → decode_sparse_attn_swa → hc_post
decode_hca   hc_pre → rmsnorm → qkv_proj_rope
                    → decode_compressor_ratio128
                    → decode_sparse_attn_hca                     → hc_post
decode_csa   hc_pre → rmsnorm → qkv_proj_rope
                    → decode_compressor_ratio4  (main, rotate=False)
                    → decode_compressor_ratio4  (inner, rotate=True)
                    → decode_indexer → decode_indexer_compressor
                    → decode_sparse_attn_csa                     → hc_post
```

- `rmsnorm` and `qkv_proj_rope` (Q/KV LoRA projections plus RoPE) are
  dynamic-shape and shared by decode and prefill.
- The `decode_sparse_attn_*` kernels own the fused grouped output projection.
  SWA sees only the sliding window; HCA takes its compressed top-k from a
  deterministic index computation; CSA takes it from the learned indexer.
- The prefill side is the same decomposition: `prefill_swa` /`prefill_hca` /
  `prefill_csa` over `prefill_sparse_attn`, `prefill_compressor_ratio{4,128}`,
  and `prefill_indexer` → `prefill_indexer_compressor`.

### MoE stage

[moe.py](../../../models/deepseek_v4_flash_mtp/moe.py) is one distributed
single-layer program that `decode_fwd`, `prefill_fwd`, the layer harnesses, and
the MTP entries all call. `gate` is RMSNorm + router + top-k + normalize and
also produces the per-token INT8 view; `dispatch` and `combine` are the EP
collectives (per-source lanes with folded notifies); `expert_shared` and
`expert_routed` are the two FFN paths.

### Output stage

`hc_head` projects the hyper-connection stack back to one hidden row, the final
`rms_norm` normalizes it, and `lm_head` all-gathers hidden rows across the DP
owners, projects them against this card's vocab shard, then all-to-alls the
logits so each owner ends with its own rows over the full vocabulary. Greedy
sampling is fused into the same program.

### MTP path

```
mtp_projection   e_proj(enorm(hidden)) + h_proj(hnorm(prev_hidden))
decode_mtp       lookup_embedding → mtp_projection → decode_swa → moe
                 → hc_head → rmsnorm → lm_head
decode_fwd_mtp   decode_fwd → verify_and_pack_mtp_tokens → decode_mtp
prefill_mtp      mtp_projection → prefill_swa → moe → hc_head → rmsnorm → lm_head
```

`decode_fwd_mtp` holds the persistent MTP serving state inline: it loads each
request's previous tail/draft, checks the draft against the main-model sample,
packs the committed window, and commits the result back to the same slot. It
also owns the device-side preamble for both halves: metadata lowering and input
packing before the main layers, embedding lookup and MTP hidden packing before
the draft layer. `decode_fwd` and `decode_mtp` therefore cover the model body
alone and take the preamble's results as inputs, which their fixtures build in
torch. `decode_prepare` lowers the packed input
IDs and the paged-cache metadata on device; `utils` is its host-side torch
counterpart used by the test fixtures.

## How it was optimized

[DeepSeek V4 decode optimization](decode_optimization.md)
follows the tuning of this decode path in order — contracts and golden first,
then the general tiling / parallelism / fusion levers, the attention,
hyper-connection, MoE and router rewrites, scheduling, and finally
serving-level residency and lowering — with the limit measured at each step.

## Files

| Group | Files |
| --- | --- |
| Full forward | [decode_fwd.py](../../../models/deepseek_v4_flash_mtp/decode_fwd.py), [prefill_fwd.py](../../../models/deepseek_v4_flash_mtp/prefill_fwd.py), [decode_fwd_mtp.py](../../../models/deepseek_v4_flash_mtp/decode_fwd_mtp.py) |
| Layer composition | [decode_layer.py](../../../models/deepseek_v4_flash_mtp/decode_layer.py), [prefill_layer.py](../../../models/deepseek_v4_flash_mtp/prefill_layer.py) |
| MTP | [decode_mtp.py](../../../models/deepseek_v4_flash_mtp/decode_mtp.py), [prefill_mtp.py](../../../models/deepseek_v4_flash_mtp/prefill_mtp.py), [mtp_projection.py](../../../models/deepseek_v4_flash_mtp/mtp_projection.py) |
| Decode attention orchestration | [decode_swa.py](../../../models/deepseek_v4_flash_mtp/decode_swa.py), [decode_csa.py](../../../models/deepseek_v4_flash_mtp/decode_csa.py), [decode_hca.py](../../../models/deepseek_v4_flash_mtp/decode_hca.py) |
| Decode sparse attention (fused o-proj) | [decode_sparse_attn_swa.py](../../../models/deepseek_v4_flash_mtp/decode_sparse_attn_swa.py), [decode_sparse_attn_csa.py](../../../models/deepseek_v4_flash_mtp/decode_sparse_attn_csa.py), [decode_sparse_attn_hca.py](../../../models/deepseek_v4_flash_mtp/decode_sparse_attn_hca.py) |
| Decode compressors and indexer | [decode_compressor_ratio4.py](../../../models/deepseek_v4_flash_mtp/decode_compressor_ratio4.py), [decode_compressor_ratio128.py](../../../models/deepseek_v4_flash_mtp/decode_compressor_ratio128.py), [decode_indexer.py](../../../models/deepseek_v4_flash_mtp/decode_indexer.py), [decode_indexer_compressor.py](../../../models/deepseek_v4_flash_mtp/decode_indexer_compressor.py) |
| Prefill attention and cache | [prefill_swa.py](../../../models/deepseek_v4_flash_mtp/prefill_swa.py), [prefill_csa.py](../../../models/deepseek_v4_flash_mtp/prefill_csa.py), [prefill_hca.py](../../../models/deepseek_v4_flash_mtp/prefill_hca.py), [prefill_sparse_attn.py](../../../models/deepseek_v4_flash_mtp/prefill_sparse_attn.py), [prefill_compressor_ratio4.py](../../../models/deepseek_v4_flash_mtp/prefill_compressor_ratio4.py), [prefill_compressor_ratio128.py](../../../models/deepseek_v4_flash_mtp/prefill_compressor_ratio128.py), [prefill_indexer.py](../../../models/deepseek_v4_flash_mtp/prefill_indexer.py), [prefill_indexer_compressor.py](../../../models/deepseek_v4_flash_mtp/prefill_indexer_compressor.py) |
| Shared transforms | [rmsnorm.py](../../../models/deepseek_v4_flash_mtp/rmsnorm.py), [qkv_proj_rope.py](../../../models/deepseek_v4_flash_mtp/qkv_proj_rope.py), [hc_pre.py](../../../models/deepseek_v4_flash_mtp/hc_pre.py), [hc_post.py](../../../models/deepseek_v4_flash_mtp/hc_post.py), [hc_head.py](../../../models/deepseek_v4_flash_mtp/hc_head.py), [rope_interleave.py](../../../models/deepseek_v4_flash_mtp/rope_interleave.py), [lookup_embedding.py](../../../models/deepseek_v4_flash_mtp/lookup_embedding.py) |
| MoE and output | [moe.py](../../../models/deepseek_v4_flash_mtp/moe.py), [gate.py](../../../models/deepseek_v4_flash_mtp/gate.py), [expert_shared.py](../../../models/deepseek_v4_flash_mtp/expert_shared.py), [expert_routed.py](../../../models/deepseek_v4_flash_mtp/expert_routed.py), [lm_head.py](../../../models/deepseek_v4_flash_mtp/lm_head.py) |
| Metadata and host helpers | [decode_prepare.py](../../../models/deepseek_v4_flash_mtp/decode_prepare.py), [config.py](../../../models/deepseek_v4_flash_mtp/config.py), [utils.py](../../../models/deepseek_v4_flash_mtp/utils.py) |

`config.py`, `utils.py`, `rope_interleave.py`, and `decode_prepare.py` have
no `__main__` block: they are imported rather than run. Executable compositions,
including `decode_fwd_mtp.py`, are scheduled by the
[daily model workflow](../../../.github/workflows/daily_ci.yml).
