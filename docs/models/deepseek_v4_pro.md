# DeepSeek V4-Pro

`models/deepseek_v4_pro/` is the Ascend 950 (A5) variant: the same V4 operator
set and compositions as [V4-Flash](deepseek_v4_flash_mtp.md), built for the
larger **DeepSeek-V4-Pro** checkpoint.

## Deployment configuration

The `PRO` preset in [config.py](../../models/deepseek_v4_pro/config.py) mirrors
the HuggingFace model's `config.json`; the kernels import `PRO_KERNEL`, which
is the same architecture at a shorter sequence budget.

| Deployment property | Value |
| --- | --- |
| Speculative decoding | MTP = 1 (`DECODE_SEQ = 2`) |
| Decode batch per card | 4 requests → 8 token rows per step |
| Decode context length | up to 16,384 positions (`KERNEL_MAX_SEQ_LEN`), 128-token pages |
| Prefill shape | one request of 128 tokens per program |
| Platform | Ascend A5 (`-p a5`); the parsers also accept A2/A3 and both simulators |
| Expert parallelism | `--ep 2/4/8`, `384 / ep` routed experts per rank |
| LM-head parallelism | `--tp 2/4/8` vocab shards; `LM_HEAD_TP_SIZE = 8` is the deployment value |
| Other components | no tensor parallelism — attention is data-parallel, MoE is expert-parallel |
| Quantization | Hybrid MXFP8-MXFP4 — MXFP8 for the dense path, MXFP4 for the routed-expert weights |
| Serving | none — no `pypto-serving` deployment consumes this tree |

`PRO.max_position_embeddings` keeps the architectural one-million-position
value, but admitting 1 M positions would need a ~64× larger physical pool than
the cases allocate and a host-side golden nobody can compute. `PRO_KERNEL`
therefore replaces it with `KERNEL_MAX_SEQ_LEN = 16384` — an 8k prompt plus 512
decode steps, the budget the Flash cases already exercise. Raise that one
constant if a case needs a longer context.

Native MXFP8-MXFP4 is not implemented yet. The tracked kernels run an INT8
stand-in with the same tensor split as
[V4-Flash](deepseek_v4_flash_mtp.md#what-is-quantized): `gen_routed_weight` in
[expert_routed.py](../../models/deepseek_v4_pro/expert_routed.py) re-quantizes
off the MXFP4 grid into INT8 rather than feeding the cube MXFP4 weights.

### Model shape and layer schedule

Pro is wider and deeper than Flash: 7168 hidden, 128 attention heads, a 1536
Q-LoRA rank, 16 output-projection groups, `moe_intermediate_size = 3072`, 384
routed experts with top-6 routing plus 1 shared expert, and an indexer top-k of
1024. `compress_ratios` carries 62 entries — 61 model layers plus the MTP
layer:

| Ratio | Path | Layers | Count |
| ---: | --- | --- | ---: |
| 128 | HCA — ratio-128 compressor, deterministic top-k | 0, 1, 3, 5, …, 59 | 31 |
| 4 | CSA — ratio-4 compressor plus the learned indexer | 2, 4, …, 60 | 30 |
| 0 | SWA — sliding window only | the MTP layer | 1 |

Unlike Flash, the main stack has no SWA layer; SWA appears only in the MTP
layer, which is why `decode_attention_swa` is reachable from `decode_mtp` but
not from `decode_fwd`. The first three layers route by hash
(`num_hash_layers = 3`).

## Model structure, top down

### `decode_fwd` and `prefill_fwd`

Both hand-unroll the layer schedule inside one rank-generic `@pl.jit` kernel
launched per EP rank, with every attention and MoE stage in its own
`pl.scope()`:

```
decode_fwd     layers 0, 1        decode_attention_hca → moe
               loop over pairs    decode_attention_csa → moe   (even layers)
                                  decode_attention_hca → moe   (odd layers)
               layer 60           decode_attention_csa → moe
               tail               hc_head → rms_norm

prefill_fwd    same schedule with prefill_attention_{hca,csa} → moe,
               tail               hc_head → rms_norm
```

Both forwards stop at the final norm: unlike the Flash tree, the LM head is not
part of them. [lm_head.py](../../models/deepseek_v4_pro/lm_head.py) is a
standalone distributed harness here.

### One layer

```
attention   hc_pre → rmsnorm → qkv_proj_rope → (compress / index) → sparse_attn → hc_post
moe         hc_pre → gate → expert_shared → dispatch → expert_routed → combine → hc_post
```

`decode_layer` and `prefill_layer` expose exactly this pair as standalone
two-rank harnesses.

### Attention paths

```
decode_attention_swa   hc_pre → rmsnorm → qkv_proj_rope
                              → decode_sparse_attn_swa                   → hc_post
decode_attention_hca   hc_pre → rmsnorm → qkv_proj_rope
                              → decode_compressor_ratio128
                              → decode_sparse_attn_hca                   → hc_post
decode_attention_csa   hc_pre → rmsnorm → qkv_proj_rope
                              → decode_compressor_ratio4 (main + inner)
                              → decode_indexer → decode_indexer_compressor
                              → decode_sparse_attn                       → hc_post

prefill_attention_swa  hc_pre/hc_post + rmsnorm + qkv_proj_rope + prefill_sparse_attn
prefill_attention_hca  … + prefill_compressor_ratio128
prefill_attention_csa  … + prefill_compressor_ratio4
                          + prefill_indexer → prefill_indexer_compressor
```

`decode_sparse_attn` is the CSA variant here (the Flash tree names it
`decode_sparse_attn_csa`); all three own the fused grouped output projection.
`rope_tables` generates the RoPE/YaRN tables on the host and `decode_metadata`
lowers the fixture's paged-cache metadata.

### MTP path

```
mtp_projection  e_proj(enorm(hidden)) + h_proj(hnorm(prev_hidden))
decode_mtp      mtp_projection → decode_attention_swa → moe → hc_head → rmsnorm
prefill_mtp     mtp_projection → prefill_attention_swa → moe → hc_head → rmsnorm
```

`prefill_mtp` reuses `prefill_fwd`'s driver for the main-model pass.

## Files

| Group | Files |
| --- | --- |
| Full forward | [decode_fwd.py](../../models/deepseek_v4_pro/decode_fwd.py), [prefill_fwd.py](../../models/deepseek_v4_pro/prefill_fwd.py) |
| Layer composition | [decode_layer.py](../../models/deepseek_v4_pro/decode_layer.py), [prefill_layer.py](../../models/deepseek_v4_pro/prefill_layer.py) |
| MTP | [decode_mtp.py](../../models/deepseek_v4_pro/decode_mtp.py), [prefill_mtp.py](../../models/deepseek_v4_pro/prefill_mtp.py), [mtp_projection.py](../../models/deepseek_v4_pro/mtp_projection.py) |
| Decode attention orchestration | [decode_attention_swa.py](../../models/deepseek_v4_pro/decode_attention_swa.py), [decode_attention_csa.py](../../models/deepseek_v4_pro/decode_attention_csa.py), [decode_attention_hca.py](../../models/deepseek_v4_pro/decode_attention_hca.py) |
| Decode sparse attention (fused o-proj) | [decode_sparse_attn.py](../../models/deepseek_v4_pro/decode_sparse_attn.py), [decode_sparse_attn_swa.py](../../models/deepseek_v4_pro/decode_sparse_attn_swa.py), [decode_sparse_attn_hca.py](../../models/deepseek_v4_pro/decode_sparse_attn_hca.py) |
| Decode compressors and indexer | [decode_compressor_ratio4.py](../../models/deepseek_v4_pro/decode_compressor_ratio4.py), [decode_compressor_ratio128.py](../../models/deepseek_v4_pro/decode_compressor_ratio128.py), [decode_indexer.py](../../models/deepseek_v4_pro/decode_indexer.py), [decode_indexer_compressor.py](../../models/deepseek_v4_pro/decode_indexer_compressor.py) |
| Prefill attention and cache | [prefill_attention_swa.py](../../models/deepseek_v4_pro/prefill_attention_swa.py), [prefill_attention_csa.py](../../models/deepseek_v4_pro/prefill_attention_csa.py), [prefill_attention_hca.py](../../models/deepseek_v4_pro/prefill_attention_hca.py), [prefill_sparse_attn.py](../../models/deepseek_v4_pro/prefill_sparse_attn.py), [prefill_compressor_ratio4.py](../../models/deepseek_v4_pro/prefill_compressor_ratio4.py), [prefill_compressor_ratio128.py](../../models/deepseek_v4_pro/prefill_compressor_ratio128.py), [prefill_indexer.py](../../models/deepseek_v4_pro/prefill_indexer.py), [prefill_indexer_compressor.py](../../models/deepseek_v4_pro/prefill_indexer_compressor.py) |
| Shared transforms | [rmsnorm.py](../../models/deepseek_v4_pro/rmsnorm.py), [qkv_proj_rope.py](../../models/deepseek_v4_pro/qkv_proj_rope.py), [hc_pre.py](../../models/deepseek_v4_pro/hc_pre.py), [hc_post.py](../../models/deepseek_v4_pro/hc_post.py), [hc_head.py](../../models/deepseek_v4_pro/hc_head.py) |
| MoE and output | [moe.py](../../models/deepseek_v4_pro/moe.py), [gate.py](../../models/deepseek_v4_pro/gate.py), [expert_shared.py](../../models/deepseek_v4_pro/expert_shared.py), [expert_routed.py](../../models/deepseek_v4_pro/expert_routed.py), [lm_head.py](../../models/deepseek_v4_pro/lm_head.py) |
| Metadata and host helpers | [config.py](../../models/deepseek_v4_pro/config.py), [decode_metadata.py](../../models/deepseek_v4_pro/decode_metadata.py), [rope_tables.py](../../models/deepseek_v4_pro/rope_tables.py) |

`config.py`, `decode_metadata.py`, and `rope_tables.py` have no `__main__`
block and are imported rather than run. Which entry points CI schedules is
defined by the [daily model workflow](../../.github/workflows/daily_ci.yml).
