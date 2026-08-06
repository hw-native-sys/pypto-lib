# Qwen3-14B

`models/qwen3_14b/` implements the HuggingFace **Qwen3-14B** checkpoint: a BF16
prefill and decode pair with the serving contract, plus A8W8 and TurboQuant
variants and the sampling components. Together with
[V4-Flash MTP](deepseek_v4_flash_mtp.md) it is one of the two trees wired up
for full `pypto-serving` integration.

## Deployment configuration

[constants.py](../../models/qwen3_14b/constants.py) is the single source of the
model shape and of every constant that is part of the external ABI;
[config.py](../../models/qwen3_14b/config.py) adds the `pl.dynamic` dimensions
the kernel signatures bind.

| Deployment property | Value |
| --- | --- |
| Layers / heads | 40 layers, 40 attention heads over 8 KV heads (GQA), `head_dim = 128` |
| Hidden / MLP | 5120 hidden, 17,408 intermediate |
| Vocabulary | 152,064 padded, 151,936 real |
| Context length | up to 4096 positions, paged in 128-token pages (`seq_tile`) |
| Decode batch | the pipeline is padded to **16 rows** (`batch_pad`); any public batch ≥ 1 runs as `ceil(batch / 16)` row windows |
| Parallelism | single card — no TP, no EP, no DP |
| Platform | Ascend A2/A3; the BF16 decode attention is A2/A3-only because it calls a CANN operator |
| Precision, main path | BF16 weights and KV cache, FP32 inter-layer residual carry, FP32 RMSNorm weights |
| Precision, A8W8 path | INT8 weights with per-token INT8 activations, in the `*_a8w8` entries only |
| Serving | [contract.py](../../models/qwen3_14b/contract.py) registers the BF16 prefill and decode stages for `pypto-serving` |

`batch_pad` is a throughput knob, not a capacity limit: a wider pad does more
rows per weight read, but raising it means bumping `kMaxBatch` and the metadata
length arrays on the CCE side, regenerating the RoPE body, and re-validating
the M-dimension tiling. Past 16 rows correctness scales and cost does not
amortize — windows serialize on the single paged-attention metadata/workspace
pair and re-read the weights.

## Model structure, top down

### `prefill_fwd`

[prefill_fwd.py](../../models/qwen3_14b/prefill_fwd.py) loops the same fused
layer body over all 40 layer rows of the flattened weight tensors:

```
prefill_fwd   per layer   input RMSNorm → Q/K/V projection → RoPE
                          → KV cache update → causal attention
                          → output projection → post-attention RMSNorm
                          → SwiGLU MLP → residual
              tail        rms_lm_head (final RMSNorm + LM head)
```

Every batch-dependent signature dim is a `pl.dynamic` variable
(`BATCH_DYN`, `PREFILL_TOKENS_DYN`, `KV_CACHE_ROWS_DYN`,
`BLOCK_TABLE_FLAT_DYN`), so one compiled program serves any batch that fits the
host KV cache. Inputs are packed token-major (`T = sum(chunk_lens)`, no
`[batch, max_seq]` padding), the embedding is gathered on device, and hidden
state lifetime is bounded by processing 128-token windows.

### `decode_fwd`

[decode_fwd.py](../../models/qwen3_14b/decode_fwd.py) is a single fused
device-side step:

```
decode_fwd    _token_embed_inline (embed the previously sampled id)
              ×40  _decode_layer
                     RMSNorm → QKV projection → Q/K norm → RoPE
                     → KV cache write
                     → paged_attention_cce  (CANN FusedInferAttentionScore)
                     → output projection → post-attention RMSNorm
                     → SwiGLU MLP → FP32 residual
              rms_lm_head → _greedy_sample_inline → sampled_ids_out
```

The attention stage is the one part that is not pypto-generated:
[paged_attention_cce.py](../../models/qwen3_14b/paged_attention_cce.py) binds
the hand-written CCE kernel under
[kernels/paged_attention_cce/](../../models/qwen3_14b/kernels/paged_attention_cce/)
through `pl.jit.extern`. Its public ABI matches vLLM — Q/O are active TND and
the flat paged K/V buffers hold BSND bytes ordered `[page, token, kv_head,
dim]`. The kernel embeds a copied pypto/ptoas codegen artifact,
`kernel/rope_qkv_generated.hpp`, as its in-kernel phase 0;
[rope_qkv_regen.py](../../models/qwen3_14b/rope_qkv_regen.py) is the standalone
program that regenerates that header — it is compile-only and never reaches a
device.

`decode_fwd_layers` in the same file is the same fused body over a contiguous
layer *chunk* with no LM head, for callers that compose the stack externally.

### Quantized and compressed variants

```
decode_layer_a8w8   one A8W8 decode layer (INT8 weights, per-token INT8 activations)
prefill_fwd_a8w8    the A8W8 full-layer prefill, imported rather than run directly
prefill_tq_draft    prefill_layer_tq → turboquant_kv_quantize (PolarQuant)
                                     → turboquant_qjl_k       (QJL, K only)
decode_tq_draft     the TurboQuant decode counterpart, built on decode_fwd
decode_ssn_draft    serial 4D-blocked single-layer decode
```

[turboquant_kv.py](../../models/qwen3_14b/turboquant_kv.py) holds the Lloyd-Max
codebook computation, the prefill KV quantization, and the QJL K-residual
quantization. The three `*_draft.py` files are work in progress and excluded
from CI.

### Sampling and output

`rms_lm_head` (final RMSNorm plus the LM-head projection, in vocab chunks of
512) is shared by both forwards. [greedy_sample.py](../../models/qwen3_14b/greedy_sample.py)
and [topk_select.py](../../models/qwen3_14b/topk_select.py) are the standalone
sampling components; `decode_fwd` inlines its own greedy sample so a step
produces the next token without a host round trip.

### Serving glue

[weights.py](../../models/qwen3_14b/weights.py) prepares the kernel-ready
weight layout and [contract.py](../../models/qwen3_14b/contract.py) — colocated
with the entry points it names — registers the prefill, decode, and
greedy-sample stages, their compile-time argument builders, and the ABI
constants an external runtime needs.

## Files

| Group | Files |
| --- | --- |
| Forwards | [decode_fwd.py](../../models/qwen3_14b/decode_fwd.py), [prefill_fwd.py](../../models/qwen3_14b/prefill_fwd.py) |
| Quantized variants | [decode_layer_a8w8.py](../../models/qwen3_14b/decode_layer_a8w8.py), [prefill_fwd_a8w8.py](../../models/qwen3_14b/prefill_fwd_a8w8.py), [turboquant_kv.py](../../models/qwen3_14b/turboquant_kv.py) |
| CCE attention | [paged_attention_cce.py](../../models/qwen3_14b/paged_attention_cce.py), [kernels/paged_attention_cce/](../../models/qwen3_14b/kernels/paged_attention_cce/), [rope_qkv_regen.py](../../models/qwen3_14b/rope_qkv_regen.py), [test_paged_attention_cce.py](../../models/qwen3_14b/test_paged_attention_cce.py) |
| Output and sampling | [rms_lm_head.py](../../models/qwen3_14b/rms_lm_head.py), [greedy_sample.py](../../models/qwen3_14b/greedy_sample.py), [topk_select.py](../../models/qwen3_14b/topk_select.py) |
| Configuration and serving | [constants.py](../../models/qwen3_14b/constants.py), [config.py](../../models/qwen3_14b/config.py), [weights.py](../../models/qwen3_14b/weights.py), [contract.py](../../models/qwen3_14b/contract.py) |
| Drafts (excluded from CI) | [decode_ssn_draft.py](../../models/qwen3_14b/decode_ssn_draft.py), [decode_tq_draft.py](../../models/qwen3_14b/decode_tq_draft.py), [prefill_tq_draft.py](../../models/qwen3_14b/prefill_tq_draft.py) |

`constants.py`, `config.py`, `contract.py`, `weights.py`, `rms_lm_head.py`,
`paged_attention_cce.py`, `turboquant_kv.py`, and `prefill_fwd_a8w8.py` have no
`__main__` block and are imported rather than run. Which entry points CI
schedules is defined by the
[daily model workflow](../../.github/workflows/daily_ci.yml).
