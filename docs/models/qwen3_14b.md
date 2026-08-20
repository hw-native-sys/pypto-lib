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
| Platform | Ascend A2/A3; the native PyPTO BF16 decode path also has A2/A3-sim compile coverage |
| Precision, main path | BF16 weights and KV cache, FP32 inter-layer residual carry, FP32 RMSNorm weights |
| Precision, A8W8 path | INT8 weights with per-token INT8 activations, in the `*_a8w8` entries only |
| Serving | [contract.py](../../models/qwen3_14b/contract.py) registers the BF16 prefill and decode stages for `pypto-serving` |

`batch_pad` is a throughput knob, not a capacity limit: a wider pad does more
rows per weight read, while public batches above 16 are split into row windows
by the device-side `decode_fwd`. Each window reuses the native Page Attention
scratch tensors through explicit task dependencies and re-reads the weights.

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
                     → BSND KV cache write → native PyPTO paged attention
                     → output projection → post-attention RMSNorm
                     → SwiGLU MLP → FP32 residual
              rms_lm_head → _greedy_sample_inline → sampled_ids_out
```

The complete attention stage is generated from
[paged_attention_pypto.py](../../models/qwen3_14b/paged_attention_pypto.py).
Its Phase 0 applies Q/K norm and RoPE, then appends K/V to the paged BSND cache;
the following mixed AIC/AIV task computes ragged GQA Page Attention. The public
ABI remains vLLM-compatible: Q/O are active TND and the flat paged K/V buffers
are ordered `[page, token, kv_head, dim]`. The former hand-written CCE extern,
generated Phase-0 header, runtime tiler, and migration selector have been
removed, so production decode has one implementation path.

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
| Page Attention | [paged_attention_pypto.py](../../models/qwen3_14b/paged_attention_pypto.py), [test_paged_attention_pypto.py](../../models/qwen3_14b/test_paged_attention_pypto.py) |
| Output and sampling | [rms_lm_head.py](../../models/qwen3_14b/rms_lm_head.py), [greedy_sample.py](../../models/qwen3_14b/greedy_sample.py), [topk_select.py](../../models/qwen3_14b/topk_select.py) |
| Configuration and serving | [constants.py](../../models/qwen3_14b/constants.py), [config.py](../../models/qwen3_14b/config.py), [weights.py](../../models/qwen3_14b/weights.py), [contract.py](../../models/qwen3_14b/contract.py) |
| Drafts (excluded from CI) | [decode_ssn_draft.py](../../models/qwen3_14b/decode_ssn_draft.py), [decode_tq_draft.py](../../models/qwen3_14b/decode_tq_draft.py), [prefill_tq_draft.py](../../models/qwen3_14b/prefill_tq_draft.py) |

`constants.py`, `config.py`, `contract.py`, `weights.py`, `rms_lm_head.py`,
`paged_attention_pypto.py`, `turboquant_kv.py`, and `prefill_fwd_a8w8.py` have no
`__main__` block and are imported rather than run. Which entry points CI
schedules is defined by the
[daily model workflow](../../.github/workflows/daily_ci.yml).

## Validation

The Page Attention component driver provides focused deterministic Torch-oracle
cases, focused and full pairwise matrix presets, wrapper-codegen checks, and
optional raw benchmark recording. Performance reports are informational and
never gate the production implementation.

Run the component PR matrix and a B17 production-decode golden (one full window
plus a one-row tail window) on an A2/A3 device with:

```bash
python models/qwen3_14b/test_paged_attention_pypto.py \
  -p a2a3 -d "$DEVICE_ID" --matrix pr \
  --matrix-json /tmp/qwen3-pa-matrix.json

python models/qwen3_14b/decode_fwd.py \
  -p a2a3 -d "$DEVICE_ID" --validate-fwd --fwd-layers 1 -b 17 \
  --seq-lens 1,2,127,128,129,255,256,257,511,512,513,1023,2048,3584,4095,4096,129
```

Repository CI routes PA changes through the generic A2/A3 component smoke and
single-layer decode golden, then runs real-weight `pypto-serving` accuracy
coverage on A2/A3. The larger matrix commands above remain available for
focused validation.
