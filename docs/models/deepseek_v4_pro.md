# DeepSeek V4-Pro

`models/deepseek_v4_pro/` is the Ascend 950 (A5) implementation for DeepSeek-V4
Pro, with an optional Flash architecture preset. Select the preset at import
and compile time with `DEEPSEEK_V4_VARIANT=pro|flash` or `--variant pro|flash`.

## Deployment configuration

The `PRO` and `FLASH` presets in [config.py](../../models/deepseek_v4_pro/config.py)
define the architecture-specific shapes and layer schedules. Pro remains the
default so existing operator entry points and DailyCI keep their prior behavior.

| Deployment property | Value |
| --- | --- |
| Speculative decoding | MTP = 1 (`DECODE_SEQ = 2`) |
| Decode batch per card | 4 requests → 8 token rows per step |
| Decode context length | up to 16,384 positions (`KERNEL_MAX_SEQ_LEN`), 128-token pages |
| Prefill shape | one request of 128 tokens per program |
| Platform | Ascend A5 (`-p a5`); full forwards are device-only |
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

Both forwards finish with the final norm and LM-head sampling. The standalone
[lm_head.py](../../models/deepseek_v4_pro/lm_head.py) entry point validates that
distributed tail separately.

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

## Real weights (Flash)

`weights_flash.py` converts the released DeepSeek-V4-Flash checkpoint (hybrid
MXFP4 routed experts + block-FP8 attention/shared-expert linears) into the
host-tensor ABI of the two forward drivers: FP4/FP8 tensors are dequantized
and re-quantized to the kernels' INT8 + per-output-channel FP32-scale form,
per-layer tensors are stacked and EP/TP-sharded exactly like the fixture
specs. Convert once offline, then point the drivers at the cache:

```bash
python models/deepseek_v4_pro/utils/weights_flash.py --variant flash --ep 8 --tp 2 \
    --ckpt /path/to/DeepSeek-V4-Flash --out build_output/flash_weights_ep8_tp2
python models/deepseek_v4_pro/prefill_fwd.py --variant flash --ep 8 --tp 2 \
    -p a5 -d 0,1,2,3,4,5,6,7 --weights build_output/flash_weights_ep8_tp2
```

`--weights` also accepts the raw checkpoint directory (converted on the fly;
slower and RAM-hungry — the cache is the recommended path). Only EP8 deploys
the full 256-expert model: the kernel programs keep 32 local experts per rank
(`moe.py` shrinks the global routing space to `32*EP`), so an EP4/EP2
real-weight run uses the first `32*EP` checkpoint experts with reduced router
tables — a smoke configuration, not the true model output.

Numeric validation on real weights:

- `decode_layer.py` / `prefill_layer.py` accept `--weights <ckpt_dir>` to
  inject one layer's real weights (converted on demand); the layer golden then
  recomputes with the same weights, so the existing per-layer validation runs
  on real dynamic ranges.
- `prefill_fwd.py --validate` enables a full-network torch golden
  (`golden_fwd.py`: embed → 43 chained layer goldens → hc_head → final norm →
  LM head). End-of-network gates are cosine/rel-L2 on the selected logit rows
  plus greedy-sample agreement; per-element gates on deep hidden states and
  compressor state pools accumulate cross-layer drift and are expected to
  need looser budgets than the single-layer drivers. RoPE tables, the
indexer Hadamard, caches, and per-step metadata keep their fixture
initializers. The drivers stay smoke-only (`golden_fn=None`): a real-weight
run validates that the network executes with real dynamic ranges and produces
finite logits/sensible tokens, not a golden comparison.

Golden data can be computed once and replayed: `prefill_fwd.py --validate
--save-data` persists the generated inputs and golden outputs under
`<runtime_dir>/data/`, and `--golden-data <dir>` loads them back and runs
only the device pass plus the comparison — the CPU-heavy golden compute can
run on a host without NPU access while the short device pass reuses it.
`--prompt-file <file> --tokenizer <tokenizer.json>` replaces the synthetic
`input_ids` with a real prompt (replicated across ranks; `num_tokens`
follows the prompt length).

### End-to-end token generation

[synthetic_token_loop.py](../../models/deepseek_v4_pro/synthetic_token_loop.py)
drives the full prompt-to-text path on real weights: the prompt is encoded
with the checkpoint's `tokenizer.json` (BOS prepended unless `--no-bos`),
the resident session runs one prefill plus `--decode-steps` greedy decode
steps, every step asserts that all ranks sampled the same token, and the
sampled ids are detokenized at the end. Decoding stops early when
`--eos-id` (default 1) is sampled. Without `--weights` the loop keeps its
synthetic zero-weight control-path behavior.

The EP8 example carries the two workaround flags the caveats below explain
(`--prefill-tokens 16 --prefill-no-retire`); at EP2 neither is needed:

```bash
python models/deepseek_v4_pro/synthetic_token_loop.py --variant flash \
    --ep 8 --tp 2 -d 0,1,2,3,4,5,6,7 \
    --prefill-tokens 16 --prefill-no-retire \
    --weights build_output/flash_weights_ep8_tp2 \
    --tokenizer /path/to/DeepSeek-V4-Flash/tokenizer.json \
    --prompt "The capital of France is" --decode-steps 32
```

Two EP8 caveats, pending a proper fix:

- The `moe_signal_retire` scope can deadlock an EP8 dispatch
  (`SCHEDULER_TIMEOUT`): its single-element anchor orders the negative
  credits only after the task writing `pre_hc_hidden_out[0, 0, 0]`, so they
  can land while later waits of the same dispatch are still pending. The
  trace-time knob `DSV4_DISABLE_MOE_RETIRE=1` compiles a forward without
  the scope; the loop's `--prefill-no-retire` sets it for the prefill
  compile only. Prefill's un-retired credits did not disturb the following
  decode steps in testing, while multi-step decode does require its own
  retirement (without it, decode produced non-finite logits on part of the
  ranks by the second step) — so decode keeps the scope.
- An EP8 prefill compiled at `--num-tokens 128` stalls on-device
  (`S1:running-stalled` on the same task id on every rank); the same source
  at `--num-tokens 16` runs. The loop's `--prefill-tokens 16` compiles the
  prefill at that extent — the prompt (including BOS) must then fit in 16
  tokens. Keep the extent small until the stall is root-caused (EP2 at 128
  and EP8 at 6/16 both run).

The [daily model workflow](../../.github/workflows/daily_ci.yml) runs this
EP8 loop nightly on the A5 runner (job `e2e-flash-a5`: real
DeepSeek-V4-Flash weights, `--prefill-tokens 16 --prefill-no-retire`,
32 greedy decode steps from the prompt "The capital of France is") and
publishes the prompt and the generated text in the run summary under
"Daily CI Model Test Results", so a reviewer can read the continuation
every day instead of a pass/fail tick. The runner finds the checkpoint
through `PYPTO_DSV4_FLASH_CKPT_DIR` in its `.env` (falling back to the A5
host's `/home/pyptouser/models/DeepSeek-V4-Flash-0731`). The
`weights_flash.py` cache (ep8/tp2) is resolved in this order:
`PYPTO_DSV4_FLASH_WEIGHTS_DIR` from the runner's `.env` if set, else the
shared cache next to the checkpoint (`pypto-weights-cache/flash_ep8_tp2`),
else the runner's own `CI_CACHE_ROOT/dsv4-flash-weights/flash_ep8_tp2`,
which the job builds from the checkpoint once (~25 min) when it is missing.

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
| Real-weight loading | [weights_flash.py](../../models/deepseek_v4_pro/utils/weights_flash.py) |
| Token loop | [synthetic_token_loop.py](../../models/deepseek_v4_pro/synthetic_token_loop.py) |

`config.py`, `decode_metadata.py`, and `rope_tables.py` have no `__main__`
block and are imported rather than run. Which entry points CI schedules is
defined by the [daily model workflow](../../.github/workflows/daily_ci.yml).
