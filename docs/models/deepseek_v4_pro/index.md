# DeepSeek V4-Pro

`models/deepseek_v4_pro/` is the Ascend 950 (A5) implementation for DeepSeek-V4
Pro, with an optional Flash architecture preset. Select the preset at import
and compile time with `DEEPSEEK_V4_VARIANT=pro|flash` or `--variant pro|flash`.

## Deployment configuration

The `PRO` and `FLASH` presets in [config.py](../../../models/deepseek_v4_pro/config.py)
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

Native MXFP8-MXFP4 MoE migration (staged):

- **Expert W1/W3 (landed)**: [expert_shared.py](../../../models/deepseek_v4_pro/expert_shared.py) and [expert_routed.py](../../../models/deepseek_v4_pro/expert_routed.py) use ``pl.matmul_mx`` for the up-projection weights; W2 remains INT8.
- **Host helpers**: [mx_utils.py](../../../models/deepseek_v4_pro/mx_utils.py) — MXFP grid simulation, E8M0 pack/unpack, FP4→FP8 LUT tables, and ``matmul_mx`` goldens.
- **Gate / MoE dispatch (deferred)**: [gate.py](../../../models/deepseek_v4_pro/gate.py) and [moe.py](../../../models/deepseek_v4_pro/moe.py) still use the **INT8 stand-in** ABI until MX activation quant and dispatch wiring land. **Do not use ``pl.quant_mx``.**

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
[lm_head.py](../../../models/deepseek_v4_pro/lm_head.py) entry point validates that
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

`utils.py` converts the released DeepSeek-V4-Flash checkpoint (hybrid
MXFP4 routed experts + block-FP8 attention/shared-expert linears) into the
host-tensor ABI of the two forward drivers: FP4/FP8 tensors are dequantized
and re-quantized to the kernels' INT8 + per-output-channel FP32-scale form,
per-layer tensors are stacked and EP/TP-sharded exactly like the fixture
specs. Convert once offline, then point the drivers at the cache:

```bash
PYTHONPATH=.:models/deepseek_v4_pro python -c 'import utils; utils.main()' \
    --variant flash --ep 8 --tp 2 \
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

[synthetic_token_loop.py](../../../models/deepseek_v4_pro/synthetic_token_loop.py)
drives the full prompt-to-text path on real weights: the prompt is encoded
with the checkpoint's `tokenizer.json` (BOS prepended unless `--no-bos`),
the resident session runs one prefill plus `--decode-steps` greedy decode
steps, every step asserts that all ranks sampled the same token, and the
sampled ids are detokenized at the end. Decoding stops early when
`--eos-id` (default 1) is sampled. Without `--weights` the loop keeps its
synthetic zero-weight control-path behavior.

The EP8 example uses the fixed 128-row prefill capacity; active rows follow the
encoded prompt length:

```bash
python models/deepseek_v4_pro/synthetic_token_loop.py --variant flash \
    --ep 8 --tp 2 -d 0,1,2,3,4,5,6,7 \
    --weights build_output/flash_weights_ep8_tp2 \
    --tokenizer /path/to/DeepSeek-V4-Flash/tokenizer.json \
    --prompt "The capital of France is" --decode-steps 32
```

The full prefill and decode programs carry runtime `num_tokens` and
`moe_epoch_base` scalars in their compiled ABI. Their `ScalarSpec`s use
`compile_runtime=True`, so `run` passes `pl.RUNTIME` during
signature-driven compilation instead of folding the initial values into
generated task arguments. `num_tokens` follows the real prompt/decode row
count, while callers advance the epoch scalar by
`LAST_MOE_EPOCH` for every physical dispatch on a persistent worker.

MoE payload readiness uses one cache-line-padded epoch slot per source and
producer block. Each dispatch or combine block stores its current epoch with
`Set` only after that block's self-draining tensor puts; a separate whole-grid
wait observes every remote slot with `>= epoch` before gather or reduction.
A separate per-rank `consumed` epoch is published after the complete reduction,
and the next MoE invocation waits for every rank to consume the previous epoch
before reusing payload windows. This avoids shared-counter atomic fan-in,
detached notifications, and mixed readiness/lifetime credit arithmetic. The
full forwards, standalone MoE, and decode-layer driver keep these epochs
monotonic for the lifetime of their persistent program. The packed
prefill-layer and fixed-epoch MTP drivers instead quiesce all final consumed
markers and clear only their own inbound slots before a later synchronous
dispatch.

Both full programs must be recompiled when moving from an older artifact. The
token loop rejects artifacts that omit the runtime scalar or whose generated
`host_orch.py` does not forward it through `TaskArgs.add_scalar`. After a
timeout or partial dispatch failure, callers must discard the worker and its
persistent windows rather than retrying with a guessed epoch.

The prefill and decode RoPE paths use fixed even/odd lane gather and scatter
operations for adjacent-lane permutations instead of synthesizing tile-local
index tensors.

The [daily model workflow](../../../.github/workflows/daily_ci.yml) runs this
EP8 loop nightly on the A5 runner (job `e2e-flash-a5`: real
DeepSeek-V4-Flash weights, fixed 128-row prefill capacity with active rows set
from the prompt, and 32 greedy decode steps from "The capital of France is") and
publishes the prompt and the generated text in the run summary under
"Daily CI Model Test Results", so a reviewer can read the continuation
every day instead of a pass/fail tick. The runner finds the checkpoint
through `PYPTO_DSV4_FLASH_CKPT_DIR` in its `.env` (falling back to the A5
host's `/home/pyptouser/models/DeepSeek-V4-Flash-0731`). The
`utils.py` cache (ep8/tp2) is resolved in this order:
`PYPTO_DSV4_FLASH_WEIGHTS_DIR` from the runner's `.env` if set, else the
shared cache next to the checkpoint (`pypto-weights-cache/flash_ep8_tp2`),
else the runner's own `CI_CACHE_ROOT/dsv4-flash-weights/flash_ep8_tp2`,
which the job builds from the checkpoint once (~25 min) when it is missing.
Unlike the rest of the nightly, this job currently builds a pinned pypto
(`pypto-ref` in the workflow, with the full story in its comment): the
toolchain that pypto HEAD pins carries a pto-isa A5 dispatch regression
that stalls every EP8 prefill before the first token. The pin comes off
once the upstream fix reaches pto-isa's mirror and the simpler/pypto pins
move past it. While the pinned toolchain's probabilistic cross-rank
divergence ([#1043](https://github.com/hw-native-sys/pypto-lib/issues/1043))
stays open, the job also retries the loop once; `e2e.json` carries the
attempt count and the first attempt's error, so a flaky night still reads
as exactly what it was.

## Files

| Group | Files |
| --- | --- |
| Full forward | [decode_fwd.py](../../../models/deepseek_v4_pro/decode_fwd.py), [prefill_fwd.py](../../../models/deepseek_v4_pro/prefill_fwd.py) |
| Layer composition | [decode_layer.py](../../../models/deepseek_v4_pro/decode_layer.py), [prefill_layer.py](../../../models/deepseek_v4_pro/prefill_layer.py) |
| MTP | [decode_mtp.py](../../../models/deepseek_v4_pro/decode_mtp.py), [prefill_mtp.py](../../../models/deepseek_v4_pro/prefill_mtp.py), [mtp_projection.py](../../../models/deepseek_v4_pro/mtp_projection.py) |
| Decode attention orchestration | [decode_attention_swa.py](../../../models/deepseek_v4_pro/decode_attention_swa.py), [decode_attention_csa.py](../../../models/deepseek_v4_pro/decode_attention_csa.py), [decode_attention_hca.py](../../../models/deepseek_v4_pro/decode_attention_hca.py) |
| Decode sparse attention (fused o-proj) | [decode_sparse_attn.py](../../../models/deepseek_v4_pro/decode_sparse_attn.py), [decode_sparse_attn_swa.py](../../../models/deepseek_v4_pro/decode_sparse_attn_swa.py), [decode_sparse_attn_hca.py](../../../models/deepseek_v4_pro/decode_sparse_attn_hca.py) |
| Decode compressors and indexer | [decode_compressor_ratio4.py](../../../models/deepseek_v4_pro/decode_compressor_ratio4.py), [decode_compressor_ratio128.py](../../../models/deepseek_v4_pro/decode_compressor_ratio128.py), [decode_indexer.py](../../../models/deepseek_v4_pro/decode_indexer.py), [decode_indexer_compressor.py](../../../models/deepseek_v4_pro/decode_indexer_compressor.py) |
| Prefill attention and cache | [prefill_attention_swa.py](../../../models/deepseek_v4_pro/prefill_attention_swa.py), [prefill_attention_csa.py](../../../models/deepseek_v4_pro/prefill_attention_csa.py), [prefill_attention_hca.py](../../../models/deepseek_v4_pro/prefill_attention_hca.py), [prefill_sparse_attn.py](../../../models/deepseek_v4_pro/prefill_sparse_attn.py), [prefill_compressor_ratio4.py](../../../models/deepseek_v4_pro/prefill_compressor_ratio4.py), [prefill_compressor_ratio128.py](../../../models/deepseek_v4_pro/prefill_compressor_ratio128.py), [prefill_indexer.py](../../../models/deepseek_v4_pro/prefill_indexer.py), [prefill_indexer_compressor.py](../../../models/deepseek_v4_pro/prefill_indexer_compressor.py) |
| Shared transforms | [rmsnorm.py](../../../models/deepseek_v4_pro/rmsnorm.py), [qkv_proj_rope.py](../../../models/deepseek_v4_pro/qkv_proj_rope.py), [hc_pre.py](../../../models/deepseek_v4_pro/hc_pre.py), [hc_post.py](../../../models/deepseek_v4_pro/hc_post.py), [hc_head.py](../../../models/deepseek_v4_pro/hc_head.py) |
| MoE and output | [moe.py](../../../models/deepseek_v4_pro/moe.py), [gate.py](../../../models/deepseek_v4_pro/gate.py), [expert_shared.py](../../../models/deepseek_v4_pro/expert_shared.py), [expert_routed.py](../../../models/deepseek_v4_pro/expert_routed.py), [lm_head.py](../../../models/deepseek_v4_pro/lm_head.py) |
| Metadata and host helpers | [config.py](../../../models/deepseek_v4_pro/config.py), [decode_metadata.py](../../../models/deepseek_v4_pro/decode_metadata.py), [rope_tables.py](../../../models/deepseek_v4_pro/rope_tables.py) |
| Real-weight loading | [utils.py](../../../models/deepseek_v4_pro/utils.py) |
| Token loop | [synthetic_token_loop.py](../../../models/deepseek_v4_pro/synthetic_token_loop.py) |

`config.py`, `decode_metadata.py`, and `rope_tables.py` have no `__main__`
block and are imported rather than run. Which entry points CI schedules is
defined by the [daily model workflow](../../../.github/workflows/daily_ci.yml).
