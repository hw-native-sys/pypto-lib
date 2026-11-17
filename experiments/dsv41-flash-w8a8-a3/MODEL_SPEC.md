# Model specification

- Revision: draft 2 (three user decisions recorded 2026-11-02; remaining
  decisions open).
- Model/checkpoint identity and revision: DeepSeek-V4.1-Flash, 552B backbone
  (multimodal MoE, per the vllm-ascend tutorial), local read-only checkpoint
  `DeepSeek-V4.1-Flash-w8a8` (early-preview release, not the official
  ModelSlim branch quantization, per the checkpoint README). Aurora export
  manifest `aurora_export_manifest_v1` (completed 2026-09-10) with per-file
  sha256s; export scope states "Export integrity only; numerical accuracy and
  serving require separate validation".
- Config, tokenizer, chat template, and original forward sources:
  `MODEL_ROOT/config.json` (transformers 5.6.0,
  `DeepseekV41ForCausalLM`), `tokenizer.json` + `tokenizer_config.json`
  (no `chat_template` field), `quant_model_description.json`,
  `Aurora_best_practice.yaml`, `encoding/` (shipped encoding implementation);
  original forward: HF `DeepseekV41ForCausalLM` (transformers 5.6.0).
  Spec basis (user-agreed): vllm-ascend tutorial "DeepSeek-V4.1-Flash",
  vLLM Ascend v0.30.0rc2 documentation baseline, section 5.1 (A3 single-node
  colocated form).
- Scope (text/multimodal, supported modes, exclusions): the A3 form serves
  text and image requests (tutorial sections 6.1/6.2). Proposed first goal:
  text-only; image (32-layer vision tower + aligner, both BF16 in this
  checkpoint) as an agreed follow-up phase — open decision. DSpark
  speculation and MTP layers are excluded from the first goal (the A3 5.1
  colocated form runs without DSpark; only the PD decode form uses it) —
  open confirmation. Multi-request batching and streaming prefill are out of
  scope.
- Target platform, device count, parallel degrees, and rank mapping: `a2a3`
  (Ascend910B-class), 16 logical devices x 65536 MB HBM on this host
  (container view matches the A3 form's 16 logical devices). A3 form (5.1):
  DP4/TP4 + `--enable-expert-parallel` on all 16 logical devices, async
  scheduling, Engram host offload. Exact rank/group table to freeze at
  agreement; note the in-repo scaffold is documented TP4/DP2/EP8 —
  reconciliation is an open decision.
- Quantization policy and checkpoint conversion: as-shipped `ascend`
  W8A8_DYNAMIC (per-channel int8 weights with `weight_scale` and
  `weight_offset` — asymmetric — plus dynamic per-token int8 activations);
  142,495 tensors W8A8_DYNAMIC, 1,002 tensors FLOAT (embed, head, aligner,
  ViT, image tokens, norms, sinks); backbone norm-folding + QuaRot applied
  at export (global rotation map shipped as `optional/quarot.safetensors`);
  Engram tables stored per `embedding_storage` spec (int8_sym, group 32,
  axis 1, fp32 scales, bf16 output) — but the checkpoint README describes
  them as decoded-to-BF16 without scales; contradiction open
  (WEIGHT_MANIFEST). FP4 main KV cache (CSA2) per the tutorial.
- Weight, activation, accumulation, communication, and state dtypes: as
  above; global KV cache 890 bytes/token and persistent KV about 1/8 of
  DeepSeek-V4-Flash per the tutorial; SWA window 128; exact accumulation and
  communication dtypes TBD at agreement.
- Supported batch, context, prefill, and decode shapes: from the A3 form:
  `--max-model-len 150000`, `--max-num-batched-tokens 8192`,
  `--max-num-seqs 32`, `--block-size 128` (serving-side limits; supported
  model-entry boundaries to be frozen in the same units); multi-token
  prefill (token tiles > 1 within layers) is a required default per the
  harness contract.
- Memory estimate (weights, state, scratch): MEASURED from shard headers
  (2026-11-02): dense 11.70 GiB total -> 2.92 GiB/rank at TP4 (replicated
  per DP group); routed experts 507.36 GiB -> 31.71 GiB/rank at EP16;
  per-rank weights ~34.63 GiB on 65536 MB devices (57.6 GiB usable at the
  A3 form's 0.9 utilization) -> ~23 GiB headroom for FP4 KV (890 B/token
  global), activations, and scratch. Engram 206.59 GiB host-side per the
  A3 form (`cpu_offload` + `dp_shared_memory`; the tutorial notes the
  model's 24 Engram heads cannot split evenly across 16 logical devices).
  MTP/DSpark 16.10 GiB and vision/aligner 0.90 GiB are excluded from this
  phase. Feasibility gate: PASSES with reserve.
- Goal owner thread and agreed implementation objective: draft in `goal.md`;
  three launch decisions already agreed by the user (checkpoint, environment,
  A3-form spec basis); the host goal facility has not been activated (awaits
  remaining decisions).
- Prefill/decode source paths, callable signatures, and compilation
  boundary: prior art is `models/deepseek_v4_1_flash/prefill_fwd.py` and
  `decode_fwd.py` (plus per-mode attention variants c1a/c2a/swa, indexer,
  moe, engram, hc, dspark files); compilation must cover embedding, all 40
  layers, final norm, and head. The A3 form captures decode only
  (`FULL_DECODE_ONLY` ACL graph) with prefill eager — for the harness
  gates, both complete-model entries must be compiled model entries
  regardless of serving-side graph capture policy.
- Prefill/decode execution contract: multi-token prefill tiles within layers
  (projections and MoE/expert work keep token tiles > 1); decode processes
  current active tokens; exact tile sizes, MLA/indexer/compressor/expert
  operand shapes, and dispatch granularity frozen at agreement. Attention
  modes per config: `compress_ratios` = [0,0, 2x18, 1x20, 0,0,0] (C2A layers
  2–19, C1A layers 20–39, uncompressed 0–1 and 38–39... exact layer-mode map
  from `compress_ratios`), `kv_source_layer_ids` [2,8,14,20],
  `index_source_layer_ids` [2,8,14,20,24,28,32,36], candidate source layer
  20 with top-2048 blocks of size 8 (hierarchical indexer).
- Structural verification command, generated loop/shape evidence, and
  tokenwise full-stack negative control: TBD; reuse the golden harness plus
  model-specific shape/nesting checks; the tokenwise complete-stack
  negative control is a required gate.
- Generation entry path and how it invokes compiled model forwards: TBD;
  host-side tokenization/sampling allowed; each forward must invoke the
  compiled complete-model entry; deterministic greedy decoding for
  acceptance (the tutorial's functional checks use temperature 0).
- Required deliverables and acceptance rows (including documentation): see
  ACCEPTANCE.md (draft).
- Original generation preflight, fixture-selection policy and frozen case
  IDs: not started; original model = HF reference with this exact checkpoint;
  the tutorial's text probe is "Who are you?" (temperature 0,
  256 completion tokens) and an image describe probe — candidates must be
  calibrated on the original before freezing.
- Physical world/rank table; attention, expert, tensor and pipeline group
  membership: TBD at topology freeze (A3 form: world 16 = DP4 x TP4, EP on).
- Per-rank request count; dense/shared replication and routed-expert
  ownership: TBD (A3 form: 32 seqs per DP engine, 4 DP engines; shared expert
  in DP per `enable_shared_expert_dp`).
- Dispatch/combine protocol, maximum receives, EOS/idle-rank participation:
  TBD (EP in scope; worst-case receive capacity from dispatched token
  tiles).
- Observed usable HBM per rank and peak memory breakdown with reserve:
  65536 MB per logical device, ~2.9–3.1 GB baseline use; exact breakdown
  TBD at topology freeze.
- Checkpoint download/cache location, pinned revision, bytes and local
  availability: `/mnt/data/models/DeepSeek-V4.1-Flash-w8a8` (read-only),
  742.7 GiB on disk, verified complete for all 143,496 index-referenced
  tensors (see WEIGHT_MANIFEST); Aurora manifest records 272 shards and the
  relocation of the last two shards' content into 4 standalone Engram files.

## Architecture facts

| Fact | Value | Evidence |
| --- | --- | --- |
| Layer types and order | 40-layer CED backbone: 20 causal encoder + 20 decoder layers (tutorial); attention compression per layer from `compress_ratios` [0,0,(2)x18,(1)x20,0,0,0]; MoE FFN all layers | config.json; tutorial section 1 |
| Hidden and projection dimensions | hidden 5120; MLA: 64 heads x head_dim 512, `qk_rope_head_dim` 64, `q_lora_rank` 1280, `o_lora_rank` 1024, `o_groups` 8, `num_key_value_heads` 1; MoE intermediate 2304 | config.json |
| Attention, positions, and mask | compressed sparse attention 2 (CSA2): per-layer compressor (C2A/C1A modes), DSA indexer (32 heads x 128 dim, top-k 512), hierarchical candidates (source layer 20, top-2048 blocks of 8); sliding window 128; YaRN 16x (65536 -> 1,048,576), `compress_rope_theta` 160000 | config.json; tutorial |
| Norm and feed-forward computation | RMSNorm eps 1e-20; SwiGLU limit 10.0; MoE: 384 routed + 1 shared experts, top-6, `sqrtsoftplus`, `noaux_tc`, `norm_topk_prob`, routed scaling 1.5; single-pass mHC (`hc_mult` 4, sinkhorn 20, eps 1e-6) | config.json; tutorial |
| Persistent state | FP4 main KV (CSA2), global KV 890 bytes/token, persistent KV about 1/8 of V4-Flash; SWA bounded replay semantics (tutorial section 12); compressed KV state per layer with source layers [2,8,14,20] | tutorial; config.json |
| Embedding, vocabulary, and LM head | vocab 129280; untied embed/head (both FLOAT, unquantized in this checkpoint) | config.json; quant_model_description.json |
| Conditional memory (Engram) | layers [1,14]; ~384.0M embeddings each; ngram size 4; engram vocab 16M, compressed vocab 99092; 8 heads x 256 dim; int8_sym group-32 + fp32 scales per files (README says BF16 — contradiction open); QuaRot global rotation | config.json; `optional` entry; file headers; README |
| Speculative paths (excluded from first goal) | MTP: 3 nextn predict layers; DSpark: block 5, target layers [37,38,39], Markov rank 256, own MoE (128 experts, top-3), noise token 128799 | config.json |
| Vision (follow-up phase) | 32-layer tower, hidden 1024, 16 heads, intermediate 2816, patch 14, downsample 3, max 1024 image tokens, `image_token_id` 129264; aligner + image tokens BF16 | config.json; quant_model_description.json |

## Decisions for user agreement

Agreed by the user (2026-11-02; spec revision 1):

1. Checkpoint: local read-only `DeepSeek-V4.1-Flash-w8a8`.
2. Environment: `.venv-local` via `activate-pypto.sh`.
3. Spec basis: the vllm-ascend A3-supported form (tutorial rev vLLM Ascend
   v0.30.0rc2, section 5.1 single-node colocated).
4. Goal framing: fresh implementation for A3 with this checkpoint's W8A8
   policy; the existing `models/deepseek_v4_1_flash` (A5 platform,
   different quantization) is a semantics reference only — not reusable
   implementation.
5. Topology: A3 form DP4/TP4 + expert parallel on 16 logical devices;
   pypto platform `-p a2a3`.
6. Modality: text-only first phase; vision tower and aligner excluded.
7. DSpark and MTP excluded from the first goal; Engram included
   (host-offloaded, int8_sym group-32 + fp32 scales; file headers are
   ground truth — the checkpoint README's decoded-BF16 prose is stale).

Remaining minor items, resolved at fixture calibration (not
implementation-blocking): chat-template proposal for generation fixtures
(plain completion + deepseek_v41 format), and numerical thresholds on the
original HF reference (transformers 5.6.0).

Delegation (user, 2026-11-02): implementation is executed by a launched
model Main Agent; the outer session acts as launcher/observer and collects
time statistics in `TIME_LOG.md`.

Multi-token prefill must retain prompt-token tiles within layers. Necessary
recurrent scans and token tiling are allowed; an outer token loop through the
entire decoder stack is diagnostic-only by default. Performance tuning being
excluded does not waive this requirement. Record any reduced tokenwise-only
scope as an explicit user decision, never as an agent-authorized implementation
choice. Distinguish within-prompt token processing from optional multi-request
batching and external streaming prefill.

## Scope revision 2 (user decision 9, 2026-11-03): execution contract change

The execution contract changes from compile-time constexpr envelopes plus flat
row buffers to **dynamic axes (`_DYN`) and runtime scalars inside one compiled
artifact**, with real dynamic batch (multi-request) and dynamic S (runtime token
counts and context lengths), and the serving-adapted paged KV cache format:

- paged raw window cache (`[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM]`-style), paged
  compressed cache, index cache, per-request `index_block_table
  [B_DYN, TABLE_DYN]`, `request_ids`, `compressed_lens`/slots,
  `window_slots`/indices;
- interface parameters follow the A5 implementation
  (`models/deepseek_v4_1_flash/`) as closely as possible: fully type-annotated
  port-level signatures with `_DYN` axes, `num_tokens` as a runtime scalar, HC
  inputs in the chain signature, no exposed chain-internal scratch buffers.

Permitted divergences: A3 quantization (per-channel int8 + weight scale/offset +
dynamic per-token int8) and A3 parallelism (DP4/TP4 + EP16 exchange windows +
tp_reduce) only; other exceptions require individual justification in MODULES.
Staged validation hooks are kept at layer level (attn_input-style Out hooks)
where the A5 port signatures conflict with chain-internal scratch exposure.

Sequencing: complete-entry semantics unit (in flight) -> interface migration ->
per-mode/layer/structural device revalidation -> complete-entry device
validation + tokenizer/greedy generation + peak HBM once on the final interface
-> EP16 device evidence -> closeout. Prior interface-shaped evidence is
semantics-valid but interface-stale after the migration.
