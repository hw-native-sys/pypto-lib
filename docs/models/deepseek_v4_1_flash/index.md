# DeepSeek V4.1 Flash

This directory documents the Ascend A5 implementation in
[`models/deepseek_v4_1_flash/`](../../../models/deepseek_v4_1_flash/config.py).
It includes the text backbone, prefill and decode forwards, DSpark speculative
decoding, and a separately callable vision encoder. Operator entries provide
Torch references and standalone validation; forward entries compose those same
operators.

## Model configuration

[`config.py`](../../../models/deepseek_v4_1_flash/config.py) owns the model
preset, layer schedule, and parallel configuration. The tensor specs in each
operator define its executable storage and layout contracts.

| Property | Configuration |
| --- | --- |
| Text backbone | 40 layers, hidden width 5,120, vocabulary 129,280 |
| Attention | 64 heads, head width 512, rotary width 64, sliding window 128 |
| Projections | Query LoRA rank 1,280; output LoRA rank 1,024 in 8 groups |
| MoE | 384 routed experts, top-6 routing, one shared expert; intermediate width 2,304 |
| Hyper-connections | 4 streams; persistent hidden state is FP32 |
| Context ceiling | 1,048,576 positions, with 128-token cache pages |
| Engram | Layers 1 and 14; forward consumes host-gathered embedding rows |
| DSpark | 5 speculative tokens plus one verification row per request; default decode batch 12, or 72 rows |

Quantized projection paths use MXFP8 weights and activations; routed expert
weights use packed MXFP4. Scale tensors use the owning operator's MX layout.
Norms, residual mixing, cache payloads, and unquantized weights retain their
specified BF16 or FP32 types. Checkpoint metadata alone does not describe every
runtime tensor's dtype.

## Attention schedule and cache ownership

Layer numbers below are zero-based. All attention stages include the sliding
window; C1A and C2A also attend to selected compressed-cache entries.

| Layers | Family | Cache and selection behavior |
| --- | --- | --- |
| 0–1 | SWA | Sliding-window attention |
| 2–19 | C2A, ratio 2 | Full paths at 2, 8, and 14; intervening layers reuse their source's cache and selection |
| 20–39 | C1A, ratio 1 | Full path at 20; reindex at 24, 28, 32, and 36; other layers reuse |

A **full** path projects and publishes KV, updates compression and index state,
and selects entries for attention. **Reindex** reuses the source KV cache and
computes a new selection. **Reuse** consumes existing KV and selection state.
The source-layer tables in `config.py` determine ownership; these are different
roles in one forward, rather than interchangeable execution options. C2A reuse
uses the shared C1A reuse kernels with ratio-2 metadata.

## Execution structure

The text forward composes embedding and metadata preparation, the configured
attention/MoE layer schedule, and final HC head normalization and LM-head output.
Engram contributes at its configured layers. A layer follows this structure:

```text
HC state → HC pre-mix / attention norm → Q/KV projection and RoPE
         → cache publication / compression / selection as required
         → sparse attention → output projection + HC post
         → HC pre-mix / router → shared and routed experts → combine / HC post
```

The output projection uses the fused O-B + HC-post implementation in
[`o_proj.py`](../../../models/deepseek_v4_1_flash/o_proj.py).
Attention entries own chunk scheduling and dependencies around that shared
implementation. MoE dispatches local token rows to EP expert owners and combines
the results back into the local HC state.

## Parallel configuration

| Component | Contract |
| --- | --- |
| Attention weights | Full weights on every rank; no attention TP |
| Prefill CP | `--cp 1/2/4/8`; CP1 uses local attention, CP>1 uses head-tail query placement and KV publication across peers |
| MoE EP | `--ep 2/4/8`; EP1 is unsupported |
| Prefill DP | `DP_SIZE = EP_SIZE // CP_SIZE`; CP must divide EP |
| Decode attention | Independent requests and local KV/cache state per rank; no decode CP partitioning |
| Decode MoE | EP dispatch/combine across ranks, while attention retains local request ownership |
| LM-head TP | Independent `--lm-head-tp 1/2/4/8`, subject to the rank-group constraints checked by the entry |

For prefill, at CP1, each DP group may pack multiple independent requests. At CP>1, each DP
group supports one request. Different DP groups own independent requests and
cache state. The standalone attention entries can run CP1 on a single device;
layer and forward entries also compose EP MoE and require multiple devices.

For a sequence divided into `2 * CP` contiguous segments, CP rank `r` owns
segments `r` and `2 * CP - 1 - r`. Thus CP2 assigns segments `(0, 3)` to rank 0
and `(1, 2)` to rank 1. Metadata restores chronological KV order and the caller's
row order. CP1 avoids the CP communication path.

Prefill supports a nonzero preceding sequence length for chunked continuation.
This requires the caller to preserve matching cache and compressor state;
a prefix length does not populate that state or provide a prefix-cache manager.
The checkpoint `generate` entry currently requires CP1.

Decode runs attention independently on each rank, with full attention weights
and rank-local request metadata, KV caches, and compressor state. The prefill
formula `DP_SIZE = EP_SIZE // CP_SIZE` describes prefill grouping; decode does
not use its head-tail CP layout. Use CP1 for the current decode deployment.
The default per-rank batch is `DECODE_BATCH = 12`; DSpark verifies
`DECODE_SEQ = 6` rows per request, giving up to 72 local rows per step.
MoE routes these local rows across the EP group and returns results to their
originating rank.

LM-head TP partitions the vocabulary projection independently of attention.
The forward requires `EP_SIZE % LM_HEAD_TP_SIZE == 0`. Contiguous groups of
`LM_HEAD_TP_SIZE` ranks exchange selected hidden rows, compute vocabulary
shards, and return output to the row owners; requests remain independently
owned by their original ranks. For example, EP8 with LM-head TP2 has eight
local attention batches and four two-rank output-projection groups. With
LM-head TP1, each rank computes its own complete vocabulary projection.

## Entry points and validation

| Purpose | Entry files |
| --- | --- |
| Prefill attention | `prefill_swa.py`, `prefill_c1a_full.py`, `prefill_c1a_reindex.py`, `prefill_c1a_reuse.py`, `prefill_c2a_full.py`, `prefill_c2a_reuse.py` |
| Decode attention | Corresponding `decode_swa.py`, `decode_c1a_*` and `decode_c2a_*` files |
| Layer composition | [`prefill_layer.py`](../../../models/deepseek_v4_1_flash/prefill_layer.py), [`decode_layer.py`](../../../models/deepseek_v4_1_flash/decode_layer.py) |
| Text forward | [`prefill_fwd.py`](../../../models/deepseek_v4_1_flash/prefill_fwd.py), [`decode_fwd.py`](../../../models/deepseek_v4_1_flash/decode_fwd.py) |
| Speculative decoding | [`decode_fwd_dspark.py`](../../../models/deepseek_v4_1_flash/decode_fwd_dspark.py), `dspark_drafter.py`, `dspark_state.py`, and sampling helpers |
| Shared operators | `hc.py`, `qkv_proj_rope.py`, `compressor.py`, `sparse_attention.py`, `o_proj.py`, `engram.py`, and `lm_head.py` |

For prefill attention, `--tokens` is capacity per DP group, before CP partitioning.
`--query-lens '65,31;96'` describes two requests in DP0 and one in DP1;
`--start-pos '127,0;256'` supplies their respective preceding lengths.
Commas separate requests and semicolons separate DP groups. `--requests` provides
a balanced split when explicit query lengths are omitted. The forward entry
uses `--capacity` for its DP capacity.

For example, these commands generate SWA artifacts without executing devices:

```bash
python models/deepseek_v4_1_flash/prefill_swa.py -p a5 --cp 1 --tokens 512 --compile-only
python models/deepseek_v4_1_flash/prefill_swa.py -p a5 --cp 2 --tokens 1024 --compile-only
```

Use allocated devices for execution and consult each entry's `--help` for its
options. See the [compile/runtime workflow](../../run-and-validate/compile-runtime-workflow.md)
and [golden harness](../../run-and-validate/golden-harness.md) for artifact and
comparison behavior. Standalone operators compare against Torch references.
Forward smoke fixtures exercise composition and execution; they do not establish
checkpoint-level generation accuracy or a complete numerical forward golden.
The configured context ceiling and supported parallel-size choices are not a
claim that every size/sequence-length combination has been device validated.

## Vision

[`vision.py`](../../../models/deepseek_v4_1_flash/vision.py) provides a BF16 ViT
and visual aligner on A5. The released profile has 32 blocks, width 1,024,
16 attention heads, and 14×14 RGB patches. The aligner packs 3×3 neighborhoods
and projects them to 5,120-channel language embeddings.

`VisionEncoder` accepts preprocessed CPU BF16 patches and a patch-grid shape;
`load_vision_weights` loads the vision and aligner checkpoint tensors. One call
processes one image. The caller owns resizing, normalization, image delimiters,
and insertion of aligned rows into the language input. Vision has separate 2D
RoPE tables. This callable component does not itself provide an end-to-end
multimodal serving pipeline.

The standalone entry defaults to a reduced two-layer Torch comparison;
`--checkpoint` selects the released profile and weights, while `--patches` and
`--grid` supply preprocessed input.

## Development priorities

| Field | Current priority |
| --- | --- |
| Performance target | 72-token decode and 1,024-token prefill with DP1/CP1, using full attention weights. Apply optimizations to the shared implementation across all attention layers and adapt all affected callers. |
| Deferred performance work | Performance tuning for other token counts and prefill parallel configurations remains secondary to these targets; correctness and the shared fused output projection / HC-post implementation remain required. |
| Compatibility requirements | Preserve correctness and interface contracts for already supported paths, including other token counts and DP/CP configurations. Update affected callers to use revised shared implementations. The optimized C++ routed FFN requires A5 device execution; simulation requests fail explicitly rather than selecting a different numerical implementation. |
