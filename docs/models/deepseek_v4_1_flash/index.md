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
| Engram | Layers 1 and 14; forward consumes BF16 lookup rows produced on Device |
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

Prefill composes embedding and metadata preparation, the configured attention/MoE
layer schedule, HC collapse, RMSNorm, vocabulary projection, and greedy sampling
for the last query row per request. The reusable `decode_fwd` and its L3 entry
run only the backbone, retaining final HC state and DSpark target taps. DSpark
composes the output head and sampling after that backbone. `lm_head` writes
FP32 logits in place; `greedy_sample` and `markov_sample` are separate operations.
Engram contributes at its configured layers. A layer follows this structure:

```text
HC state → HC pre-mix / attention norm → Q/KV projection and RoPE
         → cache publication / compression / selection as required
         → sparse attention → output projection + HC post
         → HC pre-mix / router → shared and routed experts → combine / HC post
```

HC-post retains its completed residual state in FP32 for the next HC-pre.
HC collapse rounds its FP32 sum to BF16 before RMSNorm and quantized
projections. HC coefficients and residual mixing remain FP32.

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
`DECODE_SEQ = 6` rows per request, giving 72 local rows in the default workload.
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
| Speculative decoding | [`decode_fwd_dspark.py`](../../../models/deepseek_v4_1_flash/decode_fwd_dspark.py), `dspark_drafter.py`, `dspark_init.py`, `decode_metadata.py`, `dspark_attn.py`, and `markov.py` |
| Shared operators | `hc.py`, `qkv_proj_rope.py`, `compressor.py`, `sparse_attention.py`, `o_proj.py`, `engram.py`, and `lm_head.py` |

The six standalone decode attention entries accept explicit request starts and
query lengths through `--start-pos` and `--query-lens`. Both lists have one
comma-separated value per request. Their lengths and sum determine the request
and token counts; the default batch of 12 and 72-token workload are not limits
for these standalone entries. For example, batch 64 with four query tokens per
request uses 64 starting positions and 64 query lengths of `4` (256 tokens).
Query blocks must remain within the model position range. This does not change
the forward or DSpark serving capacities.

Compressed-attention decode fixtures retain only the sliding-window pages visible
to the current query block, plus an unmapped control page. Compressed and index caches retain
their history. Random cache generation uses bounded page batches; index goldens
batch tokens sharing one request while retaining sequential BF16 head rounding.
C2A reuse fixture preparation computes its source compressed cache and selected
indices without computing the source attention output.

C2A reuse validates its composed HC output with `atol=0.01`, `rtol=1/128`,
and a maximum outlier ratio of 1%. Its cache and HC-mix checks retain their
separate tolerances.

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
The prefill, decode, DSpark decode, and drafter smoke entries use `golden.run`
without a numerical golden. Multi-stage initialization is owned by `dspark_init.py`.
Forward smoke fixtures exercise composition and execution; they do not establish
checkpoint-level generation accuracy or a complete numerical forward golden.
The configured context ceiling and supported parallel-size choices are not a
claim that every size/sequence-length combination has been device validated.

## Engram Host offload development

This experimental integration targets PyPTO `integration/urma-ccu-test` at
`864c8b86b2bae5c53b951d880b6794c5887cb0d2` and its pinned runtime and toolchain.
Host URMA window support in other PyPTO versions has not been verified here.

The shared lookup in `engram.py` implements sliced ENP and Full-Vocab local
fetch with FP8 E4M3 payloads transported as UINT8 bytes. Eight E8M0 group-32
scales per row remain on the Device; the lookup dequantizes into the BF16
`engram_lookup` contract consumed by both forwards. `EngramHistory` owns the
common compressed-token hash and request-history calculation.

[`engram_host_offload.py`](../../../models/deepseek_v4_1_flash/engram_host_offload.py)
provides a standalone validation entry. Prepare frozen inputs, Torch reference
outputs and A5 artifacts outside the device allocation, then execute with the
queue-assigned devices:

```bash
python models/deepseek_v4_1_flash/engram_host_offload.py --compile-only --ranks 4 --tokens 3 --artifact build_output/engram-enp4
python models/deepseek_v4_1_flash/engram_host_offload.py --artifact build_output/engram-enp4 --devices "$TASK_DEVICE"
```

Preparation accepts `--strategy sliced_enp` or `--strategy full_vocab_local`.
The default bounded cases cover independent request histories, continuation,
rollback, inactive rows, duplicate lookups, shard boundaries, and staging reuse.
Use `--tokens 72` or `--tokens 8192` for the requested Decode/Prefill lookup
volumes; these still use synthetic tables and do not establish model accuracy
or representative bandwidth. Four- and eight-rank artifact preparation is
supported; eight-device execution is deferred.

The remote path GETs directly into rank-local Device staging and consumes
those FP8 bytes on the Device. Two independent 32-row slots use 16 KiB per rank, and each fetch
task issues at most 32 remote GETs. Each slot's next writer explicitly waits
for its previous consumer. Separate slots permit fetch/consume overlap across
batches; a speedup requires device measurements. A local row bypasses remote
staging. Negative IDs are explicit zero padding; nonnegative IDs must be within
the prepared table's padded row extent.

The selected PyPTO Host-window API initializes separate physical windows per
rank. Full-Vocab local therefore currently replicates its Host payload storage;
passing the same source tensor to every rank does not share the destination.
A separate native-mapper capability experiment validated one shared backing up
to 1 GiB on four devices. Binding that storage through the public L3 API and
validating full-table capacity remain required before claiming shared-backing
Full-Vocab integration.

The two released payload tables contain 768,022,850 rows in total: approximately
183.1 GiB of FP8 payload and 5.72 GiB of scales. Each Device keeps the complete
scale data in this implementation. Ignoring alignment and staging, sliced ENP
needs 183.1 GiB of aggregate Host payload storage; duplicated Full-Vocab needs
approximately 732.4 GiB on four ranks or 1,464.9 GiB on eight ranks. A shared
physical Full-Vocab backing would need approximately 183.1 GiB. These are
storage calculations, not verified registration or virtual-address limits.

The checkpoint generation path prepares hash IDs on the CPU and passes them
with Device-resident scales into each forward. The L3 entry initializes one
Host payload shard per rank. Its L2 entry runs the shared lookup inline,
fetches remote FP8 bytes directly into two bounded Device staging buffers,
dequantizes on Device, and feeds the existing forward computation. Prefill and
Decode retain one computation body each; lookup is part of the same L2 graph.
A task consuming the lookup result waits for its asynchronous GET completion.

Compile both forwards with the same `--engram-strategy sliced_enp` and actual
rank count. Checkpoint runs default to the released `--engram-table-rows`
extent; synthetic forward fixtures default to 4096 rows per layer. Set the
released extent explicitly when compiling a checkpoint artifact without loading
the checkpoint. The checkpoint
entry rejects a table extent that does not match its checkpoint. The generation
entry accepts the integrated Prefill artifact via `--runtime-dir` and Decode
artifact via `--decode-runtime-dir`; no separate Engram runtime artifact is
needed. Each forward entry writes `engram_layout.json` beside its compiled
artifact. The controller rejects missing layouts or a mismatch in rank count,
table extent, strategy, or ABI version before creating runtime windows. When
compiling through the Python API, call `write_forward_engram_layout(output_dir)`
in the same configuration after compilation. Old BF16-row forward artifacts
must be recompiled.

Payloads initialize from seekable checkpoint slices with streamed padding;
scales upload once per rank. ENP4 uses four shards and ENP8 eight shards, with
one aggregate table in Host memory per compiled program. PyPTO retains domains
independently for Prefill and Decode: preparing and dispatching both currently
retains two complete sets of shards. There is no public cross-program window
binding or per-program release API. The default single-Prefill-program controller
also handles continuation and retains one set. `full_vocab_local` uses independent
full-table windows per rank per program. The controller budgets the configured
program count, checks available Host memory before initialization, and reports
both per-program and total payload bytes. It does not silently change strategy.

The checkpoint entry supports capacities up to the configured 8192-row bound.
Its report includes metadata/upload, integrated lookup plus forward time, and
output copy-back. Numerical acceptance with real checkpoint weights and large
Host registrations remains pending for the requested large workloads. The unmodified pinned runtime rejects
Host windows at its new UDMA capability precheck. A local opt-in diagnostic
patch permits both strategies to pass four exact synthetic lookup cases at
72 and 8192 tokens/rank on four cards. With the patch, real-checkpoint ENP4
also completes short-prompt generation through all 40 layers, capacity-16
chunked Prefill and one-token continuation using the Prefill program, with the
complete payload registered. That run uses `--ring-heap-mib 1024`; the default
four 4-GiB heaps exhaust EP4 HBM while allocating checkpoint weights and full
Device scales. This is separate from the deferred intermittent scheduler
timeout and does not establish reference-level model accuracy or the requested
large-workload end-to-end acceptance.
A second four-card run switches to independent Decode with shared resident
weights, caches and scales, and matches all four prompts' two generated token
IDs. Host windows remain separate per program; both full payload sets register
and release successfully. Decode has one active token per rank in a 192-row
physical allocation, so this does not validate the requested 72-active-token
model workload.
See the linked interface-limit report below.
Eight-card hardware execution is deferred.
The DSpark target-body consumer still supplies prepared BF16 rows to the shared
decode computation; adapting its speculative hash/history inputs is separate
from the text Prefill/Decode generation entry.

The explicit `--benchmark` entry measures a prepared lookup with retained
windows, Device scales and Device output buffers. It saves raw per-rank samples,
P50/P99, effective useful-byte bandwidth and matched local/remote/mixed cases.
See [Host offload measurements and interface limits](host-offload.md) for
commands, timing boundaries, memory accounting and outstanding dependencies.

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
