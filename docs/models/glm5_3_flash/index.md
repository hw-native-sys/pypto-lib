# GLM-5.3-Flash

`models/glm5_3_flash/` contains the implementation of
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (`glm5_next`,
`Glm5NextForConditionalGeneration`) on **Ascend a2a3**. It includes component
kernels, Torch goldens, W8A8 checkpoint staging, and composed MoE FFN entries.
Complete prefill/decode forward composition is a separate integration step.

## Current precision status: ordinary arithmetic and CPU golden

The default `decode_fwd.py` invocation used by device CI executes only layer 0
for two decode steps, retaining the four-layer fixture and independent CPU
golden. Multi-layer validation is opt-in with `--end-layer 4` and is temporarily
excluded from CI. Use `--start-layer N --end-layer N+1` (with integer values)
to validate a different layer independently. On local a2a3 TP2/EP2, all four
layers pass separately against their CPU goldens. The DSA CPU golden rounds
each block's FP32 exponential weights to BF16 before PV, as the kernel does;
its softmax denominator remains FP32. Before this contract correction, layer 3
failed the MLP output and stream gates. Acceptance thresholds are unchanged.
The explicit four-layer chain still fails the independent CPU golden: step-0
hidden has 23.7198% outliers against the 5% limit.

The results below describe the local decode baseline before merging upstream
`4a7c42d`. That merge adopts upstream BF16-normalized routing and EP notification
updates. The earlier decode reference used FP32 PV operands, while the sparse
attention executable already used BF16 PV operands. The current decode golden
also models that BF16 operand boundary. The explicit four-layer run above uses
the merged revision and this reference.

The current working tree removes arithmetic.py, its seven consumers' software
compensation, and the unused compensated linear helper. CPU golden remains
the acceptance reference; native hybrid comparisons are diagnostic only.

Selected local operator cases and TP2 compositions pass. Both synthetic and
official-weight first-layer/two-step CPU gates pass. One comparator correction
allows two FP32 machine epsilons of relative error for derived RMSNorm scales,
with zero outliers; inactive rows remain exactly zero. Other thresholds and
all production CPU golden formulas are unchanged.

The independent four-layer INT8-Dense gate still fails, with 23.8068% step-0
hidden outliers against a 5% limit and several failing caches. CPU replay on
captured intermediate inputs passes all nine output groups. Quantization
sensitivity is measurable: 49 differing first-Dense input codes grow to 7,953
by the second Dense layer under independent propagation. Local acceptance does
not imply multi-layer acceptance. Earlier sections describing retained
compensation are historical and superseded by this status.

Follow-up tests add deterministic Dense cases, RMSNorm rounding fixtures and
seven captured-input RMSNorm-to-Dense replays; all pass, with exact input INT8
codes in the replays. A fixed four-layer fixture with `--end-layer 1` passes;
`--end-layer 2` fails with 7.7103% hidden outliers. The first two layer traces
match the earlier four-layer run exactly. An independent DSA-plus-MoE layer
also fails intermediate output/stream gates despite passing final hidden.
These composition gaps remain open; no compensation or tolerance change was
introduced in this follow-up.

DSA single-layer tracing reproduces its attention output bitwise from local
probes. Existing BF16 PV rounding accounts for most local attention drift,
which propagates into changed MoE INT8 codes. All sampled attention stages pass
their original local gates. Router standalone output annotations and its
biased-selection comparison were repaired; the default and captured-input
cases now perform actual comparisons and pass. Historical router PASS lines
without output comparisons are not validation evidence. Independent single-
layer acceptance remains unresolved.

Further decomposition shows that mHC adds no extra INT8 code differences on
this fixture: all 833/1,136 crossings are explained by attention drift.
Per-token quantization multipliers are unchanged. Replacing dequant scales or
using a separately evaluated normalized-BF16 quantization contract does not
remove the MoE failure. The current investigation therefore focuses on the
attention numerical path; compute code, CPU goldens and gates are unchanged.

A user-authorized isolated vLLM-Ascend amplitude-scaling trial passes local
attention checks but does not fix single-layer acceptance: step-0 MoE outliers
rise to 6.0532%, while step-1 streams improve to 5.0800% and still fail. The
candidate is archived and the working-tree attention kernel is unchanged.
This tests the per-block amplitude rule, not the vendor's complete sequential
accumulator-bit implementation.

## Simulator execution and local references

Indexer pool quantization explicitly limits reductions and stores to AIV lane 0
while both lanes participate in cube/vector transport. Sparse attention divides
padded head rows between the two lanes, so neither performs a maximum on an
empty replay tile. Prefill reloads the query in each sparse block to keep pipe
slots scoped to that block; its device performance cost is not yet measured.

RMSNorm reduces squares in 128-element FP32 chunks and then accumulates the
partial row sums, following the reduction structure in DeepSeek V4 RMSNorm.
Its CPU reference and scale tolerances are unchanged. The standalone BF16 linear
test uses PyTorch CPU FP32 matmul, matching its declared accumulation dtype,
instead of FP64 matmul rounded to FP32. The standalone FP32 check temporarily uses `atol=1e-5`, `rtol=1e-6`,
with zero outliers; its BF16 output remains subject to an exact check. The x86
simulator differed from CPU FP32 matmul by at most `5.24521e-6` on the default
fixture, while BF16 output was bitwise equal. The previous `atol=1e-6` failed
23 of 2,048 FP32 elements on that runner.

Increasing the K tile from 64 to 128 leaves simulator output bitwise unchanged.
Independent CPU partial sums reduce error against FP64 accumulation from
`5.24521e-6` to `9.53674e-7`, but a kernel replacement would require additional
partial-result reductions and transport; its performance has not been measured.
The production kernel retains native FP32 matmul accumulation with no software
compensation. These standalone validation changes do not change independent
layer or forward goldens or their thresholds, or establish multi-layer acceptance.

### Composed forward and dynamic shape bindings

The Q projection, query absorption and sparse-attention block statistics write
through their existing GM views using explicit `pl.assemble` calls without
rebinding the returned tensor. This preserves the buffer write while avoiding
unnecessary loop-carried GM tensor values. DeepSeek V4 prefill attention uses
the same in-place assembly form.

This avoids a compiler shape-binding issue in mixed-kernel expansion: fresh
scalar parameters receive new identities, while cloned tensor parameter types
retain the original dimension references. Remapped loop-carried types then
refer to different scalar identities with the same printed names. The affected
forward failed with 14 shape/valid-shape diagnostics before execution. No
arithmetic, buffer dimensions, reduction order or acceptance gates are changed
by this model-side adjustment; compiler verification remains enabled.

Local a2a3 verification passes for the affected standalone kernels and the
TP2/EP2 first-layer independent CPU golden. The four-layer forward now compiles
and executes, but its independent CPU golden still fails: step-0 hidden has
23.8083% outliers against the unchanged 5% limit. All nine captured-boundary
checks pass; these diagnostic checks do not establish multi-layer acceptance.

## Why W8A8, and why a2a3 forces it

The released checkpoint is FP8 blockwise (`e4m3`, 128x128, dynamic activations).
The a2a3 cube cannot execute it: `Intrinsic_mmad` in
CANN's `Ascend910_9392.ini` SoC config offers only `s32s8s8` / `u32u8u8` / `u8s8` and the
fp16/fp32 forms, with no fp8 or MX entry. INT8 is therefore the only quantized
path on this platform, and the deployment weights are the msmodelslim W8A8
conversion that vLLM Ascend serves on a 16-card A3 node. The int8 cube fractal
is MKN `16, 32, 16`, so every quantized K tile must be a multiple of 32.

## Checkpoint shape

| Property | Value |
| --- | ---: |
| Hidden size | 4,096 |
| Backbone layers | 45 (+1 MTP layer) |
| Linear-attention (KDA) layers | 34 |
| Sparse-attention (NoPE MLA + kpool indexer) layers | 11 (+ the MTP layer) |
| Attention heads | 64 |
| MLA `q_lora_rank` / `kv_lora_rank` | 1,536 / 512 |
| MLA `qk_head_dim` / `v_head_dim` / `qk_rope_head_dim` | 256 / 256 / **0** |
| Indexer heads / head dim / `index_topk` / `index_kpool` | 32 / 128 / 2,048 / 4 |
| KDA heads / head dim / conv kernel | 64 / 128 / 4 |
| Routed experts / active experts / shared | 288 / 8 / 1 |
| Dense layers / dense intermediate | 3 / 12,288 |
| MoE intermediate | 2,048 |
| Hyper-connection width / Sinkhorn iterations | 4 / 20 |
| Vocabulary | 154,880 |
| Maximum position | 1,048,576 |

The attention schedule is three KDA layers then one sparse layer, repeating:
layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 and 43 are sparse, every other
backbone layer is KDA. The first three layers (all KDA) use a dense MLP;
the remaining 42 and the MTP layer are MoE. Checkpoint layer 45 is the MTP
layer: an MLA layer with its own indexer and a sparse MoE, reached through
`enorm` / `hnorm` / `eh_proj`, and it carries **no** `hc_*` weights, so it uses a
plain residual rather than the four-stream mHC.

Summing every parameter group from the checkpoint index gives 321.2 G, which
matches the published 320 B total, so the reconstruction above is complete.

## What differs from DeepSeek-V4, which is the nearest port in this repo

`Glm5NextTextHyperConnection` subclasses `DeepseekV4HyperConnection` unchanged, so
the mHC coefficient algebra is identical to `models/deepseek_v4_1_flash/mhc.py`.
Three things are not:

- **The hyper-connection stream is BF16.** It is seeded by replicating the BF16
  embedding (`inputs_embeds.unsqueeze(2).expand(-1, -1, 4, -1)`) and every layer
  casts the mixes into that dtype. DeepSeek-V4 carries the stream in FP32, so the
  tile dtypes differ throughout and a kernel cannot be lifted across unchanged.
  At width 4 this is 32 KB of live residual per token instead of 64 KB.
- **The final collapse is an unweighted mean**, not a learned head applying the
  last layer's delayed pre-mix, so `mhc_head` takes no mix input.
- **There is no RoPE anywhere.** `qk_rope_head_dim` is 0 and the config validator
  rejects any other value, so the rope/nope split disappears from the main
  attention — and the indexer has none either, despite `indexer_rope_interleave`
  being set in `config.json`. That field is inherited from the GLM-MoE-DSA base and
  is dead here: `Glm5NextTextConfig` sets `rope_parameters = AttributeError()`,
  `Glm5NextTextModel.forward` passes `position_embeddings=None`, and
  `Glm5NextTextIndexer.forward` never touches a cos/sin table. So this port needs
  none of the `rope_tables.py` / interleaved-rope machinery that
  `deepseek_v4_flash_mtp` carries.

Two more differences bite at the kernel level: the router scores with `sigmoid`
where DeepSeek-V4.1 uses `sqrt(softplus(.))`, and 288 experts do not fit the
`SCORE_PAD = 256` sort window that `models/deepseek_v4_flash_mtp/gate.py` uses —
the pad has to grow to 512 and the `pl.sort32` + `pl.mrgsort` merge tree gains a
level.

## Deployment target

One Atlas 800 A3 node, 16 dies, TP16 with expert parallelism over the same 16
ranks, mirroring vLLM Ascend's recipe for this checkpoint
(`--tensor-parallel-size 16 --enable-expert-parallel --quantization ascend`,
MTP with 3 speculative tokens, `FULL_DECODE_ONLY` graph capture). Smaller TP/EP
shapes stay available for bring-up through `--tp` and `--ep`.

Per rank that gives 4 MLA heads, 4 KDA heads, 18 routed experts and 9,680
vocabulary rows. The **indexer is replicated, not head-sharded**: its score sums
over all 32 heads before the top-k, so sharding would force a cross-rank reduction
of partial scores on every sparse layer, and the kernel is small enough that 16-way
redundant compute is the cheaper trade. Weights come to about 21.6 GB per rank, of
which 19.5 GB is the routed experts; the remaining budget on a 64 GiB die
(~61.3 GiB visible) goes to the hybrid cache, which is the binding constraint and
is sized by the cache-manager work item.

### MoE expert parallelism

`decode_layer.py` and `prefill_layer.py` process each rank's local token rows.
Routed experts are distributed by expert ID across EP ranks; every local expert
retains its full intermediate dimension. The shared expert is a complete
replicated FFN on each rank, with no TP weight slices or output reduction.
Each rank adds its local shared output to the routed output returned by EP
combine. Token batches may differ across ranks.

The standalone MoE entries select `--ep` independently of attention TP. Their
fixtures use distinct rank-local activations and replicated weights for the
router, RMSNorm and shared expert. The generated routing fixture fixes eight
global experts spread across EP ranks to bound receive capacity; checkpoint mode
uses the saved router and correction bias. The common
configuration defaults attention TP to the selected EP size when `--tp` is
omitted; explicitly configured attention topologies retain their own checks.

Callers that hold a replicated attention stream must assign each token to one
MoE source rank before dispatch and restore their required output layout after
MoE. Those token-layout operations belong to the surrounding composition.

Prefill callers must also size the runtime heap for the receive capacity and
padded expert tiles. `prefill_moe.py` uses `prefill_ring_heap()` for routed
scratch. `prefill_layer.py` adds the capacity-sized Gate buffers, padded shared
expert buffers and FFN intermediates through `prefill_layer_ring_heap()`.
Both helpers retain a 16 MiB allocation reserve and round larger requirements
up to a power of two. At 64 rows per rank both entries keep the 256 MiB default
for EP2/EP4 and select 512 MiB for EP8 or 1 GiB for EP16 on ring zero. An enclosing
forward must additionally budget its own live allocations and pass sufficient
runtime capacity; see [runtime ring sizing](../../debug-and-tune/ring-heap-and-scope-stats.md).

### Vocabulary operators

`lookup_embedding.py` gathers from one BF16 vocabulary shard using global token
IDs and replicates each row into the four BF16 mHC streams. IDs outside that
shard contribute exact zeros. The caller sums these contributions within its TP
group before entering the backbone.

`lm_head.py` selects the hidden rows requested by `row_indices`, applies the
loaded final RMSNorm, rounds the normalized inputs to BF16, and projects against
one BF16 vocabulary shard with FP32 accumulation. It returns each selected row's
maximum score and **shard-local** token index; ties choose the lowest index.
A `-1` row index produces `(-inf, -1)` and contributes no sampling candidate.
Other row indices must lie within the input rows.

The caller adds the vocabulary-shard offset to valid local indices and selects
the global maximum within its TP group, breaking ties by the lowest global ID.
These local operators do not perform TP communication. The LM-head Torch golden
returns the full local FP32 logits; its executable adapter selects the requested
rows and compares the maximum and local index.

`lm_head_tp_reduce` combines the local LM-head candidates through TP windows. It
adds each rank's vocabulary offset, selects the maximum score and resolves ties
to the lowest global token ID. Invalid local IDs, NaN scores and `-inf` scores
contribute no candidate; an all-absent row returns `(-inf, -1)`. Its notification
epochs start at one and increase whenever a caller reuses the same windows.
This collective returns a global argmax candidate; sampling remains a caller
responsibility. Embedding contributions can use the existing `tp_all_reduce`
on one FP32 stream; the caller replicates the reduced BF16 embeddings into the
four identical mHC streams.

The LM-head executable includes bounded prefill/decode and boundary cases.
Validation defaults to TP2/EP2 on two devices and
also checks the TP reducer against its Torch golden. Three different input
epochs reuse the same windows and notification counts; cases cover every winning
peer, lowest-global-ID ties, all-absent rows, invalid local IDs and NaN/`-inf`
scores. With EP larger than TP, each TP group is checked independently.
Local projection cases and the distributed case run in separate processes.
Use `--case tp_reduce` for the collective alone, or a local case such as
`--case decode` with one device. `--compile-only` skips device execution:

```bash
python models/glm5_3_flash/lm_head.py -p a2a3 -d 0,1 --compile-only
```


## Decode backbone validation

`models/glm5_3_flash/decode_fwd.py` composes the implemented backbone operators
from prepared BF16 embeddings to the final normalized hidden states. The first
three layers run KDA and dense MLP; layer 3 runs DSA and MoE. Each sublayer uses
mHC pre/post processing, and row-parallel outputs use the common TP reduction.
The MoE dispatch source is rank 0 for the replicated batch; all EP ranks execute
their routed experts, and the following reduction restores replicated outputs.

The fixture uses TP2/EP2, 16 requests, four available layers, and two decode
steps; the default validation executes one layer. It reuses convolution/recurrent
state and the latent, raw-indexer and pooled-indexer caches across the steps.
Request state rows are reordered between
steps, and physical cache pages are non-identity mapped. Token embedding lookup,
LM head, sampling, MTP, checkpoint loading and serving integration are outside
this entry's validated scope. The full 45-layer checkpoint has not been validated.

Allocate two devices with the site's allocator, then run from the repository
root with their IDs in `TASK_DEVICE`:

```bash
PYPTO_CODEGEN_MAX_WORKERS=4 PYTHONPATH=. python models/glm5_3_flash/decode_fwd.py \
  -p a2a3 -d "$TASK_DEVICE" --ep 2 --tp 2 --layers 4 --steps 2 --end-layer 4
```

Optional `--validate-boundaries` replays the existing child goldens using
captured sublayer inputs. It also checks those inputs against the surrounding mHC operations,
checks final normalization, and compares every updated cache and KDA state.
Initial mutable tensors are snapshotted before either reference updates them.
TP copies and untouched state/cache rows must agree exactly. Numeric cache
checks cover written rows per rank and layer, so unused capacity cannot dilute
the outlier budget. The pooled INT8 cache uses the child cache operator's
one-LSB rounding allowance. All validation code lives in the forward entry.

The four trace fields in `layer_outputs` are attention output, MLP output,
MLP input, and normalized attention input. Attention and MLP use separate
normalized/output buffers so trace reads remain valid under asynchronous
scheduling.

The default invocation without `--validate-boundaries` uses an independent,
fully propagated end-to-end reference. The explicit four-layer, two-step
TP2/EP2 fixture still fails on A2/A3. Its thresholds are unchanged;
see the arithmetic and accuracy details below. A boundary-validation PASS
alone does not certify end-to-end numerical agreement; boundary mode also
prints the independent hidden-state gate and error distribution.

`--compile-only` checks compilation without using devices. `--start-layer`
isolates a schedule suffix starting from the provided embeddings; it does not
compute or restore the omitted prefix. The capacities in this bounded entry
are 20 KDA state rows, one latent-cache block per request, and 2,048 pooled rows.

### Compensated decode arithmetic

The decode path uses the shared kernels in `arithmetic.py` and `linear.py`
at FP32 reductions and BF16/INT8 rounding boundaries:

- `linear.py` splits each BF16 operand into an exact high/low BF16 pair.
  Four sixteen-element products accumulate with FP32 TwoSum residuals. mHC,
  all eight KDA front-end projections and the KDA output projection use the
  same implementation. The bottleneck outputs retain their BF16 boundaries.
- `arithmetic.py` reduces normalization statistics as FP32 high/low pairs,
  applies a compensated reciprocal-square-root correction, and evaluates
  exponential, sigmoid, SiLU and division with compensated FP32 arithmetic.
  Row reductions use two scaled integer slices and an FP32 residual reduction;
  the integer-slice sums fit in the FP32 significand. The exponential uses a
  degree-eight range-reduced polynomial. The former third integer slice and
  degree-nine/ten terms are removed. mHC softmax/Sinkhorn and the KDA recurrence
  also use these reductions. No device FP64 arithmetic is introduced.
- KDA projection, bottleneck/output projection and gate evaluation use
  dependent tasks. Each bottleneck is computed once per token tile. KDA
  recurrence readout reduces FP32 elementwise products; its FP32 state update
  order and state-row ownership are unchanged.
- Dense RMSNorm quantizes FP32 `x * gamma` and carries inverse RMS in its
  dequantization scale. Dense SwiGLU uses the corrected sigmoid and scale
  division. W8A8 storage, per-token activation quantization, saturation,
  scale encoding and the separate BF16 stream are unchanged.
- Sparse decode PV casts the weights to BF16 and performs one weighted-value
  product with FP32 accumulation. The online softmax maxima, denominators
  and block merge stay FP32. Probability residual compensation is removed.

Host references evaluate the selected dot products, reductions and unaries
in FP64 and round each result back at its original FP32 boundary. Products
and state updates remain FP32, and BF16/INT8 casts stay in their original
locations. This avoids CPU approximation and reduction-order error at later
rounding midpoints. Reference accuracy corrections and device arithmetic
fixes are both part of this change; comparisons with earlier reference
snapshots are not a pure kernel-only improvement measurement.

With single-product BF16 PV, the bounded independent TP2/EP2 comparison
passes all nine checks: latent,
raw-indexer and pooled-indexer caches, pool scales, convolution and recurrent
state, all four layer trace fields, residual streams and final hidden states.
Hidden relative L2 is approximately **0.4644%**, compared with **0.3793%**
with PV residual compensation. The saved reference outputs and initial mutable
states are identical between these runs. The existing
`atol=0.02`, `rtol=0.02`, per-output outlier budgets, exact replica checks and
untouched-state checks are unchanged. This is an independent propagated
reference result, not a boundary replay result.

The reduced arithmetic passes the same 65,536-input activation checks,
including a BF16 midpoint and finite sigmoid tails. BF16 SiLU/sigmoid outputs
are exact against the accurate reference; the sum and division probes also
match their FP32-rounded references exactly. The full forward's hidden,
streams, layer traces, convolution state and caches are elementwise identical
to the preceding single-PV version. Recurrent-state low bits differ within
the original gate. These are bounded validations, not a claim of correctly
rounded arithmetic for every finite FP32 operand.

#### Why the remaining compensation is retained

`arithmetic.py` is a new GLM-specific implementation choice, not a model
component required by the GLM architecture. The inspected Qwen and DeepSeek
implementations use native arithmetic rather than this software compensation.
The ablations below establish dependence of this fixture on the chosen accurate
host reference and acceptance checks; they do not establish that a deployed GLM
implementation must reproduce accurately rounded FP32 operations at every site.
The FP64-evaluated host reference was introduced during this precision work and
is stronger than simply retaining the original Torch FP32 implementation.

Forward validation also differs across the compared models. Qwen's stacked
decode validation obtains decoder reference outputs by running its device
decoder, then checks the final normalization, vocabulary projection and token
selection. DeepSeek V4.1's forward entry explicitly reports that numerical
accuracy was not validated. Neither is an equivalent independent, propagated
Torch reference with all nine GLM state/cache/output gates. The absence of
custom compensation in those implementations is therefore not contradicted
by these GLM ablation failures. The reference and per-output contract need to
be justified before treating this arithmetic library as a production requirement.

The comparison uses the same reference outputs, initial mutable states,
fixtures and thresholds. Each row below replaces one helper throughout the
backbone while retaining the others; every such direct rollback fails at
least one of the nine independent checks.

| Replacement | Hidden relative L2 | Result |
| --- | --- | --- |
| Ordinary BF16 GEMM with FP32 accumulation | 2.6546% | Fails caches, states and forward outputs |
| One BF16 product per K tile plus TwoSum | 1.5424% | Fails independent checks |
| Native FP32 division | 1.4375% | Fails independent checks |
| Native FP32 exp | 1.0144% | Hidden passes; caches/state/trace checks fail |
| Native high-precision rsqrt without correction | 1.3506% | Hidden passes; caches/state/trace checks fail |
| Ordinary row sums instead of FP32 pairs | 1.8031% | Fails independent checks |
| Ordinary RMS statistics and native high-precision rsqrt | 1.1059% | Hidden passes; caches/state/trace checks fail |
| Ordinary L2 statistics and native high-precision rsqrt | 0.7812% | Hidden passes; caches/state/trace checks fail |
| Native SiLU | 1.5186% | Fails independent checks |
| Native sigmoid | 1.2961% | Fails independent checks |

Two BF16 products splitting only the weights yield hidden relative L2
**0.4706%** and pass eight checks, but the independent pooled-cache check fails:
three of 3,072 written INT8 values per rank differ by one. The cache operator's
standalone check and forward boundary replay allow bounded one-LSB differences;
the independent forward check currently uses a stricter gate. Four products
are retained to preserve that existing gate, not because GLM mathematically
requires four GEMMs. Changing the gate is a separate validation-contract decision.

The normal arithmetic has clear precedents: Qwen's `decode_fwd.py` uses FP32
normalization statistics, ordinary GEMMs and native sigmoid; DeepSeek V4.1's
`hc.py` uses native sigmoid, softmax and Sinkhorn. DeepSeek V4.1's
`sparse_attention.py` and `vision.py` also use native high-precision division
or reciprocal. Substituting native high-precision division here still fails
the independent forward; replacing mHC coefficient arithmetic with the normal
DeepSeek-style path gives hidden relative L2 **1.3019%** and fails caches,
state and intermediate checks. These precedents support trying simpler
arithmetic, but do not establish equivalence at GLM's BF16 stream and subsequent
INT8 rounding boundaries. Small upstream rounding changes can alter an integer
code or the row maximum, then propagate across the next quantized layers.

The dense deferred-normalization quantizer has a direct reference in
`models/deepseek_v4_flash_mtp/gate.py`: quantize FP32 `x * gamma`, and carry
inverse RMS in the dequantization scale. The separate BF16 normalized output
is not reused as its quantization source. Accurate host references are retained
unchanged during these ablations; they do not add device operations.

The vector KDA readout is retained in the compensated implementation. Restoring
the FP32 GEMM at the smaller value tile reached native CANN compilation but
emitted a zero-extent vector tile move and failed before execution. A later
native-arithmetic rollback restored the original value tile of 64 and executed
the GEMM readout successfully, both standalone and in the TP2 forward. The
earlier compilation limitation therefore does not establish a general need for
the vector readout or a measured numerical advantage.

Validation entries remain in the owning kernel files. The retained checks
also cover ragged projection/output tiles, single-token output, zero/inactive
RMSNorm rows, non-aligned mHC tokens, reordered KDA state rows, ragged and
long convolution requests, and the dense MLP's final partial token block.
Single-device TP16-shaped operator checks do not constitute distributed TP16
validation. Additional compensation and task boundaries have not been
performance-tuned or benchmarked. The full 45-layer checkpoint remains
outside this validation.

### vLLM Ascend alignment experiments

Source inspection used vLLM Ascend commit `ff62595360a0`. The
[GLM Dense MLP](https://github.com/vllm-project/vllm-ascend/blob/ff62595360a069343ecdc3b2cd3a0daf45b4f1f0/vllm_ascend/models/glm5next/model.py#L99)
composes linear, activation, and linear calls. Its separate
[dynamic W8A8 linear](https://github.com/vllm-project/vllm-ascend/blob/ff62595360a069343ecdc3b2cd3a0daf45b4f1f0/vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py#L104)
returns the input activation dtype. The experiments therefore tested BF16
gate/up, SwiGLU, and local down-projection boundaries. FP32 ABI buffers were
retained as storage for widened BF16 values during this investigation.

The separate RMSNorm/quantization path and the custom fused
[RMSNorm dynamic quantization kernel](https://github.com/vllm-project/vllm-ascend/blob/ff62595360a069343ecdc3b2cd3a0daf45b4f1f0/csrc/attention/rms_norm_dynamic_quant/op_kernel/rms_norm_dynamic_quant_normal_kernel.h#L130)
have distinct rounding boundaries. The latter quantizes its internal FP32
normalized result directly. Both alternatives were evaluated; this does not
assert that every GLM Dense layer selects that fused kernel in vLLM Ascend.

All rows below use native arithmetic without `arithmetic.py`, ordinary BF16
GEMMs with FP32 accumulation, FP32 host references, TP2/EP2, four layers, two
decode steps, and 16 tokens. Acceptance thresholds and fixtures were unchanged.
Each newly adopted rounding boundary was mirrored in its reference.

| Cumulative candidate | Hidden relative L2 | Independent forward |
| --- | --- | --- |
| Mainline FP32 reference and native arithmetic | 3.0734% | FAIL |
| BF16 Dense linear and activation boundaries | 3.2156% | FAIL |
| Quantize FP32 normalized values in fused RMSNorm | 3.2938% | FAIL |
| Native high-precision Dense quantization division | 3.2938% | FAIL |
| BF16 attention projection before TP2 reduction | 3.5562% | FAIL |
| Native FP32 vector KDA readout | 3.4625% | FAIL |

All six candidates passed the nine existing forward boundary-replay checks.
The independent forward passed only the pool-scale and recurrent-state gates.
The RMSNorm plain/quant/add cases also passed. The earlier standalone Dense
PASS claim is withdrawn: its output lacked a `pl.Out` annotation, so the
harness did not perform an output comparison. The Dense and routed-expert
validation entries now declare their outputs explicitly. The fused quantizer's one-level INT8 error limit and mismatch-count
limit were retained; its former exact check against quantizing the emitted
BF16 output does not describe the fused FP32 contract.

The division experiment had bitwise-identical expected and actual outputs to
its preceding candidate. The vector-readout experiment retained exactly the
preceding projection candidate's reference and used the ordinary multiply/sum
structure of vLLM Ascend's
[Triton recurrent KDA](https://github.com/vllm-project/vllm-ascend/blob/ff62595360a069343ecdc3b2cd3a0daf45b4f1f0/vllm_ascend/ops/triton/kda/fused_recurrent_kda.py#L115).
In a separate seeded TP2-shape KDA case, matrix and vector readouts both passed
strict `atol=rtol=0.02`; they differed from the same BF16 reference at one and
three elements, respectively. Neither experiment establishes a general accuracy
advantage for changing the readout. Rows that introduce different rounding
contracts do not provide a fixed-reference accuracy comparison.

A CPU replay of the first-layer Dense input found relative L2 of 0.02808%
between the device and independent mixed inputs. Separate BF16 normalization
and fused FP32 normalization produced 31 and 25 changed INT8 codes,
respectively, each differing by one level. This demonstrates propagation
through quantization, without identifying the earliest arithmetic operation
responsible for the upstream difference.

These candidates remain experimental and have not replaced the working
kernels. The six measurements above are PyPTO ports of inspected operation
boundaries. Later native torch-npu comparisons and the checkpoint audit are
reported below; they do not establish whole-model vLLM equivalence.

### Checkpoint audit and native comparisons (2026-10-20)

Safetensors headers were inspected for the original checkpoint and its published
Ascend A3 W8A8 deployment artifact. No original and quantized tensors were mixed
in a fixture. The inventory includes auxiliary tensors and is not a parameter-count
estimate.

| Artifact | Safetensors files | FP8 tensors | INT8 tensors | BF16 tensors | FP32 tensors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original GLM-5.3-Flash | 62 | 37,338 | 0 | 1,141 | 37,629 |
| Published Ascend A3 W8A8 | 85, including auxiliary data | 0 | 37,338 | 1,392 | 74,719 |

All 90 mHC function tensors are **BF16 in both artifacts**, with shape
`[24, 16384]`. Widening them into vLLM's FP32 parameters changes execution dtype;
it does not recover checkpoint information lost by our BF16 storage. The original
mHC scale/base tensors and KDA `A_log`/`dt_bias` are FP32, while the published
W8A8 artifact stores them as BF16. RMSNorm weights are BF16 in both artifacts.
Published Dense weight scales are stored FP32, but the inspected
[W8A8 parameter allocation](https://github.com/vllm-project/vllm-ascend/blob/ff62595360a069343ecdc3b2cd3a0daf45b4f1f0/vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py#L84)
uses the model parameter dtype. The BF16 model experiment therefore rounds these
scales to BF16 before widening into the existing FP32 kernel buffers.

A native environment with Torch 2.8.0 and torch-npu 2.8.0.post2 was found and used
through allocator-managed devices. Actual published layer-0 Dense and layer-3
shared-expert weights were sliced for TP2. Each fixture has five input rows,
including an all-zero row. The reference runs native RMSNorm, dynamic quantization,
BF16 quantized linear, clipped SwiGLU, dynamic quantization, and BF16 down projection.

| Candidate component | Reference | Device result |
| --- | --- | --- |
| Dense local TP2 projection | Native Ascend with real layer-0 weights | PASS; final candidate bitwise equal |
| Shared expert | Native Ascend with real layer-3 weights | PASS; bitwise equal |
| RMSNorm plus activation quantization | Native Ascend with real layer-0 norm weights | PASS; BF16 output, INT8 codes, and scales bitwise equal |
| BF16 pool cache scatter/compression | Owning Torch reference | PASS, including preservation of incomplete/untouched rows |
| Indexer query/key/weight/gate projection | Owning Torch reference | PASS |
| BF16 KPool score | Inspected GLM Triton formula | PASS for both bounded score cases |

These are bounded component comparisons, not distributed native-vLLM runs.
The first Dense native comparison had relative L2 `3.8506e-7`; computing the
activation dequantization scale directly as `amax / 127`, rather than the
reciprocal of the quantization multiplier, made the final candidate bitwise equal.
The native all-zero row has a dequantization scale of zero. The safe denominator
for producing INT8 codes is kept separate from the returned scale.

### GLM KPool contract corrections in the candidate

The inspected GLM-specific source differs from the INT8/Hadamard indexer in the
working tree:

| Boundary | Working-tree path | Alignment candidate |
| --- | --- | --- |
| Completed pooled keys | INT8 plus FP32 dequantization scales | BF16 cache, no separate scales |
| Raw keys/gates and compression APE | Key/APE boundaries partly BF16 | FP32 compression inputs; BF16 key projection materialized before FP32 LayerNorm |
| Score | Hadamard rotation, query quantization, per-head ReLU | FP32 weighted query-head sum dotted with BF16 pooled keys |
| Incomplete pool event | Padded destination can be cleared | Destination is preserved |

The candidate removes the obsolete Hadamard matrix and pool-scale parameters,
updates all local callers and goldens, and tightens the old INT8 one-quantum cache
comparison to a BF16 tolerance. No independent-forward threshold was loosened.

Sources at the inspected revision:

- [BF16 cache and FP32 tail requirements](https://github.com/vllm-project/vllm-ascend/blob/ff62595360a069343ecdc3b2cd3a0daf45b4f1f0/vllm_ascend/models/glm5next/sparse_attn_indexer_kpool.py#L98).
- [GLM weighted-head score without per-head ReLU](https://github.com/vllm-project/vllm-ascend/blob/ff62595360a069343ecdc3b2cd3a0daf45b4f1f0/vllm_ascend/ops/triton/glm5_next_lightning_indexer.py#L286).
- [FP32 pool compression and BF16 storage](https://github.com/vllm-project/vllm-ascend/blob/ff62595360a069343ecdc3b2cd3a0daf45b4f1f0/vllm_ascend/ops/triton/glm5_next_kpool_tail_compress.py#L147).

The default CANN scoring operator is external and was unavailable in the native
environment. Score parity is established against the inspected **Triton formula**,
not against that opaque CANN implementation. The native environment also lacks
vLLM Ascend's custom mHC operator and the external FLA recurrent KDA package.
HcPre's source test explicitly models HF32 round-toward-zero matmul; equivalence
to that execution mode remains to be checked for synthetic FP32 function weights.
Actual BF16 checkpoint function weights have no additional mantissa bits to discard.

### Forward status after the contract migration

The clean candidate uses ordinary arithmetic, FP32 host references, the BF16
boundaries above, FP32 mHC execution parameters, and BF16 local attention/FFN
outputs before the existing FP32 TP exchange adapters. `arithmetic.py` is removed
from the candidate. It has **not replaced the working kernels**.

| TP2/EP2 experiment | Independent hidden relative L2 | Result |
| --- | ---: | --- |
| Clean BF16 KPool candidate, four layers, two steps, 16 tokens | 3.5828% | FAIL |
| Earlier BF16 KPool prototype, real first-three-layer weights and synthetic fourth-layer weights | 3.4374% | FAIL |

The clean candidate passes all eight boundary-replay checks, including the normal
BF16 pool-cache gate; this does not make its independent forward pass. Removing
obsolete layout entries changes synthetic initializer seeds, so these rows are
not a fixed-fixture accuracy improvement comparison with earlier candidates.
The mixed fixture also uses actual published input embedding rows and final norm
weights. It is **not** a complete real-weight four-layer model validation.

In the mixed fixture's first decode step, layer-0 normalized attention input is
bitwise equal, attention-output relative L2 is 0.07417%, mixed FFN-input relative
L2 is 0.09392%, and FFN-output relative L2 is 1.37046%. Thus divergence is present
before the DSA indexer and is amplified across subsequent quantization boundaries.
Same-input Dense agreement does not explain the earliest upstream difference.
The remaining work is to isolate the KDA/mHC difference with native-equivalent
intermediates and then rerun the independent forward. Three-layer-only validation
also hits an unresolved integer-bound failure with a zero DSA/MoE branch;
no successful three-layer run is claimed.

The candidate patch, checkpoint summary, source hashes and run logs are saved
locally under `build_output/glm53_vllm_alignment/`. The patch is relative to the
current uncommitted working tree, not repository HEAD; it has passed
`git apply --check`. Only the two standalone output annotations and this report
are applied in the working tree for this investigation. No serving repository,
LMHead, embedding implementation, sampling, or MTP work was added.

### Official-checkpoint KDA/mHC validation (2026-10-20)

The next experiment uses **only the original official checkpoint**. It loads
layer-0 KDA projections, convolutions, output norm/projection, both mHC sites,
both RMSNorm weights and embedding rows 1000-1031. It preserves the checkpoint's
FP32 mHC scale/base and `A_log`/`dt_bias`; BF16 mHC function weights are widened
for the candidate's FP32 execution interface. No published W8A8 weights, FP8 FFN
conversion or FFN computation is involved.

Validation uses the saved ordinary-arithmetic alignment candidate, not the
compensated working kernels. The probe composes the existing operators and
exposes their outputs as ordinary `pl.Out`/`pl.InOut` tensors. The temporary
instrumentation and fixture generator remain local artifacts rather than new
model test files.

The distributed chain covers:

```text
embedding rows -> mHC mixes/pre -> RMSNorm -> KDA projections
  -> convolution -> recurrent KDA -> gated output norm/projection
  -> BF16 local projection boundary -> TP2 all-reduce
  -> mHC post -> FFN-side mHC mixes/pre -> RMSNorm
```

It runs 16 token rows through two decode steps with reordered state slots, zero
initial states and independent expected-state continuation. Both TP weight slices
are real. The second step exercises convolution/recurrent state updates and reuse
of the collective windows. This is a bounded first-layer test, not a prefill-derived
cache, full checkpoint forward or generated-text evaluation.

| Comparison | Result |
| --- | --- |
| Each TP2 slice, two local decode steps, 14 captured outputs per step | PASS at unchanged `atol=rtol=0.02`, zero allowed outliers |
| Distributed TP2, two steps, all 18 exposed outputs/states | PASS at the same strict gate |
| Same-input native Ascend RMSNorm and merged KDA projection | BF16 RMSNorm, QKV, FP32 decay/beta and output gate bitwise equal in all four local cases |
| Same-input gated-output formula followed by native BF16 output linear | BF16 local projection bitwise equal in all four cases |

The distributed run used allocator devices 0 and 6 and ended with
`[RUN] PASS (16.46s)`. Relative L2 against the independent CPU FP32 chain was
0.05788% for reduced attention and 0.09448% for the FFN normalized input.
These CPU-chain figures are distinct from the native-reference figures below.

#### Earliest reference difference

The first difference against the CPU chain is in mHC's FP32 projection. In the
first token batch, raw projection relative L2 is `2.0969e-6` against CPU FP32;
against an FP64 diagnostic rounded to FP32 it is `4.7885e-7`. FP64 is used only
for this diagnostic, not as a replacement golden. The inverse-RMS difference is
smaller (`5.3151e-8` relative L2). Exposing these intermediates left the original
pre-mix, mixed activation, normalized input and KDA partial bitwise unchanged.

Replaying the CPU sigmoid with the **actual** raw projection and inverse RMS
reduces pre-coefficient relative L2 from `2.3224e-6` to `3.9804e-8`. An independent
native primitive composition gives exactly the same BF16 mHC mixed activation as
PyPTO in this batch, despite the CPU-chain difference at 33 mixed-activation
values. This evidence does not justify adding compensation to reproduce the
CPU FP32 reduction order.

#### Independent native primitive/formula chain

A separate native reference starts from the official weights and embedding rows,
maintains its own two-step state, uses native BF16 merged linears and RMSNorm,
and evaluates convolution, recurrent update and mHC with explicit FP32 formulas.
It reconstructs the TP2 sum from the two local BF16 partials. It does **not** run
vLLM's external recurrent/convolution kernels, custom HcPre/HcPost or native HCCL.
The formula reference follows pypto-lib's state indexing; vLLM cache-manager and
null-slot metadata compatibility are not validated here.

| Boundary, per rank | Step 0 relative L2 | Step 1 relative L2 |
| --- | ---: | ---: |
| Reduced attention | 0% | 0.01833% |
| FFN normalized input | 0.005122% | 0.02945% |
| Different INT8 codes after native quantization | 0 / 65,536 | 6 / 65,536, each by one level |
| Different native dequantization scales | 0 / 16 | 0 / 16 |

The BF16 boundary checks pass at `atol=rtol=0.02` with zero allowed outliers.
INT8-code deltas are reported separately and are not accepted through a BF16
activation gate. At the second step the explicit recurrent formula differs from
PyPTO at 7 and 8 BF16 readout values for the two ranks. This is not evidence of
parity with the unavailable external fused recurrent operator.

No kernel replacement, new compensation or acceptance-threshold change was made
from this experiment. The earlier four-layer W8A8 independent-forward failure
remains unresolved. The official-weight result validates the tested attention and
mHC portion only; continuing through the official FFN requires a correctly
FP8-dequantized reference or an explicitly identified derived INT8 fixture.
The reports, kernel snapshot, replay scripts and logs are saved locally in
`build_output/glm53_official_kda/`.

### Official FP8 FFN fallback reference (2026-10-20)

The inspected vLLM Ascend revision already provides a block-FP8 checkpoint loader.
Its [block scale resolver](https://github.com/vllm-project/vllm-ascend/blob/ff62595360a069343ecdc3b2cd3a0daf45b4f1f0/vllm_ascend/quantization/methods/w8a8/fp8_block.py#L67)
multiplies each FP8 value by its FP32 `weight_scale_inv` for the corresponding
128x128 block, then converts to the model dtype. On A3, the
[post-load path](https://github.com/vllm-project/vllm-ascend/blob/ff62595360a069343ecdc3b2cd3a0daf45b4f1f0/vllm_ascend/quantization/methods/w8a8/fp8_block.py#L199)
keeps that resolved dense weight and
[executes an unquantized linear](https://github.com/vllm-project/vllm-ascend/blob/ff62595360a069343ecdc3b2cd3a0daf45b4f1f0/vllm_ascend/quantization/methods/w8a8/fp8_block.py#L250).
The MXFP8 branch is selected for Ascend 950. This is an existing source path,
not evidence that A3 directly executes the checkpoint's FP8 GEMM.

Therefore, the BF16-model official-checkpoint reference on A3 has **no INT8
activation boundary in this FFN**. The six changed INT8 codes reported in the
previous experiment describe W8A8 sensitivity, not a required boundary of the
official FP8 fallback. The working PyPTO Dense kernel still implements W8A8;
its weight/input contracts differ from this reference.

Layer-0 gate/up weights are FP8 `[12288, 4096]` with FP32 scales `[96, 32]`;
down weights are FP8 `[4096, 12288]` with scales `[32, 96]`. An independent
block-axis implementation agrees bitwise with the inspected source resolver for
all three complete matrices. Scales remain FP32 during resolution. Only the
original checkpoint was read; no deployment tensors or INT8 conversion were used.

The native reference applies BF16 merged gate/up linear, clipped SwiGLU with
`alpha=1`, `bias=0`, `limit=10`, BF16 down linear, and a BF16-rounded sum of the
two TP slices. Both slices execute on one allocated A3 device; the sum is
reconstructed on the host. It is **not** a distributed FFN/HCCL execution.
Three input sets traverse this same reference: captured PyPTO FFN inputs,
independent native primitive/formula inputs, and independent CPU-chain inputs.
A zero-input row also verifies the bias-free zero-output boundary.

| Native reference fed by captured PyPTO input versus native-chain input | Step 0 relative L2 | Step 1 relative L2 |
| --- | ---: | ---: |
| FFN normalized input | 0.005122% | 0.02945% |
| Reduced FFN output | 0.09576% | 0.13780% |
| Output streams with fixed native mHC coefficients and residual | 0.06001% | 0.09526% |

For comparison, using the CPU-chain input instead of the native-chain input
produces FFN-output relative L2 differences of 0.24214% and 0.27056% against the
captured-input result. These are input-sensitivity measurements through the same
native FFN, not measurements of a PyPTO BF16 FFN kernel.

With identical inputs, native-versus-CPU references for gate/up, activation and
local down output all pass `atol=rtol=0.02` with zero allowed outliers. Local
output relative L2 is 0.03577% and 0.03326% for the two TP slices. The output
sensitivity and fixed-mHC-post checks pass the same gate. The allocated device-0
job ends with `OFFICIAL BF16 FFN REFERENCE/SENSITIVITY PASS []`.

No current PyPTO W8A8 kernel was marked as supporting these official FP8 weights,
and no BF16 model kernel, INT8 requantization, compensation or threshold change
was added. A source-aligned BF16 FFN implementation and its integration would be
needed before claiming that the existing PyPTO forward supports the official
checkpoint. The earlier four-layer W8A8 precision failure remains a separate
unresolved result; changing weights and execution contracts is not a fixed-reference
accuracy improvement comparison. Replay scripts, resolution manifest, native
outputs and measurements are stored locally under `build_output/glm53_official_ffn/`.


### BF16 Dense MLP implementation and TP2 integration (2026-10-20)

`dense_mlp.py` now also implements the A3 BF16 fallback contract. Gate/up GEMM
uses BF16 operands and ordinary FP32 accumulation, then rounds to BF16. Clamped
SwiGLU computes in FP32 and rounds its result to BF16 before the down GEMM. The
local down output is BF16. This path uses no INT8 activation quantization and no
high/low compensation. `linear.py::bf16_linear_native` supplies the ordinary
GEMM; the previous compensated primitive and its existing callers are retained.

The owning operator file provides the tests. Default BF16 cases cover one token
and 17 tokens, including a zero row and inputs that exercise activation clamps.
The original INT8 entry remains the default. Examples, using an allocated device:

```bash
PYTHONPATH=. .venv/bin/python -m models.glm5_3_flash.dense_mlp --tp 2 --dtype bf16 -d <allocated-device>
PYTHONPATH=. .venv/bin/python -m models.glm5_3_flash.dense_mlp --tp 2 --dtype bf16 --weights <resolved-weights.pt> --tp-rank 0 --num-tokens 97 -d <allocated-device>
PYTHONPATH=. .venv/bin/python -m models.glm5_3_flash.decode_fwd --tp 2 --ep 2 --layers 1 --steps 2 --dense-dtype bf16 -d <allocated-device-pair>
```

The optional Dense fixture contains BF16 `w_gate_up[TP,2*local_inter,D]`
(gate rows followed by up rows) and `w_down[TP,D,local_inter]`. It contains
already resolved weights, not the raw checkpoint. The standalone forward CLI
uses generated fixtures; official-weight injection is an archived local probe.
The original FP8 checkpoint resolution and its manifest are described above.

Both official TP2 slices pass against native Ascend on 97 identical input rows
at `atol=rtol=0.02` with zero allowed outliers. Relative L2 errors are 0.004389%
and 0.003882%, with maximum absolute error 0.00003052 for both slices. The final
zero row is exactly zero. The original compensated linear regression also passes,
including its exact BF16 output check.

`decode_fwd.py --dense-dtype bf16` uses plain RMSNorm and the new Dense kernel,
widening its already rounded BF16 local output for the existing TP exchange.
Mode selection occurs at module import, like the existing TP/layer dimensions.
Unused scale parameters retain the existing forward ABI. Default INT8 forward
compilation still passes for four layers and two steps. The BF16 switch changes
only Dense execution; MoE, attention and cache contracts remain as before.

The actual TP2 first-layer forward, using original-checkpoint weights, 16 requests
and two decode steps, passes its independent CPU reference. Attention, Dense MLP,
both reductions, mHC, final stream mean/RMSNorm and persistent convolution/recurrent
states execute together. All checked values have zero outliers at the existing
0.02 absolute/relative tolerances; TP replicas agree and unused state/cache rows
are preserved. This uses the current working backbone, including its previous
compensated attention/reference kernels, rather than the isolated ordinary candidate.
The official fixture starts from zero KDA states and includes one continuation;
it does not validate states produced by real prefill. Unexecuted DSA/MoE weights
are zero placeholders in this one-layer experiment.

The separate independent native primitive/formula reference is a stricter
remaining alignment check. Final-hidden relative L2 is 0.38661% and 0.51379%
for steps 0 and 1. Step 1 has **19/65536 elements outside** `atol=rtol=0.02`;
step 0 has zero. This zero-outlier diagnostic therefore fails. Output-stream
relative L2 is 0.34075% and 0.37159%, with zero outliers at both steps. An existing
contract difference is upstream of the MLP: the working attention path reduces
FP32 local projections, whereas the inspected native reference rounds each local
projection to BF16 before reduction. Compensated versus ordinary attention/mHC
arithmetic also differs. Their individual contributions remain to be isolated;
the Dense same-input result does not establish full native forward agreement.

These results do not resolve the earlier four-layer ordinary-candidate W8A8
failure, establish full-model official-checkpoint support, or cover vLLM custom
cache managers/native HCCL. Scripts, source snapshots and reports are archived
locally in `build_output/glm53_bf16_dense/`. No serving, embedding or LM-head work
is included.


### KDA alignment experiment and integration regression (2026-10-21)

**Initial experiment status: broad candidate preserved, integration rolled back.** The ordinary-arithmetic
candidate improves first-layer native agreement but regresses the four-layer INT8
reference test. The source files were initially restored byte-for-byte from the pre-alignment
snapshots after that baseline passed the same test. The results in this subsection
describe that broad candidate. The subsequently adopted, narrower fixes are
described in the next subsection.

The original first-layer inputs, official weights, independent native reference,
and `atol=rtol=0.02` zero-outlier diagnostic were held fixed during isolated
experiments. No boundary replay was substituted for the independent forward.

| Isolated first-layer variant | Step 0 hidden relative L2 | Step 1 hidden relative L2 | Step 1 outliers |
| --- | ---: | ---: | ---: |
| Previous working implementation | 0.38661% | 0.51379% | 19/65536 |
| BF16 local KDA output before TP sum | 0.13597% | 0.25570% | 7/65536 |
| Ordinary KDA and FP32 mHC, with recurrent sqrt/division and vector readout | 0.07446% | 0.11711% | 2/65536 |

The local KDA linear result must round to BF16 **before** the TP sum. The exchange
continues to carry FP32 values; the candidate carries widened BF16 results. This
boundary was implemented in the candidate `kda_output.py` for all callers.
The owning validation also checks that exchange values lie exactly on the BF16
grid and delegates the numerical check to the existing harness comparator.

The candidate KDA projection/output, convolution, recurrent body and mHC use
ordinary FP32 arithmetic in place of the earlier high/low compensated operations.
Projection GEMMs reuse `bf16_linear_native`, with BF16 output boundaries between
bottleneck linears. mHC widens stored BF16 function-weight tiles to FP32 for its
projection, retaining the stored-weight ABI. The corresponding KDA and mHC goldens
use FP32 operations. Plain RMSNorm, INT8 quantization and the DSA path retain their
previous implementations; `arithmetic.py` still has those consumers.

The recurrent normalization and readout follow the inspected
[vLLM Ascend recurrent kernel](https://github.com/vllm-project/vllm-ascend/blob/ff62595360a069343ecdc3b2cd3a0daf45b4f1f0/vllm_ascend/ops/triton/kda/fused_recurrent_kda.py#L115):
q/k divide by `sqrt(sum(x*x)+1e-6)`, and state readout uses an FP32 vector reduction.
Replacing those with reciprocal-square-root multiplication or a cube matmul changes
rounding. No new compensation was added. Ordinary RMSNorm was also tested in
isolation and was not retained: it did not eliminate the remaining outliers.

With identical attention-normalized inputs, the previous compensated projection
changes 65 and 70 BF16 QKV values over the two steps compared with the ordinary
projection independently verified against native Ascend. In the final isolated
source-aligned stage probe, QKV, decay, beta, output gate, recurrent BF16 output,
local projection, and TP2 attention output match the native reference bitwise at
both steps. One pre-normalization query element differs at step 1, but the stored
BF16 recurrent output is identical. This is a first-layer fixture result, not a
claim of bitwise equivalence for arbitrary states or inputs.

The remaining two final-hidden failures are downstream of the now-matching
attention result. Step 1 positions `(10,765)` and `(10,4025)` have errors
0.02734375 and 0.0234375, against allowed errors approximately 0.02334 and 0.02305.
The FP32 mHC path still changes 29 BF16 mixed-input elements at that step compared
with the independent native formula reference. The full-forward zero-outlier
requirement therefore remains **unmet**; neither the small failure count nor
passing the existing CPU gate establishes completion.

The candidate, temporarily applied to the working tree, reproduces the isolated result
in the official TP2 first-layer run
exactly and passes the existing independent CPU-reference gate. That CPU gate
allows a fraction of elementwise mismatches; it is distinct from the zero-outlier
native diagnostic above. The corrected operator regression passes projection and
output single/ragged cases, convolution prefill/decode cases, recurrent single/batch
cases with nonzero states, and all four mHC cases (component default TP16 local shapes on one device).
Its first submission failed at
argument parsing because the component CLIs do not accept `--tp`; no kernels ran
in that initial submission.

The same-input native FFN probe is bitwise identical at both decode steps
for the reduced FFN output. The probe does not separately compare each local
projection output. A separate same-input mHC probe finds coefficient differences up to
`1.55e-6` in the FFN pre-mix. Applying those coefficients to identical streams
changes 32 BF16 mixed values across the two steps. This localizes a remaining
rounding difference before the FFN rather than establishing a Dense MLP defect.
An isolated mHC `sqrt`/division and sigmoid-division trial, motivated by the
AscendC source, leaves the final hidden result and both outliers unchanged and
was not adopted. Source alignment does not imply that the primitive/formula
reference executes the custom mHC operator: the inspected `ProcessYHc4` uses
fused multiply-add for stream collapse, whereas this reference evaluates
separate multiplication and reduction. Direct custom-op validation is still
needed before claiming exact vLLM mHC equivalence.

The candidate four-layer, two-step INT8 TP2 regression executes but fails the
independent CPU-reference gate. Its first reported layer-output failure is step 0,
layer 1, FFN output (`phase=1`): 27.478% elementwise mismatches versus the existing
5% allowance. Cache, stream and final-hidden checks also fail. This is a separate
multi-layer validation failure; the passing first-layer BF16 result does not cover
it. The preserved pre-alignment source passes all outputs in the same test
(`PASS (140.01s)`), confirming a regression introduced by the candidate relative
to the existing independent reference. The candidate was therefore rolled back;
its source and logs remain archived. At that rollback point the
first-layer native baseline retained 19 outliers, not the candidate's two.
The subsequent investigation below isolates a scale-rounding mismatch.
No tolerance was relaxed.

The independent native reference reconstructs primitives/formulas and the TP sum;
actual vLLM Triton recurrent execution is unavailable in the reference environment
because Triton/vLLM are not installed. No custom-op or native HCCL equivalence is
claimed. Local source snapshots, ablations and stage captures are archived under
`build_output/glm53_native_alignment/`.

### INT8 scale rounding and the adopted KDA boundary (2026-10-21)

The ordinary KDA-only and mHC-only groups both fail the four-layer independent
reference. Same-input FFN replay has zero tolerance violations across all three
Dense layers and both steps for each group. In the mHC-only case, just six BF16
mixed-input differences at step 0/layer 0 change four INT8 codes; later layers
amplify those differences. This does not establish a local Dense MLP defect.

Testing only BF16 local KDA projection rounding isolates a separate mismatch:
the first two attention outputs and FFN mixed inputs are bitwise identical, but
one second-layer FFN output differs at each step. Capturing RMSNorm and both
local Dense outputs shows that normalized BF16 values and INT8 codes match;
one FP32 dequantization scale differs by one ULP. Given the captured scale,
both local Dense outputs match the independent reference bitwise.

The scale mismatch comes from operation order. The in-place A3 `TDIVS` path used
by the old kernel evaluates division through multiplication by a rounded reciprocal.
The old kernel therefore
computed `amax * float32(1/127) * inverse_rms`, while the CPU reference computed
`(amax / 127) * inverse_rms`. At step 0/row 14 these produce
`0.031914882361888885` and `0.031914886087179184`. That small difference changes
one BF16 FFN result and propagates into later quantization decisions.

`rmsnorm.py` now reads the row maximum as a scalar and computes scalar division,
following the existing [Qwen quantizer](../../../models/qwen3_14b/decode_layer_a8w8.py).
The scale still uses the existing tile-store path for active and inactive rows.
The reference is unchanged and no compensated arithmetic was added. The owning
scale check now requires exact agreement: the old implementation fails the
17-token case by one ULP, while the fix passes 1-, 5-, and 17-token cases.

`kda_output.py` now rounds the local projection to BF16 before widening to FP32
for TP exchange. Its golden follows that boundary, and its owning comparator
checks that exchange values lie on the BF16 grid. Other KDA/mHC arithmetic remains
at the previously validated implementation; the broad ordinary-arithmetic
candidate is still isolated.

| Validation | Result |
| --- | --- |
| Four-layer INT8 TP2, two steps, independent reference | PASS, all outputs, 93.08 s |
| Same-input scale and local Dense outputs at the first divergence | Bitwise identical, both steps |
| RMSNorm quantization, 1/5/17 tokens including inactive rows | PASS, exact scales and INT8 codes |
| KDA output single/ragged, default TP16 local shapes | PASS |
| Official-weight first-layer BF16 TP2, existing CPU gate | PASS, 41.07 s |
| Frozen native first-layer hidden, step 0 | 0.135970% relative L2, zero outliers |
| Frozen native first-layer hidden, step 1 | 0.255702% relative L2, 7/65536 outliers |

The current implementation improves the native diagnostic from 19 to 7 outliers
while passing the four-layer INT8 reference. The native zero-outlier requirement
remains unmet. The native dynamic-quant API rejects this FP32 deferred-normalization
input, accepting only FP16/BF16; the scale probe therefore does not claim direct
native quantizer equivalence. No serving or actual vLLM custom-op execution is
included. Experiments, source diffs, and logs are preserved in
`build_output/glm53_int8_regression/`.

### Follow-up arithmetic isolation (2026-10-21)

The following trials keep the frozen official-weight inputs, native reference,
toolchain, and tolerances unchanged. All source variants remain isolated; none
was adopted. Native hidden outliers below use `atol=rtol=0.02` with zero allowed
outliers, separately from the existing CPU forward gates.

| Isolated trial | Native hidden outliers, step 0 / step 1 | Other validation |
| --- | --- | --- |
| Current working implementation | 0 / 7 | Previously passed four-layer INT8 TP2 |
| Ordinary candidate, mHC without split-K, K tiles 64 and 256 | 0 / 5, identical for both tiles | First-layer CPU gates PASS |
| Current implementation, only Q/K sqrt-and-division normalization | 0 / 7 | First-layer CPU gate PASS |
| Ordinary candidate, separate vector operations for mHC logits | 0 / 2 | First-layer CPU gate PASS; no outlier improvement over ordinary baseline |
| Current implementation, ordinary KDA projection and block-16 CPU accumulation | 0 / 5 | First-layer CPU gate PASS; four-layer INT8 FAIL, 138.47 s |
| Earlier ordinary KDA/mHC candidate with the adopted scale fix | Not rerun against native | Four-layer INT8 FAIL, 86.84 s |

The projection trial does not justify replacing the CPU reference. A separate
same-input audit across Q/K/V, both ranks and both steps finds 1--9 differing
BF16 outputs per projection with block-16 accumulation. It is not an exact native
GEMM emulator. CPU BF16 matmul is also unsuitable as a drop-in reference here:
the rank-0 Q projection differs from native at 36,249 and 36,299 elements across
the two steps, compared with 21 and 12 for FP32 matmul followed by BF16 rounding.

The scale fix alone does not rescue the ordinary KDA/mHC candidate. Same-input
Dense replay is bitwise identical at all three Dense layers and both steps.
At step 0/layer 0, 244 BF16 mixed-input differences change 21 INT8 codes; the
independent FFN comparison then has 818 tolerance violations. This localizes the
amplification to upstream rounding and quantization decisions rather than a
same-input Dense computation defect.

The working source therefore retains the previously validated arithmetic and
the two adopted boundary/scale fixes. Its native zero-outlier requirement is
still unmet; passing CPU gates does not close that requirement. The reference
still uses independent native primitives/formulas, not actual vLLM custom ops.
Source variants, scripts, metrics and logs are archived locally under
`build_output/glm53_remaining_validation/`.

### Retained-implementation causal trace (2026-10-21)

A fresh two-step TP2 capture uses the retained kernels and frozen official
weights. Its normalized attention inputs, attention outputs and FFN mixed inputs
are bitwise identical to the retained full-forward capture. Regenerated native
stage tensors reproduce the frozen native tensors bitwise. A separate second-step
replay starting from captured first-step states reproduces the full-forward
hidden output bitwise, including all seven outlier coordinates.

The attention-side mHC coefficients differ slightly, but their BF16 mixed inputs
and subsequent RMSNorm outputs are bitwise identical to native in both steps.
The first activation differences appear at KDA projection: QKV differs at 37/28
elements on the two ranks in step 0, and 28/42 elements in step 1. Gate outputs
also differ. This establishes a projection arithmetic difference on identical
normalized inputs; it does not establish which implementation matches the actual
vLLM custom-operator chain more closely.

Controlled substitutions give the following second-step native hidden results.
These are localization experiments, not replacement references or accepted fixes.
All downstream kernels and the zero-outlier threshold remain unchanged.

| Substitution | Outliers | Relative L2 |
| --- | --- | --- |
| Captured current first-step states, no substitution | 7 | 0.255702% |
| Native first-step Conv state only | 5 | 0.248167% |
| Native first-step recurrent state only | 7; hidden bitwise unchanged | 0.255702% |
| Both native first-step states | 5 | 0.248167% |
| Native projection outputs in both steps | 5 | 0.196681% |
| Native attention output in both steps | 3 | 0.117162% |
| Native FFN mixed input only | 1, at a new coordinate | 0.070506% |
| Native attention output and FFN mixed input together | 0 | 0.007917% |

Native Conv history removes the outliers at `(10, 2620)` and `(10, 3648)`.
Conv state writeback itself is bitwise equal to the expected shift-and-append
operation using the captured QKV values. Thus the historical discrepancy is
inherited from projection values, not an incorrect cache write or row mapping.
Replacing recurrent history alone has no effect on final hidden values in this
fixture; this does not establish equivalence for arbitrary recurrent inputs.

Replacing the attention output still leaves `(6, 3704)`, `(10, 765)` and
`(10, 4025)`. Replacing only the FFN mixed input removes the original seven but
introduces `(5, 1957)`, because the residual branch retains the original attention
result. Replacing both boundaries removes all outliers, localizing the remaining
investigation to KDA arithmetic and the mHC path before FFN. It does not prove
that every downstream operator is bitwise equivalent.

The working kernels are unchanged and still have seven outliers. Actual
`npu_hc_pre_v2` and `npu_hc_post` are not registered in the native reference
environment. Direct comparison with those vLLM-Ascend operators remains necessary
before changing mHC arithmetic to match the reconstructed formula. Captures,
state replays, isolated substitutions and source hashes are archived locally in
`build_output/glm53_state_trace/`.

### Actual vLLM-Ascend mHC operator comparison (2026-10-21)

The reference environment can now execute the pinned vLLM-Ascend `HcPre` and
`HcPost` kernels through a temporary ACLNN bridge. The installed CANN build
template builds only these two operators, with the source compiler options;
all 24 operator source files match the pinned checkout byte for byte. Build
artifacts and custom-op registration paths are local to the diagnostic process.
The shared environment and working model kernels were not modified.

The comparison uses the three outputs consumed by `npu_hc_pre_v2`: mixed input,
post coefficients and residual coefficients. The extra optional pre-coefficient
buffer is not part of that API's returned contract and is excluded from the
diagnostic. On identical retained inputs at the original 16-token shape:

| Boundary | Step 0 | Step 1 |
| --- | --- | --- |
| Attention-side HcPre mixed input, PyPTO vs custom | Bitwise equal | Bitwise equal |
| FFN-side HcPre mixed input, PyPTO vs custom | 9 differing BF16 values | 9 differing BF16 values |
| FFN-side HcPre mixed input, Torch formula vs custom | 17 differing BF16 values | 6 differing BF16 values |
| HcPost with identical inputs and coefficients, both sites | Bitwise equal | Bitwise equal |

The full first-layer reference was then rerun with actual custom mHC operators,
keeping native Torch KDA, Dense, normalization and the reconstructed TP sum.
This is a hybrid reference, not a complete vLLM execution: recurrent Triton/FLA
kernels and native HCCL are still outside its coverage. The original frozen
formula reference remains unchanged.

| Implementation compared with the custom-mHC hybrid reference | Step 0 outliers | Step 1 outliers |
| --- | --- | --- |
| Retained implementation | 0 | 4 |
| Earlier isolated ordinary-arithmetic candidate | 0 | 0 |
| Isolated ordinary projection only | 0 | 2 |
| Native projection and native recurrent-core outputs substituted | 0 | 2 |
| Native attention output substituted | 0 | 0 |
| Only input/output projection GEMMs changed to ordinary accumulation | 0 | 0 |

All rows use the original `atol=rtol=0.02` and zero allowed outliers. The retained
step-1 outliers against this hybrid reference are `(10, 2620)`, `(10, 3648)`,
`(15, 2283)` and `(15, 3326)`. The last two persist even when projection gates and
recurrent-core outputs are native, narrowing that contribution to KDA output
normalization/projection. HcPost does not need a precision workaround for the
tested same-input cases.

An owning KDA output probe further separates normalization from GEMM. For step 1,
token 15, both ranks' BF16 gated-normalization inputs are bitwise identical to
native. The local output projection differs at just column 3220 on rank 0
(`2.9243528842926025e-7` versus `2.905726432800293e-7`) and column 1419 on rank 1
(`-8.392333984375e-5` versus `-8.344650268554688e-5`). Replacing only output GEMM
accumulation in the native-core substitution removes the remaining two outliers.

The minimal non-substitution candidate changes only `bf16_linear` calls to
`bf16_linear_native` in `kda_projection.py` and `kda_output.py`; all nonlinear
arithmetic, state updates, golden formulas and tolerances remain unchanged.
It passes the official first-layer CPU gate in 38.79 s and has zero outliers
against the custom-mHC hybrid reference, but still has three against the frozen
formula reference and fails the unchanged four-layer INT8 gate in 90.08 s.
The candidate is therefore isolated, not adopted. A native GEMM reference for
the four-layer quantized chain is the next check needed to distinguish backend
rounding differences from an implementation regression before changing the
accepted arithmetic or golden contract.

The original formula reference itself has two step-1 outliers against the
custom-mHC hybrid reference. Consequently, the earlier seven-outlier diagnosis
cannot be treated as seven demonstrated defects relative to actual vLLM mHC.
This does not justify changing the frozen reference silently or adopting the
ordinary candidate: that candidate still fails the existing four-layer INT8
independent reference. Neither implementation nor tolerance has been changed.
Artifacts are archived under `build_output/glm53_actual_mhc/`.

### Four-layer independent native GEMM reference (2026-10-21)

The isolated two-GEMM candidate was rerun on A3 with TP2, EP2, four layers and
two decode steps. A separate Torch NPU process evaluated the KDA input/output
BF16 linear calls (108 calls, 54 distinct weights). All other golden operations,
fixtures, comparators and thresholds were unchanged. The reference propagated
its own outputs throughout the chain; no kernel observations were injected.
The worker exited before the distributed kernel runtime started. This checks
native GEMM rounding, not a complete native vLLM or native mHC reference.

The fresh run still fails (101.04 s). Recurrent state, pool scale and hidden
pass their existing gates, but convolution state, DSA caches, layer outputs and
streams do not. The first failing layer-output gate moves from layer 1 Dense
to layer 3 MoE at step 0: 7.1777% outliers exceeds its existing 5% allowance.
Cache and convolution-state checks retain their zero-outlier requirement.

Comparing the frozen candidate capture against the two independent references:

| Final hidden metric, both ranks and steps | Original CPU reference | Native GEMM reference |
| --- | --- | --- |
| Relative L2 | 2.8876% | 1.2103% |
| Outliers at atol=rtol=0.02 | 58,306 / 262,144 | 7,916 / 262,144 |
| Maximum absolute error | 0.1796875 | 0.1953125 |

The improvement is substantial but does not establish equivalence. Passing
hidden's existing allowance does not imply zero-outlier correctness, and the
maximum absolute error does not improve.

A separate diagnostic replays all six Dense boundaries using their recorded
mixed inputs. Each output is bitwise equal to the candidate's Dense output.
Against the native GEMM independent reference, layer 0 mixed input already has
91/65 differing BF16 values at steps 0/1. These produce 8/6 different INT8 codes
and 315/216 Dense-output outliers on one rank. Later layers amplify this drift.
These boundary replays isolate propagation; they do not replace the independent
end-to-end gate or demonstrate that every quantization contract matches vLLM.

Therefore CPU versus NPU GEMM rounding explains part, but not all, of the
observed regression. The next isolation target is the first KDA/mHC boundary
before layer 0 Dense quantization, separating same-input projection accumulation
from nonlinear and mHC rounding on this generated fixture. The two-GEMM
candidate remains unadopted; production kernels, CPU goldens and tolerances
remain unchanged. Diagnostic scripts, gate reports and reference captures are
archived in `build_output/glm53_native_gemm_reference/`.

### Generated-fixture output-gate isolation (2026-10-21)

A TP2 first-layer trace now uses the exact weights, embeddings, initial states
and row mappings from the four-layer generated fixture. Both its kernel outputs
and independent native-GEMM reference match the corresponding four-layer
attention, normalized-input and FFN-mixed captures bitwise at both steps.
The trace passes its original 0.02 absolute/relative zero-outlier checks in
37.00 s; this does not change the failing four-layer gate.

Within this trace, mixed input, RMSNorm output, QKV projection, beta, convolution
Q/K/V and recurrent-core BF16 output are bitwise equal to the independent
reference for both ranks and steps. Decay has a few FP32 last-bit differences
(maximum 4.77e-7), but they do not change the captured BF16 core outputs.
The output-gate branch has 27/40 differing BF16 outputs across ranks at step 0
and 219/217 at step 1. On identical step-0 inputs, KDA output projection and both
following mHC boundaries are bitwise equal to their references. The first-step
attention drift is therefore inherited from the output-gate branch in this case.

An isolated two-GEMM probe reproduces the output-gate trace bitwise. Its first
4096-to-128 GEMM differs from native Torch at only one value per step; the second
128-to-4096 GEMM propagates that difference. First-stage compensation is not a
consistent remedy: at step 0 it increases the two ranks' gate differences to
660/704; at step 1 it eliminates them.

| Step | First-GEMM coordinate | Native BF16 | PyPTO ordinary BF16 |
| --- | --- | --- | --- |
| 0 | (10, 106) | 0.0211181640625 | 0.021240234375 |
| 1 | (3, 45) | 0.134765625 | 0.1337890625 |

Native GEMM itself changes rounding with the output width. Zero-padding the
same 128-row weight to 256, 512 or 4096 rows makes its first 128 outputs match
PyPTO ordinary accumulation exactly at both steps. Leaving the width at 128
reproduces the two discrepancies above. This demonstrates shape-dependent
backend rounding; it does not establish the backend's internal tiling algorithm.
Sequential FP32 sums of 2, 4, 8, 16 or 32 K partitions do not consistently match
the native 128-column result, so no split-K workaround is adopted.

Actual pinned vLLM-Ascend HcPre also matches the PyPTO mixed outputs on identical
attention-side inputs at both steps and FFN-side inputs at step 0. At step 1,
the FFN-side same-input comparison has two differing BF16 values. These findings
do not establish that all arithmetic compensation is necessary or removable;
they identify the first-step gate GEMM as the current causal target.

A four-layer causal experiment changes only the independent reference's
4096-to-128 native GEMMs: pad their weights to 256 output rows and retain the
first 128 results. The candidate kernel is unchanged. With this diagnostic
reference, all four recorded phases of layers 0, 1 and 2 are bitwise equal at
both steps. Layer 3 normalized input is also bitwise equal; remaining drift
first appears in its DSA path and propagates into MoE.

The fresh TP2 run against this diagnostic reference finishes in 101.41 s.
All existing gates pass except `pool_cache`, which reports one failing INT8 value
on rank 0 among the 3072 checked cache elements (actual 5, expected 4). Final
hidden has zero 0.02/0.02 outliers and 0.4533% relative L2 in the frozen-capture
comparison. This is a causal isolation result, not acceptance against native
128-column GEMM: the original native and CPU references still fail, remain
unchanged, and are not replaced by the padded diagnostic. No candidate is
adopted. Remaining targets are native narrow-GEMM accumulation semantics and
the independently isolated DSA pooled-cache quantization boundary.

All probes remain temporary, with artifacts in
`build_output/glm53_synthetic_gate_trace/`. Production model source, tolerances
and the original CPU/native references remain unchanged.

### Removing arithmetic compensation: corrected reference and candidate (2026-10-21)

The removal target is now an implementation with no dependency on
`arithmetic.py`. An isolated candidate replaces the seven callers' compensated
sum, divide, exp, sigmoid, SiLU and inverse-norm operations with ordinary FP32
operations. Its KDA/Dense BF16 GEMMs use ordinary FP32 accumulation; mHC uses
an ordinary FP32 split-K projection. The helper file is physically absent in
the candidate. The working model remains unchanged pending full validation.

**Reference correction.** The pinned vLLM-Ascend implementation does not execute
the six first-stage KDA projections separately. In
`vllm_ascend/models/glm5next/kda.py`, construction at line 175 and forward at
line 295 merge Q/K/V/b/f_a/g_a and split the result afterward. The earlier
four-layer native-GEMM diagnostic used separate calls, unlike both that source
and the frozen official-weight native hybrid. Its narrow 128-column rounding
mismatch is therefore not an implementation target. A new independent reference
uses the actual merged projection layout, preserving the older diagnostic
artifacts rather than silently overwriting them. The two-GEMM candidate against
this corrected layout fails only the previously isolated pooled-cache check.

**Pool-cache isolation.** Recomputing both steps' pooled tables from their
recorded raw-cache inputs is bitwise equal on both ranks. In step 1, pool event
7 uses raw rows 1028--1031. Across those rows, the independent reference differs
at one BF16 key value (maximum 0.001953125) and 232 FP32 gate values (maximum
8.11e-6). One pooled BF16 dimension, 61, changes. After Hadamard rotation,
column 122's quantization input is 4.506616 versus 4.493121, producing 5 versus
4. Column 107 also rounds differently (-66 versus -67), but passes the existing
relative tolerance. No pool-writer compensation is indicated by this replay.
This does not establish equivalence with vLLM's BF16 pooled-cache contract.

**Official-weight check without compensation.** Ordinary mHC split-K removes
the two native-hybrid outliers of the initial no-compensation candidate. The
updated candidate passes the original first-layer, two-step TP2 CPU gate and
has zero outliers at both steps against the frozen actual-mHC/native-KDA hybrid
(atol=rtol=0.02, zero allowance). Relative L2 is 0.08676%/0.08603%. This hybrid
still uses Torch recurrent math and reconstructed TP reduction, not full vLLM
FLA/HCCL execution.

**Generated-fixture RMSNorm isolation.** On the same-input first-layer trace,
native HcPre mixed inputs match bitwise, but ordinary RMSNorm initially differs
at (token 1, column 1484) and (token 1, column 2812) in step 0. Ordinary 256,
512 and 1024-column chunk reductions and sqrt/reciprocal variants retain those
two differences. A trace of the FP32 argument and inverse RMS identifies the
reduction boundary: for the affected row the PyPTO argument is 3.95869875,
while a separate native Torch mean operation gives 3.95869923. The native
RMSNorm returned inverse is 0.502601445 versus PyPTO's 0.502601504; the separate
Torch mean is a diagnostic, not a claim to expose CANN's internal reduction.

Accumulating square vectors across chunks before a single row reduction,
instead of summing already reduced chunks, removes both BF16 differences for
512- and 1024-column chunks. This is ordinary FP32 arithmetic with a different
reduction order, without high/low components or correction terms. The 512-column
candidate also preserves the official-weight zero-outlier hybrid result and
passes its original CPU gate in 26.23 s.

The broader four-layer no-compensation run with native mHC/RMSNorm but CPU KDA
math still fails. Further validation uses native FP32 projection gates,
convolution, recurrence, gated normalization and deferred RMSNorm quantization
formulas as well. DSA, MoE and integer GEMM reference portions remain CPU, so
this is explicitly a hybrid reference. No acceptance threshold has changed.

The expanded native-FP32 hybrid run finishes with an independent end-to-end
FAIL in 102.55 s. First-layer normalized inputs now match bitwise at both steps.
Attention outputs differ at 180/76 BF16 values (none exceed 0.02/0.02), leaving
36/12 differing mixed inputs before Dense. Dense then has 205/82 outliers on
one rank, and later layers amplify the differences. Thus the ordinary RMSNorm
change removes the identified initial mismatch, but does not complete the
end-to-end precision work.

The same compensation-free vector-reduction candidate passes all existing
four-layer, two-step TP2 boundary-mode checks in 90.74 s: latent/raw/pooled
caches, scales, convolution/recurrent state, layer outputs, streams and hidden.
This uses the pre-existing explicit `--validate-boundaries` mode with recorded
sublayer inputs; its pooled-cache comparator already has a local INT8 rounding
allowance. Neither that allowance nor any independent gate was changed. The
boundary run separately prints the independent hidden gate as FAIL. Local
correctness and accumulated independent error are therefore separate results.

Artifacts and all three no-compensation candidate snapshots are archived in
`build_output/glm53_noarith_removal/`. Removal from the working model is pending
resolution of the remaining independent KDA/normalization rounding propagation;
the working `arithmetic.py` is still used. No compensation has been reintroduced
into the candidate to make the failing independent test pass.


### Ordinary convolution SiLU and the remaining boundaries (2026-10-21)

First-layer tracing now uses the expanded native hybrid reference throughout.
RMSNorm, merged QKV projection and gates match bitwise. Same-input replay
isolates the first discrepancy to convolution SiLU; changing the tap addition
order does not affect this fixture. Native Torch NPU SiLU matches
`x / (1 + exp(-x))`, while `x * reciprocal(1 + exp(-x))` differs by an FP32 bit
at two BF16 rounding boundaries. For example, at argument -0.07799091935 the
results are -0.03747558594 and -0.03747558221, respectively. Their BF16 outputs
therefore differ despite the small FP32 discrepancy.

The isolated `noarith_convdiv` candidate changes only that ordinary expression
relative to `noarith_vector`; it adds no compensation. Its first-layer Q/K/V,
recurrent output, local output projection and TP2 attention match the hybrid
reference bitwise on both steps and ranks. The stage captures also match the
corresponding full four-layer run bitwise. This establishes agreement with the
Torch NPU convolution reference, not execution of the external FLA AscendC
convolution used by pinned vLLM-Ascend.

Validation results:

- Official-weight first-layer TP2 CPU gate: PASS, 28.08 s. Frozen native hybrid
  hidden comparison: zero outliers on both steps at unchanged 0.02/0.02.
- Four-layer TP2 independent hybrid gate: FAIL, 95.23 s. Step-0 hidden outliers
  decrease from 16,496 to 13,620 out of 131,072; the allowed fraction remains 5%.
  The first failing layer-output gate remains layer 1 Dense, with 15,204
  outliers out of 131,072. Cache gates also remain unresolved.
- The post/residual coefficient trace passes in 25.14 s. Identical-coefficient
  replay through actual native HcPost matches bitwise on both steps/ranks.
  Independent streams still differ at two BF16 values per step/rank; these
  differences originate in mHC coefficient generation, whose maximum observed
  coefficient discrepancy is 1.79e-7.
- First-layer Dense still has 159/82 outliers on rank 0 at steps 0/1. At step 1
  its mixed input matches bitwise, so upstream attention or mHC propagation
  alone cannot explain the remaining Dense/quantization discrepancy.

Next isolation targets are mHC coefficient generation and Dense quantization
on identical inputs. Production model sources remain unchanged and still use
`arithmetic.py`. Candidate source, replay scripts, task IDs and metrics are
archived under `build_output/glm53_noarith_removal/`. An initial full-run retry
selected the previous candidate accidentally and is explicitly excluded from
these results; the reported full run verifies the corrected variant and trace
integrity. No precision threshold has been relaxed.

## Ordinary arithmetic closure attempt (2026-10-21)

At this stage of the experiment, production compensation was still active;
no kernel changes from this experiment had been integrated.

Dense isolation identified reciprocal multiplication in the diagnostic hybrid
quantizer, differing from the explicit division in the unchanged CPU golden.
Correcting that diagnostic expression makes INT8 codes and integer GEMM
accumulators exact; local Dense output has zero outliers (maximum 2.39e-6).

Source-aligned ordinary mHC partitioning and reduction orders improve native
agreement. With identical native projection inputs, coefficient postprocessing
is bitwise exact. Actual projection still differs at FP32 precision; native
HcPre explicitly enables HF32, unlike the current PyPTO projection path.
The intermediate candidate passes owning mHC cases, official-weight first-layer
TP2 and explicit boundary replay. The final independent four-layer candidate
still fails: step-1 hidden has 7,716/131,072 outliers (5.8868%, allowed 5%),
and several caches fail their zero-outlier checks.

Direct reference comparison also proves a blocking acceptance conflict:
updated latent element `[0, 0, 6, 15]` has CPU expected value -0.71875 and
native-hybrid expected value -0.60546875. At the unchanged tolerances their
acceptance intervals do not overlap. The hybrid includes CPU components and
is not a complete vLLM run. Resolving the authoritative operator/reference
contract is required before production removal; thresholds were not relaxed.

Scripts, candidate snapshots, logs and integrity hashes are archived locally
under `build_output/glm53_precision_closure/`. All 37 production source hashes
match the preserved baseline. A separate two-accumulator compiler reproducer
is recorded in the local issue record; separate SPMD scopes avoid that issue
in these candidates.

## What is quantized

Activations are quantized **dynamically per token** the way the a2a3 W8A8 sibling
does it: each row's amax, floored by `INT8_AMAX_EPS = 1e-4`, is rescaled to
`INT8_SCALE_MAX = 127`, rounded through int32 and fp16 to match device rounding.
The router kernel produces that INT8 view once and both the shared expert and the
EP dispatch payload reuse it.

`quant_model_description.json` and the tensor dtypes determine how a checkpoint
is loaded. The AscendV1 W8A8 checkpoints used for integration have the following
formats; a floating-point MLA projection is also accepted when its metadata and
stored dtype agree.

| Tensor | Format |
| --- | --- |
| Routed/shared/dense FFN `gate_proj`, `up_proj`, `down_proj` | INT8 with per-output-channel scales; staged as INT8 plus flattened FP32 scales |
| MLA `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `o_proj` | W8A8 INT8 in these checkpoints; explicitly dequantized to BF16 for the current attention kernels |
| MLA `kv_b_proj`, KDA/indexer projections, norms, router, embedding and LM head | BF16 |
| mHC function, scale and base; router correction bias | Source BF16; the function stays BF16, while scale, base and correction bias are staged as FP32 to match the kernels |
| MLA 512-latent cache | BF16 |
| Indexer state cache | FP32 today, 256 wide per token; the sibling port quantizes its indexer cache to INT8 with a per-row FP32 scale, which would save about 2.3 KB per token here |
| Hyper-connection stream | BF16 — see above, this is where GLM departs from DeepSeek-V4's FP32 stream |
| Logits | FP32 |

The loader requires `W8A8_DYNAMIC` with `group_size: 0`. For every requested
quantized weight it checks INT8 storage, positive finite scales, and zero offsets.
It rejects incompatible formats rather than casting FP8 or BF16 values to INT8.
MLA's BF16 adaptation is `BF16(FP32(weight) * scale)`; it does not reproduce a
native W8A8 attention path with runtime activation quantization. `mla_mode="native"`
keeps the INT8/scale pairs for consumers that implement that path, and does not
produce a BF16 absorbed output weight.

### Rotation assets

The quantization description can include `is_rot_used` and an optional QuaRot
rotation map. The loader preserves that metadata and exposes the indexed
`rot.weight` through `Checkpoint.rotation()` on request. Optional files need not
be loaded to stage the 45-layer text backbone. Converted weights are used as
stored; no additional global rotation is applied to the backbone activations.
MTP and vision weights are outside this loader's full-model assembly.

The indexer's Hadamard-128 is a separate transform. The loader supplies the same
matrix as the indexer golden; it is not the checkpoint's hidden-size rotation.

## Loading weights

`models/glm5_3_flash/weights.py` provides an indexed CPU reader. It opens only
requested tensors and TP slices instead of materializing all checkpoint shards.
The checkpoint configuration must match the compiled GLM dimensions and layer
schedule. Quantization metadata, shapes, dtypes and topology are checked before
the returned weights reach a kernel.

```python
from models.glm5_3_flash.weights import Checkpoint

checkpoint = Checkpoint("/path/to/vllm-ascend-a3-w8a8")
layer = checkpoint.layer(3, tp_rank=0, tp=16, ep_rank=0, ep=16)
vocabulary = checkpoint.vocabulary(tp_rank=0, tp=16)

# Load and consume one layer at a time when full residency is unnecessary.
for layer_id, weights in checkpoint.iter_layers(tp_rank=0, tp=16, ep_rank=0, ep=16):
    ...

# Or retain all 45 layers and the layer-external weights for this rank.
model = checkpoint.model(tp_rank=0, tp=16, ep_rank=0, ep=16)
```

`layer` groups `input_norm`, `attn_hc`, `ffn_hc`, `attention`, `ffn`, and an
`indexer` for sparse-attention layers. `model["layers"]` is keyed by layer ID;
its other entries are `embedding`, `lm_head`, `final_norm`, `vocab_start` and
`vocab_stop`. Embedding and LM-head rows are TP-sharded; the final norm is
replicated. Dense FFNs are TP-sharded along their intermediate dimension.
Routed experts are EP-sharded by expert ID, and shared experts remain complete
on every EP rank. Attention follows the existing head partition; the indexer
remains replicated. Prefill and decode can reuse the same staged weights.

The existing FFN validation entries keep generated weights as their default.
Add `--checkpoint` to load real weights into the same kernels, goldens and
precision checks:

```bash
# Run these inside a two-device allocation; TASK_DEVICE comes from the allocator.
python models/glm5_3_flash/decode_layer.py -p a2a3 -d "$TASK_DEVICE" --ep 2 \
  --checkpoint /path/to/vllm-ascend-a3-w8a8 --layer 3 --num-tokens 3
python models/glm5_3_flash/prefill_layer.py -p a2a3 -d "$TASK_DEVICE" --ep 2 \
  --checkpoint /path/to/vllm-ascend-a3-w8a8 --layer 3 --tokens 64 --num-tokens 3
```

These entries execute the MoE FFN half of a layer, so `--layer` must be 3–44.
By default, checkpoint mode mixes deterministic synthetic activations with the
selected layer's real FFN mHC weights. To supply activations from another stage,
use `--hidden hidden.pt`, containing a CPU BF16 tensor `[EP, active_tokens, 4096]`
after mHC pre-mixing and before RMSNorm. The active token count is inferred from
that tensor unless supplied explicitly; it must fit the entry's capacity.
This component fixture is not a complete model forward or a text-generation test.

## Prior art, and what it is actually worth

None of the references below is drop-in code for this repo, but two of them are
worked designs for the hardest kernels on our exact platform.

| Reference | Where | What it gives us |
| --- | --- | --- |
| Chunked KDA in pypto, tuned on Ascend910B3 | cann-recipes-infer, `integration/vllm/ling-3.0-flash/npu_patch/.../ops/pypto/kda/chunk_kda_impl.py` | A complete chunked delta-rule forward at `T=4K, H=4, K=V=128` — our per-rank TP16 shape — with a design write-up covering the M1-M4 stages, the 8x8-leaf block inverse, and the two bottlenecks whose fix was worth 3.8x |
| Recurrent KDA in pypto | its `fused_recurrent_kda_impl.py` sibling | The decode-step counterpart |
| Lightning indexer, MLA prolog, hc_pre, sparse attention in pypto | cann-recipes-infer, `ops/pypto_python/impl/` | Algorithm and tiling references for the indexer and MLA items |
| vLLM Ascend `glm5next` | `vllm_ascend/models/glm5next/` | The only NPU port of this architecture: module decomposition, cache specs, W8A8 wiring, MTP. At the inspected revision it provides BF16 KPool cache orchestration with CANN and Triton scoring paths; see the alignment audit above |
| CANN recipes GLM-5 / GLM-5.2 | cann-recipes-infer, `models/glm_5{,_2}/` | Ascend-native MLA + DSA + MoE with a 16-rank W8A8 A3 config. No KDA and no mHC |
| GLM-4.5 pypto operators | the open-source pypto mirror, `models/glm_v4_5/` | MoE gate, expert selection and the shared-expert W8A8 patterns |

The two pypto references above target a **different pypto frontend**
(`pypto.frontend.jit` / `pypto.Tensor` / `pypto_impl` / auto-tiling). The pypto
this repo pins exports only `jit`, `language`, `ir`, `runtime` and friends, so
those files are design references, not code to copy: the port is from implicit
tiling to the explicit `pl` tile DSL.

## Platform envelope

| a2a3 (`Ascend910_9392`) | Value | a5 for contrast |
| --- | ---: | ---: |
| AIC / AIV per die | 24 / 48 | 36 / 72 |
| `PLATFORM_MAX_BLOCKDIM` / max cores | 24 / 72 | 36 / 108 |
| UB (safe) | 192 KB (~184 KB) | 248 KB |
| L1 / L0A / L0B / L0C | 512 / 64 / 64 / 128 KB | — |
| HBM per die | 64 GiB (~61.3 GiB visible) | — |
| fp8 / MX `mmad` | **none** | supported |
| 4-bit in-core dtypes | **not supported** | FP4 supported |
| `Acc -> Vec` and `Vec -> Mat` memory edges | **absent** | present |
| AIC/AIV cross-core pipe | through GM | on-chip |
| GM access granularity | 512 B | 128 B |
| bf16 atomic add into GM | supported | not supported |

Some of these change how a kernel is written rather than how fast it runs. The
missing `Acc -> Vec` edge forces a GM round trip where a5 would fuse. INT8 can only
reach GM through a Vec tile or the quantized fix-pipe path. And not in the table but
worth knowing before the first MIX kernel: a2a3 requires dual-AIV dispatch even for
a no-split MIX kernel, or the AIC cross-core handshake deadlocks.

### What stream B's `a2a3sim` bring-up surfaced

Each of these was a compile or simulator failure, not a guess:

- **The simulator needs a real GCC 15.** A `g++-15` shim pointing at GCC 13 builds
  simulator kernels that get every BF16 matmul wrong while FP32 still passes, so the
  failure looks like a kernel bug. `deepseek_v4_flash_mtp/decode_compressor_ratio4.py`
  is a quick canary: it passes in CI and fails under such a shim.
- **Do not mix the Tensor and Tile levels.** `pl.slice`, bracket slices, `pl.full`,
  `pl.matmul_acc` and slice assignment are Tensor level; `pl.load`, `pl.tile.full`,
  `pl.create_tile` and `pl.store` are Tile level. A Tensor-level result is written with
  `dest = pl.assemble(dest, pl.set_validshape(src, rows, cols), offset)`, a Tile-level
  one with `pl.store`; `pl.write` works on a Tensor-level on-core buffer where
  `pl.tile.write` does not.
- **`tmp_tile` is Tile-only.** `pl.row_sum` / `pl.row_max` on a Tile must take one;
  on a Tensor they must not. High-precision `rsqrt` on a Tile is
  `pl.tile.rsqrt(x, tmp)`, not `high_precision=True`.
- **FP32 reductions need a 32-byte column**, and a row-major tile needs a 32-byte row,
  so an `[H, 1]` column wants `H % 8 == 0` and is best built as `[1, H]` then
  reshaped.
- **A matmul is only split into L0 tiles when M reaches 16.** Below that the
  128 x 512 KV operand of the sparse attention stays whole and overflows the 64 KB
  right buffer; the TP16 attention pads its 4 heads to 16 zero-filled rows for this.
- **No mutable scalar flags across branches.** A Python int set in one `if` and read in
  another becomes a phi the backend cannot materialise; write each branch's work in
  place, as the donors do. Loop-carried tile state goes through `pl.yield_` on every
  path.
- **In a mixed cube/vector scope, no branch may decide whether loop-carried state is
  updated.** When the carried tiles flow through an `if` / `else`, the partitioner
  materialises their pre-loop `pl.full` initialisation in the cube half as well, and
  `ccec` rejects `vector_dup` / `set_vector_mask` for `dav-c220-cube`. `a2a3sim`
  compiles both halves with `g++` and never sees it, so this reaches CI as an a2a3-only
  failure (run 35555578466). Let the branch pick the block's inputs and keep the merge
  unconditional, or split the scope by hand with `pl.split_aic` / `pl.split_aiv` as
  `deepseek_v4_flash_dspark/prefill_sparse_attn.py` does. The check needs no card:
  compile with `-p a2a3 --compile-only` and assert that the generated
  `kernels/aic/*.cpp` carries no vector op between `#if defined(__DAV_CUBE__)` and its
  `#endif` (`TMUL` there is address arithmetic and is expected).
- **`pl.spmd` takes a Var, not a call.** Bind `pl.tensor.dim(...)` to a name first.
- **A script-entry file must put the repository root first on `sys.path`**, or
  `from golden import run` resolves to this directory's own `golden.py`.

## How this directory is organised for parallel work

Every operator has one ownership file. Each file carries the exact maths in its
module docstring, a `golden_*` Torch reference, and a typed `@pl.jit.inline` ABI
whose body raises `NotImplementedError` until its owner lands it. Shapes come
from `config.py`, so two engineers working on neighbouring operators agree on the
interface without talking to each other.

A file has **no script-entry guard until its golden is real**. The a2a3 daily CI
selects a case purely by grepping for that guard, so adding one to a file whose
golden still raises would put an unimplementable operator into the sweep. Today
`config.py`, `golden.py`, `quantization.py`, `metadata.py`, `mhc.py`, `rmsnorm.py`
and all five stream-B MLA files carry real references. RMSNorm and embedding
lookup are library operators without standalone validation entry points. The
remaining validation commands include:

```bash
source .venv/bin/activate
python models/glm5_3_flash/golden.py              # [GOLDEN] PASS norms / swiglu / moe_gate
python models/glm5_3_flash/quantization.py        # [GOLDEN] PASS quantization
python models/glm5_3_flash/mhc.py                 # [GOLDEN] PASS mhc
python models/glm5_3_flash/mla_prolog.py          # [GOLDEN] PASS mla_prolog + prolog/absorb
python models/glm5_3_flash/mla_cache.py           # [GOLDEN] PASS mla_cache + device scatter
python models/glm5_3_flash/mla_epilog.py          # [GOLDEN] PASS mla_epilog + both paths
python models/glm5_3_flash/prefill_sparse_attn.py # [GOLDEN] PASS + device sparse prefill
python models/glm5_3_flash/decode_sparse_attn.py  # [GOLDEN] PASS + device sparse decode
```

`models/deepseek_v4_1_flash/mhc.py` is the worked example of a finished item: a
golden, the four kernel bodies, a `@pl.jit` test entry, `build_mhc_tensor_specs`
and a precision comparison, in 559 lines. Read it for the shape, not for the
kernels — its directory targets A5 in `.github/ci_platforms.json`, and
`# ci: no-sim` excludes device-only entries from simulator runs.

For kernels you actually intend to run here, prefer the a2a3 files:
`models/deepseek_v4_flash_mtp/` has implemented `hc_pre.py`, `hc_post.py` and
`hc_head.py`, `gate.py`, `expert_routed.py`, `expert_shared.py`, `decode_indexer.py`,
`prefill_indexer.py` and `decode_sparse_attn_csa.py`, all of which the a2a3 daily
sweep runs today.
## Five streams, five owners

The work divides into five self-contained streams. Each owns a disjoint set of
files, so two owners never edit the same file, and they meet only at the ABIs that
`config.py` and the `@pl.jit.inline` signatures already fix. Stream E owns those
contracts and publishes them first.


| Stream | Owns | Files | Items | Days | Long pole |
| --- | --- | ---: | ---: | ---: | --- |
| **A** | KDA linear attention — 34 of the 45 backbone layers | 5 | 10 | 50-82 | `kda_chunk_delta_prefill` — the chunked delta rule, 20-32 days on its own |
| **B** | NoPE MLA sparse attention — 11 sparse layers + the MTP layer | 5 | 6 | 45-72 | `mla_sparse_attn_prefill` — gather-and-attend over 2051 selected rows, 15-25 days |
| **C** | kpool DSA indexer and mHC — 12 indexer instances; mHC on all 45 layers | 4 | 11 | 38-60 | `indexer_cache` — it blocks five downstream items, so land it in week one |
| **D** | MoE, primitives, output head, MTP — 43 sparse layers, 3 dense layers, the draft step | 12 | 14 | 61-100 | `moe_dispatch_ep16` — the 16-rank all-to-all, 14-22 days |
| **E** | Runtime, weight loading, composition — the whole model | 9 | 6 | 76-120 | `layer_composition` — 40-60 days, and by definition it lands last |
| *(staged)* | Vision tower — prefill of multimodal requests only | 5 | 6 | 23-35 | `vision_attention` — non-causal varlen attention, no donor in this repo |

Phase one is streams A-E: **269-434 engineer-days** across five owners, before any tuning. The vision tower is staged out — the A3 recipe serves this checkpoint text-only unless `--limit-mm-per-prompt` is set, and the tower is 0.53 G parameters against the text model's 320 G.

### Stream A — KDA linear attention

*34 of the 45 backbone layers · 5 files · 10 work items · 50-82 engineer-days*

**Files.** `kda_projection.py`, `kda_conv.py`, `prefill_kda.py`, `decode_kda.py`, `kda_output.py`

**Where to start.** No donor in pypto-lib. The `pl`-adjacent reference is the tuned Ascend910B3 `chunk_kda_impl.py` / `fused_recurrent_kda_impl.py` in cann-recipes, plus its design doc.

| Work item | Verdict | Size | Days | Depends on |
| --- | --- | --- | ---: | --- |
| `kda_chunk_delta_prefill` | SPLIT | XL | 20-32 | `kda_qkv_proj` |
| `kda_recurrent_decode` | ADAPT | L | 8-13 | — |
| `kda_conv1d_prefill` | NEW | M | 4-7 | `kda_qkv_proj`, `kda_conv1d_decode` |
| `kda_out_proj` | ADAPT | M | 4-7 | `mhc_post` |
| `kda_conv1d_decode` | NEW | M | 3-5 | `kda_state_cache`, `kda_qkv_proj` |
| `kda_state_cache` | NEW | M | 3-5 | — |
| `kda_gates` | NEW | S | 2-4 | `kda_qkv_proj` |
| `kda_qkv_proj` | ADAPT | S | 2-3 | — |
| `kda_l2norm_qk` | ADAPT | S | 1.5-3 | — |
| `kda_out_gated_norm` | ADAPT | S | 2-3 | — |

### Stream B — NoPE MLA sparse attention

*11 sparse layers + the MTP layer · 5 files · 6 work items · 45-72 engineer-days*

**Files.** `mla_prolog.py`, `mla_cache.py`, `prefill_sparse_attn.py`, `decode_sparse_attn.py`, `mla_epilog.py`

**Where to start.** `deepseek_v4_flash_mtp/qkv_proj_rope.py`, `prefill_sparse_attn.py` and `decode_sparse_attn_csa.py`. Same 512-wide KV row; delete the sliding window, the ratio-4 slot rewrite, the attention sink and all of the RoPE.

| Work item | Verdict | Size | Days | Depends on |
| --- | --- | --- | ---: | --- |
| `mla_sparse_attn_prefill` | ADAPT | XL | 15-25 | — |
| `mla_sparse_attn_decode` | ADAPT | L | 9-14 | — |
| `mla_epilog_o_proj` | SPLIT | L | 8-13 | — |
| `mla_prolog_nope` | ADAPT | M | 6-9 | — |
| `mla_kv_absorb` | ADAPT | M | 4-6 | — |
| `mla_latent_cache` | ADAPT | S | 3-5 | — |

### Stream C — kpool DSA indexer and mHC

*12 indexer instances; mHC on all 45 layers · 4 files · 11 work items · 38-60 engineer-days*

**Files.** `prefill_indexer.py`, `decode_indexer.py`, `indexer_cache.py`, `mhc.py`

**Where to start.** `deepseek_v4_flash_mtp/{prefill,decode}_indexer.py` for the selection pipeline (its `_cp_topk512_query` is the same K over the same candidate cap), and `hc_{pre,post,head}.py` for mHC. The k-pooling stage has no donor anywhere.

| Work item | Verdict | Size | Days | Depends on |
| --- | --- | --- | ---: | --- |
| `indexer_cache` | SPLIT | L | 7-10 | — |
| `indexer_score` | ADAPT | L | 6-9 | `indexer_cache` |
| `indexer_kpool_compress` | ADAPT | M | 5-8 | `indexer_cache` |
| `indexer_topk512` | ADAPT | M | 4-6 | `indexer_score`, `indexer_cache` |
| `indexer_qk_proj` | NEW | M | 3-5 | — |
| `indexer_share_mtp` | NEW | M | 3-5 | `indexer_score`, `indexer_cache` |
| `mhc_pre` | ADAPT | M | 3-5 | `mhc_post` |
| `mhc_mixes` | ADAPT | S | 2-4 | — |
| `indexer_expand_tail` | NEW | S | 2-3 | — |
| `mhc_post` | ADAPT | S | 1.5-3 | `mhc_pre`, `mhc_mixes` |
| `mhc_head` | ADAPT | S | 1-2 | `mhc_post` |

### Stream D — MoE, primitives, output head, MTP

*43 sparse layers, 3 dense layers, the draft step · 12 files · 14 work items · 61-100 engineer-days*

**Files.** `gate.py`, `expert_routed.py`, `expert_shared.py`, `dense_mlp.py`, `(new) prefill_moe.py`, `(new) decode_moe.py`, `rmsnorm.py`, `quantization.py`, `lookup_embedding.py`, `lm_head.py`, `mtp_projection.py`, `(new) decode_mtp.py`

**Where to start.** Almost every file has a same-named donor in `deepseek_v4_flash_mtp` that the a2a3 sweep already runs. This is the most port-like stream in the model.

| Work item | Verdict | Size | Days | Depends on |
| --- | --- | --- | ---: | --- |
| `moe_dispatch_ep16` | SPLIT | XL | 14-22 | `moe_gate` |
| `mtp_draft_loop` | SPLIT | L | 12-18 | `indexer_score`, `indexer_cache` |
| `moe_expert_routed_w8a8` | ADAPT | M | 5-8 | `moe_dispatch_ep16`, `moe_gate`, `moe_combine_ep16` |
| `moe_combine_ep16` | ADAPT | L | 5-8 | `moe_dispatch_ep16`, `moe_gate`, `moe_expert_routed_w8a8` |
| `lm_head_tp16` | ADAPT | M | 4-7 | `mhc_head` |
| `dense_mlp_w8a8` | ADAPT | M | 4-6 | `moe_shared_expert_w8a8` |
| `moe_gate` | ADAPT | M | 3-5 | — |
| `moe_shared_expert_w8a8` | ADAPT | S | 3-5 | `moe_gate`, `w8a8_weight_loader` |
| `embed_tokens` | ADAPT | M | 3-5 | — |
| `swiglu_clamp_quant` | ADAPT | S | 2-4 | — |
| `mtp_eh_proj` | ADAPT | S | 2-4 | `rms_norm` |
| `w8a8_linear` | SPLIT | S | 1.5-3 | — |
| `rms_norm` | ADAPT | S | 1.5-2.5 | — |
| `quant_per_token_int8` | ADAPT | S | 1-2 | `moe_gate` |

### Stream E — Runtime, weight loading, composition

*the whole model · 9 files · 6 work items · 76-120 engineer-days*

**Files.** `config.py`, `metadata.py`, `(new) weights.py`, `(new) cache_manager.py`, `attention_tp.py`, `(new) prefill_layer.py`, `(new) decode_layer.py`, `(new) prefill_fwd.py`, `(new) decode_fwd.py`

**Where to start.** `deepseek_v4_flash_mtp/{prefill,decode}_{layer,fwd}.py`, about 8,600 lines of donor. The 3:1 KDA/DSA schedule replaces its SWA/CSA/HCA schedule.

| Work item | Verdict | Size | Days | Depends on |
| --- | --- | --- | ---: | --- |
| `layer_composition` | SPLIT | XL | 40-60 | `mhc_head`, `mhc_post`, `mhc_mixes` |
| `tp16_collectives` | SPLIT | L | 15-27 | `forward_metadata` |
| `w8a8_weight_loader` | SPLIT | L | 8-12 | `indexer_score` |
| `hybrid_kv_cache` | ADAPT | L | 6-9 | `indexer_cache` |
| `forward_metadata` | ADAPT | M | 5-8 | — |
| `graph_capture` | ADAPT | S | 2-4 | — |

### Stream V — Vision tower (staged)

*prefill of multimodal requests only · 5 files · 6 work items · 23-35 engineer-days*

**Files.** `vision/patch_embed.py`, `vision/attention.py`, `vision/mlp.py`, `vision/merger.py`, `vision/fusion.py`

**Where to start.** No donor in this repo.

| Work item | Verdict | Size | Days | Depends on |
| --- | --- | --- | ---: | --- |
| `vision_attention` | NEW | L | 8-12 | `vision_patch_embed` |
| `vision_mlp` | SPLIT | M | 5-7 | `vision_patch_embed`, `vision_attention` |
| `vision_rope_qkv_norm` | ADAPT | M | 4-6 | `vision_patch_embed`, `vision_attention` |
| `vision_merger` | NEW | M | 3-5 | `vision_patch_embed`, `vision_attention`, `vision_mlp` |
| `mm_fusion` | NEW | S | 2-3 | `vision_merger` |
| `vision_patch_embed` | ADAPT | S | 1-2 | — |

## Sequencing

Three items block the most downstream work and should go out in week one:
`indexer_cache` (blocks five), `moe_gate` (blocks five) and `kda_qkv_proj` (blocks
four). Stream E's contract work — `forward_metadata`, `w8a8_weight_loader`,
`tp16_collectives`, `hybrid_kv_cache` — is the other week-one batch, because
streams A through D cannot run anything on device without it.

That gives stream E a bookend shape: about 36-60 days of shared contracts up front,
then `layer_composition` at the end. It is the integration role, and it is also the
natural reviewer for the other four.

## Where each file's donor lives

`models/deepseek_v4_flash_mtp/` is the a2a3 W8A8 sibling, and almost every GLM file
has a same-shaped counterpart there that the a2a3 daily sweep already runs. The
port is, in one sentence: take that model, delete the compressor / sliding-window /
RoPE machinery, and add the KDA family and the kpool indexer.

| GLM file | a2a3 donor | What changes |
| --- | --- | --- |
| `mhc.py` | `hc_pre.py`, `hc_post.py`, `hc_head.py` | FP32 stream becomes BF16; the learned head becomes an unweighted mean |
| `rmsnorm.py` | `rmsnorm.py` | Add the INT8-emitting variant for the three dense layers |
| `gate.py` | `gate.py` | `sqrt(softplus(.))` becomes `sigmoid`; `SCORE_PAD` 256 to 512; drop hash routing and the group mask |
| `expert_routed.py`, `expert_shared.py` | same names | 288/8 instead of 256/6, `moe_intermediate_size` 2048 |
| `lookup_embedding.py`, `lm_head.py` | same names | Vocabulary 154,880; the embedding also seeds the four HC streams |
| `mtp_projection.py` | `mtp_projection.py` | Same `enorm` / `hnorm` / `eh_proj` shape |
| `mla_prolog.py` | `qkv_proj_rope.py` | Delete the whole RoPE half; add the `kv_b_proj` absorption |
| `prefill_sparse_attn.py`, `decode_sparse_attn.py` | `prefill_sparse_attn.py`, `decode_sparse_attn_csa.py` | Delete the sliding-window gather, the ratio-4 slot rewrite, the attention sink and the inverse RoPE |
| `prefill_indexer.py`, `decode_indexer.py` | same names | Delete the INT8/Hadamard half and the compressor; add the k-pooling stage |
| `(new) prefill_moe.py`, `decode_moe.py` | same names | EP16 instead of EP8; 18 local experts |
| `(new) prefill_fwd.py`, `decode_fwd.py` | same names | The 3:1 KDA/DSA schedule replaces the SWA/CSA/HCA schedule |
| `kda_*.py`, the k-pooling stage | **no donor in this repo** | See the prior-art table above |

## What already exists in the directory

| File | State |
| --- | --- |
| `config.py` | Complete. Layer schedule, derived per-rank shapes, TP/EP validation, W8A8 metadata |
| `golden.py` | Complete for the shared transforms: RMSNorm, gated RMSNorm, L2 norm, clamped SwiGLU, the expert body, the sigmoid `noaux_tc` router, and the four mHC references |
| `quantization.py` | Complete. Per-channel and per-token INT8 with the repo's exact rounding, the dynamic W8A8 linear, and the fused dequant-SwiGLU-requant epilogue |
| `weights.py` | Indexed W8A8 CPU loading for all 45 backbone layers, TP vocabulary shards and final norm. Includes dense TP, complete EP experts, replicated shared experts, KDA, MLA BF16 adaptation, indexer and mHC |
| `decode_layer.py`, `prefill_layer.py`, `utils.py` | Composed MoE FFN validation with generated weights by default and optional checkpoint weights/hidden inputs; full attention-plus-FFN forward composition remains separate |
| `metadata.py` | Complete. Packed-batch lowering, the three distinct slot mappings (latent, per-token indexer state, pooled state), TP token ownership, and the indexer's pool and tail derivation |
| `_golden_smoke.py` | Complete. Deterministic CPU fixtures behind every golden that exists |
| `rmsnorm.py` | Plain BF16 RMSNorm, INT8-emitting per-token RMSNorm, and fused residual-add RMSNorm. Each has a colocated Torch golden; validation is exercised through consuming operators rather than a standalone CLI |
| `mla_prolog.py` | Goldens, the prolog body (three `b_trans` matmuls and two rms-norm scopes, no split-K yet), the per-token `absorb_query` body, and a device test entry per case. `absorb_output` is weight-time and belongs to the loader. Passes `a2a3sim` and a2a3 |
| `mla_cache.py` | Golden, kernel body (`pl.spmd` scatter, one block per row), device test entry and specs. Passes `a2a3sim` and a2a3 |
| `mla_epilog.py` | Goldens, both kernel bodies (row-parallel `b_trans` projection over D, FP32 partial sum for the TP16 all-reduce) and a device test entry per path. Passes `a2a3sim` and a2a3 |
| `prefill_sparse_attn.py` | Golden, kernel body (one item per token, `pl.yield_`-carried online softmax, value expansion after the block loop) and a device test entry. **Attends in latent space**: absorbing keeps the two expansions out of the per-token block loop, at the cost of two BF16 roundings against the expanded reference. An all-padding block is filled by one wide gather and merged with `beta` = 0 rather than skipped, so no branch carries the online-softmax state. Requires the front-packed `topk_indices` ABI. Passes `a2a3sim` and a2a3 |
| `decode_sparse_attn.py` | Golden, kernel body (24 lanes over (token, sparse block), L1 gather, flash partials, alpha/beta merge, empty-block skip) and a device test entry. Requires the front-packed `topk_indices` ABI. Passes `a2a3sim` and a2a3 |
| `attention_tp.py` | ABI only, but it is the one shared consumer: `kda_output`, both `mla_epilog` entries and `dense_mlp` all emit an FP32 row-parallel partial, and this is what adds the 16 of them |
| everything else | ABI and docstring only; the golden and the kernel body are the assignment |

## Open questions an owner will hit

- **Whether the MTP layer owes `rot.weight`.** `is_rot_used: true` and the tensor's
  presence in the index argue yes; `Glm5NextMTP` dropping it argues no, or argues
  the reference port has a gap. Stream D settles it before implementing MTP.
- **How the indexer cache is laid out.** vLLM Ascend splits it into an index-K
  cache and a separate FP32 pooled-state page class with a page of four tokens;
  storing the raw row instead is simpler but larger. Stream C decides, stream E's
  cache manager implements.
- **MLA precision against native W8A8 inference.** The current loader dequantizes
  MLA weights to BF16 and absorbs the value projection into the output weight.
  Component golden agreement does not establish equivalence to a deployment
  runtime that also quantizes MLA activations.
- **The cache budget.** At about 18 KB per token per rank the hybrid cache holds
  roughly 1.5 M tokens per node once weights are placed, so the recipe's
  `--max-model-len 133120` and `--max-num-seqs 32` cannot both be satisfied at
  full length.
