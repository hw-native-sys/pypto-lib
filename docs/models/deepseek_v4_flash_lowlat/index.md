# DeepSeek V4-Flash low-latency decode

This model directory provides the batch-one decode path with eight logical
ranks. The default verification width is eight tokens. It is an experimental
decode implementation, not a `pypto-serving` deployment contract.

## Precision contract

The native HC stream uses FP32 at `hc_pre` input and `hc_post` residual/output,
including the public `pre_hc_hidden_out` output. Dense, attention and expert
local tensors also use BF16; this is a mixed-precision contract.

The independent Torch references preserve BF16 Cube K16 and FP32 Cube K4
accumulation, ascending split-K reductions and per-group projection scales. Float64
intermediates model each product group's FP32 rounding; model operands
and outputs keep their declared dtypes. This does not establish universal
bitwise equivalence or replace the existing numerical comparisons.

Standalone Q validation checks its produced head group; the other pure-Out
heads have no initialized-value guarantee.

The FP32 gate uses deferred RMS: `xg = x_mixed.float() * norm_w.float()`.
It applies `inv_rms` after projection and includes it in the dequantization
scale for INT8 `quant(xg)`, without BF16-rounding a normalized routing vector.
Gate activation quantization refines the FP32 quotient with a product-error
residual before round-to-nearest-even conversion. A power-of-two scaling keeps
the compensation intermediates bounded for large finite denominators; it does
not change the INT8 weights, FP32 dequantization scales or deferred RMS policy.
This does not assert all-domain IEEE or nonfinite-input equivalence.

Gate, QKV and SWA share scope-local FP32 compensation primitives in
`fp32_numerics.py`; these helpers create no tasks or global-memory buffers.
The gate's softplus evaluates `log(1 + exp(-abs(x)))` before adding the
nonnegative part of the input. Its logarithm correction consumes the
unrounded two-part sum. The square-root correction applies within
`[2^-40, 2^40]` and retains the original square-root result outside that
interval. Owning Torch references round double-precision unary results to
FP32 at the specified boundaries and preserve the explicit reduction order.
Finite-case validation does not establish correctly rounded results for
every FP32 input or replace complete-forward validation.

The attention and final RMSNorm references reduce FP32 squared inputs in
FP64, then round the sum once to FP32. The mean and epsilon addition remain
FP32; the inverse square root is evaluated in FP64 and rounded to FP32.
This specifies the reference's statistical precision without changing epsilon,
activation dtypes, or the numerical comparison thresholds.

RMSNorm compensates its pairwise FP32 square-sum and refines the reciprocal-
square-root seed with a product-error residual. The compensated reduction
applies when the original squared sum is at most `2^40`; the inverse correction
applies for variance at most `2^40`. Larger values retain the original path.
The reference and comparison thresholds are unchanged. Finite input tests do
not establish all-domain correctly rounded arithmetic or a performance gain.

Q/KV kernels use
the existing square-root correction and compensated reciprocal at their
separately rounded FP32 inverse boundaries.

Public stored LM-head logits are FP32. Community vLLM can select
`hf_overrides={"head_dtype": "float32"}`; this alone does not align the forward.

The observed vLLM-Ascend 0.30 reference uses BF16 HC streams across 43 layers;
its FP32 router input widens already-rounded low-precision values. Its EP8
expert quantization groups also differ from native TP8 with intermediate
shards of 256. These are reference-policy boundaries; they establish neither
a native implementation defect nor the cause of observed token differences.

A complete golden must align weights/scales, input state, HC, quantization,
routing, reduction and logit precision. Matching dtype labels or W8A8
checkpoint names, source/component CI and local type checks do not establish
complete-model numerical correctness. The existing comparison gates apply.
The current FP32 policy is retained. A community-precision compatibility mode
is not implemented and requires an explicit precision-policy decision.

## Reproducible reduction order

The output projection publishes each rank's FP32 contribution into a separate
slot. Each of the two alternating lanes contains `N_RANKS * T_PAD` rows.
After the existing all-peer completion barrier, the consumer adds ranks
`0..N_RANKS-1` to a positive-zero accumulator and rounds to BF16 once.
Every producer overwrites its valid rows before publication; allocator contents
and arrival order do not contribute to the sum. Alternating lanes retain the
previous call's inputs until all peers have advanced beyond their consumers.

The HC head similarly stores each split-K projection in its own FP32 slot.
An ordered reduction adds the splits before the original gate, residual mix,
and BF16 conversion. The split-local matrix multiplication and RMS arithmetic
retain their original order.

HC-pre Sinkhorn column sums use the explicit four-row pairing
`(row0 + row1) + (row2 + row3)` in both implementations.

SWA, HCA, and CSA softmax use a compensated, range-reduced FP32 exponential for finite
max-subtracted scores in `[-80, 0]`, with the existing exponential outside
that interval. SWA/HCA use a 128-element denominator tree; CSA uses a
32-element tree. Each level pairs the two halves with FP32 additions. The independent
Torch reference evaluates the exponential in double precision before FP32
rounding and uses the same specified addition order. HCA/CSA compensate their
final FP32 quotient before the existing BF16 packing. This does not claim
correct rounding for every FP32 input or complete-forward agreement.

These rules make the reduction order explicit. They do not establish bitwise
equivalence across different hardware, compiler revisions, or other operations
that use different arithmetic order.

## Embedding window reuse

The embedding exchange clears its local notification counters after
readback. Its current callers use a fixed notify count of one. A retained
single window must be reused only after all ranks finish the previous
invocation; overlapping reuse is outside this contract.

## Complete-set Indexer ordering

When the number of visible compressed positions is no greater than `IDX_TOPK`,
all visible positions are selected in ascending position order, including the
original index offset. This gives single-query and grouped-query consumers the
same attention accumulation order for the complete set. Score computation and
cache updates are unchanged.

For more than `IDX_TOPK` visible positions, the original score-based selection
and tie handling remain in place. The validator requires exact positional
indices for complete sets and retains the score-aware comparator for selected
subsets. The boundary between these cases must be included when validating a
new request layout.

## Retired expert experiments

The model uses the planned persistent routed-expert kernel. The shared expert
occupies its own entry in that weight bank rather than a separate FFN chain.
Its fixture weight generator remains in `utils.py`.

An earlier scatter epilogue replaced the per-slot BF16 output and local gather
with FP32 atomic accumulation into token rows. The recorded A2/A3 TP8, T8
comparison used 20 rounds after three warmups and the fastest-rank mean:
244.5 us for the retained MoE path versus 269.0 us for scatter, across three
pairs. The standalone kernels measured 129.7 versus 136.8 us. These are
historical component measurements, not request TPOT or an isolated estimate
of scatter cost: task boundaries, initialization, stores, and rounding also
changed. The mechanism's behavior at larger token or route counts remains
unproven. The retired scatter and unused persistent A/B variants are omitted
from the runnable model; re-evaluate with real consumer inputs if that scope
changes.

## Execution shapes and component ownership

`config.py` holds model configuration and quantization constants.
`decode_shapes.py` owns the shared deployment shapes and physical cache pools;
context length, parallel rank count and pool scale must be positive.
validation start-position defaults live with the host utilities.

The attention-component part supplies `rope_interleave.py` and
`o_proj_grouped.py` together with their first Indexer/compressor and standalone
attention consumers. The former prepares query-token or compression-boundary RoPE tables;
QKV metadata preparation separately handles dynamic token tiles and its swap
indices.

## CI targets and validation scope

This model uses A2/A3 device CI. The standalone `lookup_embedding.py`,
`hc_head.py`, `hc_post.py`, `rmsnorm.py`, and `sample.py` entries also run in
simulator CI with `-p a2a3 --sim`, including their default golden comparisons.
The remaining runnable entries retain `# ci: no-sim` while simulator
qualification continues. Both pull-request and daily selectors honor these
markers.

The composed `decode_swa.py` and `decode_hca.py` entries require eight logical
TP ranks and declare `# ci: devices=8`, as does CSA. Their standalone
`decode_sparse_attn_swa.py` and `decode_sparse_attn_hca.py` entries use one
device and retain their original numerical references.

`decode_fwd.py` validates all 43 layers against an independent Torch
composition by default. It includes embedding, request-derived metadata,
2 SWA / 21 CSA / 20 HCA attention layers, 43 TP MoE layers, HC head,
final RMSNorm, vocabulary-sharded logits and sampling. All eight mutable
cache/state pools and the four public outputs are compared by the existing
harness. Source compilation and component CI do not qualify this full path.

The Torch composition reuses the child operator references. Expert gate/up/down
products accumulate INT8 operands as INT32 before applying original FP32
scales. Activation quantization uses explicit FP32 division for `127 / amax`;
the gate retains deferred RMS and its separate dequantization-scale order.
BF16 Cube consumers use K16 FP32 accumulation and their kernel's split-K
order in the CPU reference: Q/KV projections, sparse QK/PV, grouped output
projection A, compressor projections, indexer Hadamard and routing-weight
projections, and vocabulary logits. Batched rows and independent groups use
the same host helper; indexer routing weights retain four split-K partials.
Float64 evaluates each BF16 product group and each reference unary or division
before rounding its result to FP32. Model operands, outputs, INT8-to-INT32 dots
and scales retain their declared dtypes. HC-pre uses explicit balanced FP32
reductions for its square sums and four-way Sinkhorn sums; its Torch reference
spells out the stored exp/add/div boundaries instead of using a fused softmax.
The native HC-pre maps compensate their reciprocal, reciprocal-square-root and
division seeds. Exponential correction is bounded to `[-80, 80]`; the original
exponential is retained outside that interval. These bounded corrections do not
establish a general IEEE rounding guarantee for every input.
TP projection contributions include channel scales before ordered rank
reduction. Standalone grouped projections retain their own arithmetic.
These choices do not establish bitwise equivalence with device instructions;
the existing numerical comparisons remain required.

- Default execution and `--full` run the same complete-forward comparison.
  The generated bounded fixture starts at position 121 and uses one shared
  MoE bank unless `--moe-banks` is supplied. This is synthetic correctness
  coverage, not qualification of a real checkpoint or community equivalence.
- A real checkpoint comparison requires the actual weights and original FP32
  scales for all 43 layers (`--moe-banks 43`), matching inputs/state and the
  precision contract above. Shared random banks cannot stand in for them.
- `--input-data` replays complete `in/` and `out/` tensors, which take precedence
  over the locally computed Torch reference. Partial numerical snapshots fail.
- `--smoke` explicitly executes all 43 layers without numerical comparison.
  It permits generated inputs or input-only replay and reports no correctness
  verdict. It cannot be combined with saved expected outputs.
- `--compile-only` performs code generation and ABI checks without computing
  a Torch golden or executing the model.

The full TP reference checks that shared attention operands, metadata and
initial caches are replicated; mismatches are rejected explicitly. All three
attention kinds consume the actual projection shards; unused full-projection
arguments do not affect the reference. Every owner's MoE valid-row count is
consumed; counts may differ across owners and must lie in `[0, T]`. The standalone MoE reference
also computes each rank's HC/routing operands separately.

A full-model Torch comparison establishes agreement with the implemented
native precision contract for its supplied fixture. An independent community
snapshot additionally requires alignment of the community numerical policy;
matching INT8 checkpoint names or tensor dtype labels is insufficient.
Existing operator/layer validators retain their own scope. No complete-model
numerical PASS is implied by providing this reference or by host-only checks.

Use `--seed` for reproducible generated inputs and `--save-data` to save both
inputs and expected outputs. For example:

```bash
# Complete 43-layer numerical fixture; 16384 is a supported Indexer context.
python models/deepseek_v4_flash_lowlat/decode_fwd.py -p a2a3 --max-seq-len 16384
# Generate the complete program without computing a Torch reference.
python models/deepseek_v4_flash_lowlat/decode_fwd.py -p a2a3 --max-seq-len 16384 --compile-only
# Explicit execution-only smoke, without a correctness verdict.
python models/deepseek_v4_flash_lowlat/decode_fwd.py -p a2a3 --max-seq-len 16384 --smoke
# Replay a complete snapshot with matching checkpoint and execution layout.
python models/deepseek_v4_flash_lowlat/decode_fwd.py -p a2a3 --max-seq-len 16384 --moe-banks 43 --input-data <reference>/data
```

## Validation

HC pre and HC post use a dynamic token axis and bound their final tiles to the
remaining rows. Their default entries check the configured width plus 1, 6, 9,
and 17 rows. `--tokens` selects one width. HC pre saved-golden replay defaults
to the configured width unless `--tokens` is supplied.

The component part's grouped projection uses a positive static token count
divisible by 8 and at most 16. QKV projection requires
the attention head count to divide into head groups, with each local head
count divisible by its four-head RoPE tile. Invalid configurations fail
before compilation.

The output-projection L2 eviction option is diagnostic. Its large buffer is
allocated only with `--evict-l2`; ordinary correctness runs retain a small
placeholder and perform no eviction reads.

Use the operator entry points and their original Torch references:

```bash
python models/deepseek_v4_flash_lowlat/hc_head.py --help
python models/deepseek_v4_flash_lowlat/o_proj_tp.py --help
```

`o_proj_tp.py` uses the full eight-rank communication contract. Its default
correctness entry runs two distinct inputs at epochs 1 and 2 in one dispatch,
checks both outputs against the original reference, and clears signals only
after the second epoch. `--single-epoch` retains the original one-epoch entry
and saved-fixture interface. The validation handoff is not a performance
measurement. Local operators
retain deployment shapes when run on one device. Source compilation is only a
preparation step: numerical checks and performance comparisons must use the
resulting binaries, original consumers, and complete relevant request paths.

See [precision tuning](../../debug-and-tune/precision-tuning.md) and the
[compile/runtime workflow](../../run-and-validate/compile-runtime-workflow.md)
for validation and artifact handling.

### Ratio-4 verification-step boundaries

The main and indexer compressors pool every boundary with a nonnegative compressed-cache slot mapping. `kv[b, o]` contains boundary ordinal `o` within the step: row zero remains the first boundary. Unmapped ordinals and remaining rows have no output write contract. With eight tokens, starts 0, 4 and 121 produce boundaries 3/7, 7/11 and 123/127 respectively when both cache slots are admitted.

Pooling consumes the current step's projected rows directly, including speculative rows without persistent state slots, and reads earlier prefix rows through the state block table. All pools precede persistent scatter; only nonnegative state slot mappings authorize that scatter. An early advance recomputes remaining tokens from the new step inputs and the committed prefix, without promoting rejected tokens into persistent state.

Indexer query RoPE tables have one row per absolute query-token position. Compression RoPE tables are separate, indexed by boundary ordinal and aligned to each four-token window start. For a step starting at 8193, the first boundary 8195 uses compression position 8192; its query still uses its own absolute position. Standalone half-width tables are flattened `[B*S, rope_dim/2]`; the indexer additionally takes `inner_cos` and `inner_sin` for aligned compression windows. CSA derives both tables from its existing full RoPE inputs.
