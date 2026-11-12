# DeepSeek V4-Flash low-latency decode

This model directory provides the batch-one decode path with two physical TP
ranks by default. The verification width remains eight tokens, and the eight
attention projection groups are assigned four per rank. It is an experimental
decode implementation, not a `pypto-serving` deployment contract.

## Precision contract

The native HC stream uses FP32 at `hc_pre` input and `hc_post` residual/output,
including the public `pre_hc_hidden_out` output. Dense, attention and expert
local tensors also use BF16; this is a mixed-precision contract.

The independent Torch references use standard FP32 matrix multiplication
with BF16-rounded operands where required, ascending split-K reductions, and
per-group projection scales. Model operands and outputs keep their declared
dtypes. Torch and device accumulation order may differ; numerical comparisons
do not assume bitwise equivalence.

Standalone Q validation checks its produced head group; the other pure-Out
heads have no initialized-value guarantee.

The FP32 gate uses deferred RMS: `xg = x_mixed.float() * norm_w.float()`.
It applies `inv_rms` after projection and includes it in the dequantization
scale for INT8 `quant(xg)`, without BF16-rounding a normalized routing vector.
Gate activation quantization uses FP32 division before round-to-nearest-even
conversion. The router uses the device's FP32 exponential, logarithm, square
root and division operations; the routed expert uses the same fast arithmetic
for activation and requantization. The Torch reference can differ in rounding,
so the existing numerical gates remain applicable.

The attention and final RMSNorm references reduce FP32 squared inputs in
FP64, then round the sum once to FP32. The mean and epsilon addition remain
FP32; the inverse square root is evaluated in FP64 and rounded to FP32.
This specifies the reference's statistical precision without changing epsilon,
activation dtypes, or the numerical comparison thresholds.

The fast HC, RMSNorm, QKV and attention paths use the device's FP32 reduction,
reciprocal, and exponential operations without the reference branch's full
product-error and transcendental compensation. Their results can differ from
the Torch reference and must be checked with the output-specific golden gates.
Finite input tests do not establish all-domain correctly rounded arithmetic or
a complete-model performance gain.

A previous HC-pre component comparison used three captured layer inputs at
six-token width on four dual-chip 910C boards. Its ordinary submit-to-result
median was 0.898 ms for the older arithmetic and 1.034 ms for the compensated
path. A separate instrumented capture measured the Sinkhorn kernel at 13.93
and 123.21 microseconds, respectively. This supports an implementation-cost
finding for that component and input scope; it is not a measurement of this
commit's complete-request TPOT.

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
The current FP32 storage policy is retained. A community-precision
compatibility mode is not implemented.

## Full-forward numerical checks

The independent Torch forward remains the reference. Each owner and token row
of `pre_hc_hidden_out`, `hidden_out`, and active `logits` must have relative L2
error at most 0.20. The reference norm has a per-element RMS floor of `1e-3`;
this prevents a near-zero row from making the relative denominator vanish.
Each checked row also caps its largest absolute error at eight times the L2
limit times the reference RMS, plus `1e-3`, so a large isolated outlier cannot
hide in a long vector. Written cache rows use 16 times their L2 limit; logits
use four times the L2 limit.
Inactive logits retain the strict `rtol=atol=1e-3` check, and nonfinite output
values fail.

Paged caches use the actual request metadata to select the rows the forward can
write. Every unwritten cache byte must match the reference exactly. Each written
row is checked separately for every owner and layer: the relative L2 limit is
0.15 for KV and indexer cache values and 0.30 for compressor state. Indexer
INT8 cache values are compared after applying their corresponding dequantization
scales, and the scales are checked separately. Sentinel values must match.

Sampled IDs must match sampling from the actual logits. A different ID from the
Torch reference is reported, since close logits can change their ordering. The
L2 limits are experimental coarse forward checks chosen for this fast W8A8
path; in particular, the 0.30 compressor-state limit is not a claim of 30%
accuracy or an official DSpark tolerance. These checks do not establish output
quality on SpeedBench, MT-Bench, long contexts, or other workloads.
The full-forward command uses up to 32 CPU threads for the Torch reference
unless `PYPTO_GOLDEN_NUM_THREADS` is explicitly set.
PR and daily device CI run the default 43-layer forward execution smoke. It
does not compute or compare a golden. Use `--test-layers 5` for an explicit
five-layer smoke; that result is not full-model qualification.

The standalone CSA fixture permits at most 5% of points to exceed `3e-3` under
`ratio_reldiff`'s combined absolute/relative rule and retains its worst-point cap
of 1. The 5% allowance also appears in the MoE layer fixture. Two unseeded CI
fixtures had 0.8064% and 2.4902% bad points; their inputs were not saved, so the
reason for that variation remains unresolved. This is a fixture tolerance, not
an official model-quality threshold. Generated CSA fixture inputs use seed 0 by
default and can be varied with `--seed`.

## Reproducible reduction order

The output projection publishes each logical group's FP32 contribution into a
separate slot. Each of the two alternating lanes contains `O_GROUPS * T_PAD`
rows. A physical rank announces completion after publishing every local group.
After the all-peer completion barrier, the consumer adds groups `0..O_GROUPS-1`
to a positive-zero accumulator and rounds to BF16 once.
Every producer overwrites its valid rows before publication; allocator contents
and arrival order do not contribute to the sum. Alternating lanes retain the
previous call's inputs until all peers have advanced beyond their consumers.

The HC head similarly stores each split-K projection in its own FP32 slot.
An ordered reduction adds the splits before the original gate, residual mix,
and BF16 conversion. The split-local matrix multiplication and RMS arithmetic
retain their original order.

HC-pre Sinkhorn column sums use the explicit four-row pairing
`(row0 + row1) + (row2 + row3)` in both implementations.

The fast SWA, HCA, and CSA softmax paths use the device exponential and row
reduction operations. Their reduction order and rounding can differ from the
independent Torch reference. The output-specific numerical checks cover that
difference rather than assuming bitwise equality.

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

Each composed attention entry calls QKV once for its rank's contiguous head
span. Q-LoRA projection, normalization and quantization, and KV projection and
RoPE run once per rank; Q projection and per-head normalization and RoPE cover
all owned heads. With TP2 this span contains 32 heads. The standalone QKV entry
continues to exercise one eight-head group.

## CI targets and validation scope

This model uses A2/A3 device CI with at most two devices per entry. Sixteen
standalone entries use one device; nine distributed entries use TP2 by default
and declare `# ci: devices=2`. Only the multi-layer `decode_fwd.py` and
`decode_layer.py` entries retain `# ci: no-sim`. The other twenty-three entries
run with `-p a2a3 --sim`, including their original golden and state comparisons.
Both pull-request and daily selectors honor these markers.

PR and daily device sweeps use the same runner, command defaults, logging
environment, disabled benchmarking, 3600-second queue wait and 900-second
per-entry execution budget. Entry device counts remain declared by
`# ci: devices` markers.

The composed `decode_swa.py`, `decode_hca.py`, and `decode_csa.py` entries use
two physical TP ranks by default and declare `# ci: devices=2`. Each rank
processes its consecutive head groups. CSA assigns four logical tokens per
rank and exchanges their query and output rows before projection. Their standalone
`decode_sparse_attn_swa.py` and `decode_sparse_attn_hca.py` entries use one
device and retain their original numerical references.

Projection weights are resident group-major banks: `wo_a_shard` has shape
`[N_RANKS, GROUPS_PER_RANK, O_LORA, O_GROUP_IN]` and `wo_b_shard` has shape
`[N_RANKS, GROUPS_PER_RANK, D, O_LORA]`. Each per-group matrix is contiguous.
Regenerate saved projection, layer and forward fixtures after this layout
change, including fixtures that retain an explicit TP8 configuration.
The native TP8 request profiles retain their existing flat projection-weight
inputs. Their attention call sites create singleton group views without
copying or changing the public request parameter shapes.

`decode_fwd.py` executes all 43 layers without computing or comparing a golden
by default, in both PR and daily CI. Use `--test-layers 5` for an explicit
five-layer smoke. Use `--full` to opt into the independent Torch comparison
or a complete saved reference. It includes embedding, request-derived metadata,
2 SWA / 21 CSA / 20 HCA attention layers, 43 TP MoE layers, HC head,
final RMSNorm, vocabulary-sharded logits and sampling. All eight mutable
cache/state pools and the four public outputs are compared by the existing
harness in `--full` mode. A default smoke PASS confirms execution only;
source compilation and component CI do not qualify this full numerical path.

The Torch composition reuses the child operator references. Expert gate/up/down
products accumulate INT8 operands as INT32 before applying original FP32
scales. Activation quantization uses explicit FP32 division for `127 / amax`;
the gate retains deferred RMS and its separate dequantization-scale order.
BF16 projection references round operands to BF16 and use FP32 `torch.matmul`;
the helper accumulates split-K partials in ascending order. Some operator
references, including HC-pre and RMSNorm, separately use FP64 intermediates
for their specified reductions or unary operations. The fast native HC-pre uses
the device's ordinary reciprocal, reciprocal-square-root and exponential paths.
Model operands, outputs, INT8-to-INT32 dots and scales retain their declared
dtypes. These choices do not guarantee bitwise equality across implementations.
TP projection contributions include channel scales before ordered rank
reduction. Standalone grouped projections retain their own arithmetic.
These choices do not establish bitwise equivalence with device instructions;
the existing numerical comparisons remain required.

The LM-head Torch reference projects at most two independent owners per
batch. Each owner retains its row selection, BF16-rounded operands and FP32
matrix multiplication, zero rows, and sampling inputs. Device kernels and
comparison thresholds are unchanged.

The HCA Torch reference skips matrix products for fully masked sparse tiles
and materializes their original neutral statistics: a `NEG_INF` maximum and
positive-zero denominator and numerator. Every block remains in its original
softmax merge position. Valid tiles retain their existing arithmetic.
These optimizations affect host-side reference work only; device kernels and
comparison thresholds are unchanged.

- Default execution and `--smoke` execute all 43 layers without a numerical
  comparison. They permit generated inputs or input-only replay and report
  no correctness verdict. Saved expected outputs require explicit `--full`.
- `--full` runs the complete-forward comparison. The generated bounded fixture
  starts at position 121 and uses one shared MoE bank unless `--moe-banks` is
  supplied. This is synthetic correctness coverage, not qualification of a
  real checkpoint or community equivalence.
- A real checkpoint comparison requires the actual weights and original FP32
  scales for all 43 layers (`--moe-banks 43`), matching inputs/state and the
  precision contract above. Shared random banks cannot stand in for them.
- `--full --input-data` replays complete `in/` and `out/` tensors, which take
  precedence over the locally computed Torch reference. Partial numerical
  snapshots fail. Without `--full`, replay accepts input-only snapshots.
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

Use `--seed` for reproducible generated inputs and `--save-data` to save inputs
for replay. In `--full` mode, it also saves expected outputs. For example:

```bash
# Complete 43-layer numerical fixture; 16384 is a supported Indexer context.
python models/deepseek_v4_flash_lowlat/decode_fwd.py -p a2a3 --max-seq-len 16384 --full
# Generate the complete program without computing a Torch reference.
python models/deepseek_v4_flash_lowlat/decode_fwd.py -p a2a3 --max-seq-len 16384 --compile-only
# Default execution-only smoke, without a correctness verdict.
python models/deepseek_v4_flash_lowlat/decode_fwd.py -p a2a3 --max-seq-len 16384
# Replay a complete snapshot with matching checkpoint and execution layout.
python models/deepseek_v4_flash_lowlat/decode_fwd.py -p a2a3 --max-seq-len 16384 --moe-banks 43 --full --input-data <reference>/data
```

## Validation

The persistent expert retains the guarded native TP8 single-tile schedule
when the complete intermediate shard fits in one tile. Wider shards use the
runtime's available cluster count and synchronized start. Gate/up projections
and SwiGLU use bounded column tiles; activation quantization reduces the
maximum across the entire local intermediate shard before quantizing any
tile. TP2 uses all 1024 intermediate columns, retaining the main branch's
per-element arithmetic and comparison thresholds. The default standalone
entry validates this TP2 split on the A2/A3 simulator.

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

`o_proj_tp.py` assigns all eight logical groups to the physical TP ranks. Its default
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

The main and indexer compressors write each boundary with a nonnegative
compressed-cache slot mapping at its input-token index: `kv[b, s]`, or
`kv_flat[b * S + s]`, corresponds to `position_ids[b, s]`. Non-boundary
tokens and boundaries with unmapped cache slots have no output write
contract. With eight tokens, starts 0, 4 and 121 produce absolute boundaries
3/7, 7/11 and 123/127 respectively. When both cache slots are admitted, the
output rows within the step are 3/7, 3/7 and 2/6, respectively.

Pooling consumes the current step's projected rows directly, including speculative rows without persistent state slots, and reads earlier prefix rows through the state block table. All pools precede persistent scatter; only nonnegative state slot mappings authorize that scatter. An early advance recomputes remaining tokens from the new step inputs and the committed prefix, without promoting rejected tokens into persistent state.

Indexer query RoPE tables use each token's absolute query position.
Compression RoPE tables have one row per input token, using the same
token index as the output. `prepare_csa_metadata` selects compression
position `position - COMPRESS_RATIO + 1` for boundary tokens and position
zero for non-boundary placeholders. For a step starting at 8193, boundary
8195 uses compression position 8192 at token index 2; its query uses
absolute position 8195. Standalone fixtures follow the same per-token rule,
with half-width tables flattened to `[B*S, rope_dim/2]`. The indexer
additionally takes `inner_cos` and `inner_sin` for the aligned compression
windows. CSA derives query and compression tables from its existing full
RoPE inputs.

## Batch-one DSpark request serving

### Prefill

`prefill_fwd.load_prefill_programs()` loads one 43-layer chunk program with
CP8 attention, TP8 MoE intermediate shards, and a TP8 vocabulary head. Each
rank holds the complete chunk's HC input and owns a consecutive eighth of its
attention queries. Bind the decode Worker's existing TP8 weight banks, original
quantization scales, and cache allocations. The chunk entry reshapes contiguous
layer banks through borrowed device views and never uploads another weight set.
Query LoRA, query expansion, indexer-gate and vocabulary projections keep the
decode banks' ND storage. The isolated import selects the corresponding shared
operator annotations; callers must not repack these resident banks into NZ.
`load_prefill_programs(mode="serial")` loads the six component entries for the
same TP8 numerical baseline.

`prefill_cache.prefill_cache_views` borrows each owning decode cache buffer
through that Worker's `view_tensor` API. The cache bytes remain in place:
prefill uses 32-row raw-KV pages and 2-row ratio-4 state pages, while decode
views the same rows as 128-row and 4-row pages. `prefill_block_tables` splits
and pads the scheduler's decode tables, and `prefill_slot_mappings` derives
physical rows for each active position. The caller retains both the owning
buffers and the prefill views until all dependent dispatches complete; only
the original owning allocations may be freed. `initialize_cache_pages`
initializes newly owned pages, including the compressor score sentinel,
without clearing other requests' or shared-prefix pages.

`prefill_metadata` packs each prompt chunk and its RoPE rows for the CP8 group.
Physical page IDs may differ between ranks; each rank's table is converted
against its own cache allocation. Pass the existing `block_counts` when the
compressor tables use the decode ring layout.
The caller must provide enough non-aliased cache pages for the whole chunk and
the preceding sliding window, and preserve the page mappings and stored values
of live history across chunks. The metadata helper rejects missing pages,
out-of-bounds writes, and aliased HCA history. The HCA state must also
provide distinct rows for the kernel's 128-row history plus one DSpark decode
transaction when a prefill tile publishes that many rows. Prefix checkpoints
are not requested by this helper. Host tensors passed directly to a prepared
Worker submission must use shared storage allocated before the Worker starts.
For cache initialization, the distributed Worker's `copy_to` stages
non-inherited CPU buffers.

For single-machine prefill and decode, the caller uses the same rank-local
cache allocations and waits for each prefill dispatch before dependent decode
work. The scheduler and request lifetime remain the serving caller's job.
Prepare both programs on one Worker using `extra_compiled`, retain the owning
decode cache tensors, and pass their borrowed views to prefill. After terminal
prefill completes, use its sampled token as the next decode input at the prompt
length, with `kv_seq_lens` including that input. Decode uses the original block
tables and counts, including compressor state; it does not use the expanded
prefill page tables or reinitialize the retained cache pages. The decode RoPE
capacity must cover the prompt and all subsequent generated positions. Allocate
decode metadata, scratch and any retained ABI buffers during Worker setup,
before prefill. Borrowed views require no physical layout conversion; first-use
program/domain preparation and normal dispatch synchronization remain separate
latency costs.
Registering a decode program with `extra_compiled` does not eagerly create its
communication domains. A caller that needs a warm first decode can run a startup
invocation on separately initialized scratch caches before prefill, retaining the
same prepared Worker. That invocation must not bind the real request caches.
The first real decode then reuses the domain; normal per-dispatch buffer resets,
input staging and completion fences still contribute to latency.

`prefill_fwd.prepare_prefill_bindings` combines the caller's resident decode-format
weights, input metadata and scratch with the cache views. `submit_prefill`
orders the bindings using the actual generated parameter metadata. The chunk
program submits all layers once and waits once before returning. Its final HC
output aliases the supplied `x_hc`; `x_hc_next` is scratch for the serial
baseline only. The `terminal` scalar runs the learned head and sampling only
on the final chunk. The caller supplies the original sampling, grammar,
allocation and request-lifetime inputs, and consumes its outputs. Submit chunks
in order, retaining the owning allocations through the final completion.

CSA coordinates 128-query indexer blocks with two bounded gather slots.
Gather waits for that block's exact top-512 selection and the previous consumer
of its slot. QK/PV waits for its gather and the preceding attention block.
The existing 512-row output-projection tile and all attention arithmetic remain
in their owning operators. The indexer score/pair arenas and the two KV slots
are reused across blocks; their lifetime ends with the layer's device scope.
This permits subsequent index scoring and gather to become ready before the
preceding attention finishes. Generated dependencies establish eligibility;
actual overlap and wall-time benefit still require device measurements.

Generate the chunk artifact without a Worker using:

```bash
python -m models.deepseek_v4_flash_lowlat.build \
  --profile prefill --cache build_output/prefill-cache \
  --report ../experiments/lowlat-prefill/pr1747/builds/prefill-generated.json
```

Keep experiment scripts, case manifests, datasets and timing reports under the
workspace's `experiments/` directory. `build_output/` owns generated compiler
artifacts and JIT caches. Experimental callers select explicit AgentX trace and
request identities; input length, padding and chunk boundaries must come from
the actual reconstructed input, with no fixed short-prompt fallback.

The report records chunk/tile sizes, the exact-prefetch mode, two slots, and
the retained eight projection replicas. Code generation, native compilation,
and host binding checks alone do not establish device equivalence. A reconstructed
AgentX request of 16,384 tokens passed full-output and cache byte-digest checks
against the frozen TP8 component baseline. On the same Worker, 64 subsequent
single-query decode steps retained the original weight/cache allocations and
matched a copied-cache control for sampled IDs, logits, hidden states and final
cache contents. This qualifies the tested allocation handoff; longer contexts
and complete serving sessions remain unvalidated. Histogram/EMA early prefetch, projection sharing across
ranks, and tile-specific cache publication dependencies are not implemented.
Their configuration must not be advertised as available or selected from
unmeasured cost estimates.


### Deployment

This deployment retains five draft tokens and six target verification rows.
It complements the [wide-batch DSpark model](../deepseek_v4_flash_dspark/index.md).

The source includes the batch-one drafter, Markov sampling, six-row target,
device request-state transitions, two-frame host policy and native build. A
complete request benchmark requires an external caller and actual checkpoint
weights. Clean-checkout execution and performance qualification are still
pending. No historical TPOT is claimed for this revision.

The target retains rank-ordered output projection, ordered HC-head split
reduction, and ascending complete-set Indexer selection. Target RMS uses the
fast FP32 path, including the gate's RMS and routing arithmetic.
Single-query and verification entries share these operators. Full request
numerical and performance qualification remain separate from component
validation; failures remain visible in either stage.

## Component boundary

- `drafter/entry.py` exports the three-layer drafter with TP4/CP4/EP4 geometry,
  a 128-row target-history window, and five query rows. Only the four drafter
  ranks execute it; they are a subset of the eight target ranks.
- `markov/entry.py` exports the target LM-head and sequential five-step Markov
  proposal/confidence path with four vocabulary shards.
- `drafter/device_metadata.py` selects the actual absolute-position metadata bank from
  committed state. A communication epoch is not a metadata-bank index.
- `target_bridge.py` broadcasts the seed token and five proposals with an epoch
  tag before the target can consume them.
- `target_commit.py` accepts the longest greedy-matching draft prefix and emits
  the target bonus. The original control fields distinguish logical advance
  from the number of tokens still permitted by the output budget.
- `device_control.py` materializes target metadata after the prior-cycle gate;
  `target_state.py` publishes the next state after the real history commit
  acknowledgement. `phase_reuse.py` supplies the peer-only eight-rank boundary
  for reuse of communication slots within a two-cycle parent.

The first three request-state fields are the monotonic epoch, committed anchor,
and remaining output budget. Field three stores the last delivered token.
History has 128 live rows in a 136-row owning allocation. Target features are
three concatenated 4096-wide HC means from layers 40, 41 and 42, not the final
learned HC head.

`native_pipeline.native_cycles` composes D/M on ranks 0–3 and target verification
on all eight ranks. `native_rsag_pipeline` selects the ordered MoE
reduce-scatter/allgather composition. Its target package shares unchanged kernels
with the baseline package. The `cycle_count` constexpr selects a one- or two-cycle
parent in source; it does not require editing generated host code.

The compact single-query entry is `native_single.py`; request cycles use
`native_pipeline.py` and `native_rsag_pipeline.py`, which call the corresponding
target verification modules. They share this model directory's operator files
and 43 layer-specific weight banks. Each build freezes its query width in a fresh
process. Request benchmarks require actual checkpoint weights.

The drafter and Markov packages isolate their import-time deployment arguments
and restore the caller's arguments in `finally`. Their kernel bodies and active
constants are carried forward from the batch-one implementation; unused
standalone fixtures and prefill modules are omitted. These packages have not
been substituted with the wide-batch deployment defaults.

## Streaming and termination

`stream_controller.run_request` uses two independently owned output frames and
at most two accepted parent handles. A frame is reusable only after successful
whole-parent completion, packet validation, delivery or EOS discard, and release.
It never performs a device read or frees an in-flight frame itself.

A single-cycle parent is used when fewer than seven tokens remain. A second
two-cycle parent can be accepted ahead only when at least 24 tokens remain,
so its reservation survives the predecessor's maximum twelve-token advance.
This controls preparation ahead of execution; safe execution also requires the
real runtime launch-depth-one parent retirement boundary and per-run HOST output
publication. FIFO order alone is insufficient.

The controller supports a maximum of 2048 generated tokens including the first
prefill prediction. It stops delivery at a real tokenizer EOS. An already
accepted successor is still waited on, captured and validated, then discarded;
no replacement is submitted. Exceptions remain the caller's responsibility to
persist and drain during normal Worker cleanup.

## Source generation

From the repository root, in the configured PyPTO environment:

```bash
python -m models.deepseek_v4_flash_lowlat.build \
  --profile rsag --devices 0,1,2,3,4,5,6,7 \
  --cache build_output/dspark-bs1-cache \
  --report build_output/dspark-bs1-rsag-generated.json
```

The other profiles are `single`, `one`, and `two`. This command creates generated
artifacts only: it does not load weights, construct a Worker, allocate NPU memory,
or measure performance. The eight device IDs describe logical compile topology;
runtime board/chip mapping and capacity admission are separate.
Each PTOAS subprocess has a default 180-second deadline for vector code
generation. `--ptoas-timeout-seconds` accepts a value from 1 to 600 and
records it in the generated report. This scoped build policy preserves the
backend's assembler command, version checks and errors, and restores its
binding on exit; it does not modify the installed toolchain.

The native W2 prefetch implementation is a tracked C++ Cube-body fragment.
`native_source` selects it before publication only after the complete generated
Cube, shared declarations, ABI and exports match the reviewed fingerprint. The
freshly generated Vector arithmetic is preserved. Both original and selected
sources remain in the artifact report, and cache hits replay the same graft
validation. The fragment and selector digest is a JIT constexpr key input, and
the fragment is included in the model source manifest. Normal native compilation
is still required; no retained binary is substituted for changed code.

## Runtime and benchmark scope

The source-generation entry produces the model artifacts described above.
Complete-request execution, checkpoint preparation, and performance
measurement require an external caller and data; experimental tooling
for those tasks is not part of this model package.

## Reproduction and numerical status

Performance reproduction and numerical qualification are separate. The planned
benchmark retains batch size one, concurrency one, real EOS or length termination,
initialization/upload/warmup accounting, per-request output IDs, and request-wall
TPOT. Kernel duration divided by a presumed acceptance count is not TPOT.

The merged target model's numerical checks do not establish the accuracy of
this composition. A later complete entry must retain the existing packet,
commit, history-ring, final-state and complete-output checks. Until that entry
is built and measured from a clean checkout, both full-pipeline reproduction and
numerical qualification remain pending.

## Fast arithmetic delivery

This branch restores the pre-alignment hardware arithmetic in target HC,
RMS, QKV and sparse attention. It keeps W8A8 tensors, the existing public
interfaces, request scheduling, ordered collectives, buffer padding and
ownership checks. Golden functions remain enabled. The full-forward floating
outputs use the bounded checks described above; other component gates are
unchanged.

Fast arithmetic does not guarantee bitwise agreement with the refined
reference arithmetic. FP32 reductions and transcendental approximations can
change greedy token choices and EOS positions. Model-quality scores and
whole-request speed for this port must be evaluated separately; a historical
measurement is not a measurement of the current commit.

The validated reference-arithmetic source is preserved on
[the reference branch](https://github.com/zzjc1234/pypto-lib/tree/feat/dsv4-dspark-bs1-reference).
