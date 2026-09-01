# DeepSeek V4-Flash low-latency decode

This directory contains batch-one attention and MoE components for eight
logical ranks. They build on the configuration and primitives in the same
directory; the composed forward is supplied separately. The default
verification width is eight tokens. This is experimental code, not a
`pypto-serving` deployment contract.

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
attention consumers. The former prepares batch-indexed half-width RoPE tables;
QKV metadata preparation separately handles dynamic token tiles and its swap
indices.

## CI targets and validation scope

This model currently uses A2/A3 device CI. Its runnable entries are marked
`# ci: no-sim`, which both pull-request and daily selectors honor. Simulator
qualification is deferred; this does not assert simulator incompatibility.
Other model families and their simulator coverage are unchanged.

The composed `decode_swa.py` and `decode_hca.py` entries require eight logical
TP ranks and declare `# ci: devices=8`, as does CSA. Their standalone
`decode_sparse_attn_swa.py` and `decode_sparse_attn_hca.py` entries use one
device and retain their original numerical references.

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
