# Experimental prefill O-projection B dispatch batching

The prefill-only `--prefill-oproj-b-batched` build option launches independent
(output slab, group) INT8 projection blocks in a single SPMD grid. The
original path submits one single-block task per slab and group. Group
quantization must finish before the batch starts, and the unchanged ordered
output epilogue waits for the entire batch.

Both paths call the same per-block compute body. The M128, N256 and K256
fragments, INT32 accumulation, scale application, group order and BF16
rounding are unchanged. Each block owns a disjoint output slab in the
existing partial tensor. No native decode implementation is changed.

The option defaults off. Source and CPU checks are separate from native,
device numerical and ordinary performance qualification. No speedup is
claimed before the full request comparison passes.
